"""
classifier.py – classifieur de cases (ResNet-18) intégré au projet.

Remplace la dépendance fantôme `ChessToFEN.chessClassifier`, qui n'était ni
dans le dépôt, ni sur PyPI, ni dans requirements.txt et empêchait purement et
simplement le lancement de l'application.

Le modèle attendu est `resnet18_chess.pt` (state_dict PyTorch, 13 classes),
produit par `train_resnet.py`.

Convention de nommage des classes (héritée de `dataset/`) :
    <piece>_dark  → pièce **noire**      (ex. rook_dark  = tour noire)
    <piece>_light → pièce **blanche**    (ex. rook_light = tour blanche)
    empty         → case vide
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from paths import CLASSES_PATH, DATASET_DIR, MODEL_PATH

log = logging.getLogger(__name__)

# Ordre canonique : identique à ce que produit `torchvision.datasets.ImageFolder`
# sur `dataset/`, c.-à-d. le tri alphabétique des sous-répertoires.
DEFAULT_CLASSES: tuple[str, ...] = (
    "bishop_dark", "bishop_light",
    "empty",
    "king_dark", "king_light",
    "knight_dark", "knight_light",
    "pawn_dark", "pawn_light",
    "queen_dark", "queen_light",
    "rook_dark", "rook_light",
)

# classe → caractère FEN (majuscule = blanc, minuscule = noir)
CLASS_TO_FEN: dict[str, str | None] = {
    "empty": None,
    "pawn_light": "P", "knight_light": "N", "bishop_light": "B",
    "rook_light": "R", "queen_light": "Q", "king_light": "K",
    "pawn_dark": "p", "knight_dark": "n", "bishop_dark": "b",
    "rook_dark": "r", "queen_dark": "q", "king_dark": "k",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

# ── plus-proche-voisin sur dataset/ ────────────────────────────────
# Sur un même site, une pièce est un icône rendu quasi à l'identique à
# chaque capture (aux antialiasing/surlignages près) : comparer une case à
# ce qui a déjà été vu est bien plus fiable qu'un réseau entraîné sur
# quelques dizaines d'exemples. Validé en croisée sur dataset/ (171
# images) : 99,4 % de bonnes réponses, similarité ≥ 0,984 pour toute bonne
# correspondance, contre ≤ 0,86 pour une case qui ne ressemble à rien de
# connu (ex. une pop-up qui recouvre le plateau).
TEMPLATE_SIZE = 48
TEMPLATE_TRUST = 0.93   # au-delà : on fait confiance au gabarit sans consulter le réseau


def _template_vector(im: Image.Image) -> np.ndarray:
    small = im.convert("RGB").resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


class TemplateBank:
    """
    Banque de gabarits construite à partir de `dataset/` : une image par
    fichier, classée par le nom de son dossier parent.  `best_matches`
    renvoie, pour chaque case, la classe et la similarité cosinus de son
    gabarit le plus proche.
    """

    def __init__(self, dataset_dir=DATASET_DIR) -> None:
        labels: list[str] = []
        vectors: list[np.ndarray] = []
        if dataset_dir.is_dir():
            for class_dir in sorted(dataset_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for img_path in sorted(class_dir.iterdir()):
                    try:
                        with Image.open(img_path) as im:
                            vectors.append(_template_vector(im))
                    except OSError:
                        continue
                    labels.append(class_dir.name)
        self._labels = labels
        self._matrix = (np.stack(vectors) if vectors
                        else np.zeros((0, TEMPLATE_SIZE * TEMPLATE_SIZE * 3), dtype=np.float32))

    @property
    def available(self) -> bool:
        return len(self._labels) > 0

    def topk_matches(self, images: Sequence[Image.Image],
                     k: int = 2) -> list[list[tuple[str, float]]]:
        """→ les `k` gabarits les plus proches (classe, similarité), par image."""
        if not self.available:
            return [[("empty", 0.0)] * k for _ in images]
        k = min(k, len(self._labels))
        results = []
        for im in images:
            sims = self._matrix @ _template_vector(im)
            idx = np.argsort(sims)[::-1][:k]
            results.append([(self._labels[i], float(sims[i])) for i in idx])
        return results


class ClassifierError(RuntimeError):
    """Modèle absent, illisible ou incompatible."""


def load_class_names() -> list[str]:
    """
    Ordre des classes, par priorité :
      1. `classes.json` écrit par train_resnet.py (source de vérité) ;
      2. le tri alphabétique de `dataset/` ;
      3. la constante DEFAULT_CLASSES.
    """
    if CLASSES_PATH.exists():
        try:
            with open(CLASSES_PATH, encoding="utf-8") as fh:
                names = json.load(fh)
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return names
            log.warning("classes.json malformé — ignoré")
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("classes.json illisible (%s) — ignoré", exc)

    if DATASET_DIR.is_dir():
        names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
        if names:
            return names

    return list(DEFAULT_CLASSES)


def build_model(num_classes: int, *, pretrained: bool = False) -> torch.nn.Module:
    """ResNet-18 dont la tête est redimensionnée à `num_classes`."""
    weights = "DEFAULT" if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    return net


def build_transform() -> transforms.Compose:
    """Prétraitement d'inférence — doit rester identique à l'entraînement."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PieceClassifier:
    """
    Classifieur des 64 cases.  Chargement paresseux du modèle (le premier
    import ne doit rien coûter) et inférence en **un seul batch**, ce qui est
    nettement plus rapide que 64 passes séparées.
    """

    def __init__(self,
                 model_path=MODEL_PATH,
                 device: str | None = None,
                 classes: Sequence[str] | None = None):
        self.model_path = model_path
        self.classes = list(classes) if classes else load_class_names()
        self.device = pick_device(device)
        self.transform = build_transform()
        self._net: torch.nn.Module | None = None
        self._lock = threading.Lock()
        self._templates: TemplateBank | None = None
        self._templates_lock = threading.Lock()

    # ── chargement ──────────────────────────────────────────────────
    def _ensure_loaded(self) -> torch.nn.Module:
        if self._net is not None:
            return self._net
        with self._lock:
            if self._net is not None:      # double-checked locking
                return self._net
            if not self.model_path.exists():
                raise ClassifierError(
                    f"Modèle introuvable : {self.model_path}\n"
                    "Entraînez-le avec « python train_resnet.py » ou récupérez "
                    "resnet18_chess.pt depuis les releases du projet."
                )
            try:
                state = torch.load(self.model_path, map_location="cpu",
                                   weights_only=True)
            except Exception as exc:                      # noqa: BLE001
                raise ClassifierError(
                    f"Modèle illisible ({self.model_path}) : {exc}"
                ) from exc

            # Le nombre de classes du checkpoint fait foi : si `dataset/` a
            # changé depuis l'entraînement, mieux vaut le savoir tout de suite.
            fc_bias = state.get("fc.bias")
            if fc_bias is None:
                raise ClassifierError(
                    "Checkpoint inattendu : clé « fc.bias » absente. "
                    "Attendu : un state_dict de torchvision.models.resnet18."
                )
            ckpt_classes = int(fc_bias.shape[0])
            if ckpt_classes != len(self.classes):
                raise ClassifierError(
                    f"Le modèle expose {ckpt_classes} classes mais "
                    f"{len(self.classes)} noms sont connus ({self.classes}). "
                    "Réentraînez le modèle ou corrigez classes.json."
                )

            net = build_model(ckpt_classes)
            net.load_state_dict(state)
            net.eval().to(self.device)
            self._net = net
            log.info("Modèle chargé : %s (%d classes, %s)",
                     self.model_path.name, ckpt_classes, self.device)
            return net

    def warmup(self) -> None:
        """Précharge le modèle (à appeler hors du chemin critique)."""
        self._ensure_loaded()

    def _ensure_templates(self) -> TemplateBank:
        if self._templates is None:
            with self._templates_lock:
                if self._templates is None:
                    self._templates = TemplateBank()
        return self._templates

    def reload_templates(self) -> None:
        """Recharge la banque de gabarits (après enrichissement de dataset/)."""
        with self._templates_lock:
            self._templates = TemplateBank()

    # ── inférence ───────────────────────────────────────────────────
    @torch.inference_mode()
    def _cnn_topk(self, images: list[Image.Image], k: int) -> list[list[tuple[str, float]]]:
        net = self._ensure_loaded()
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        logits = net(batch.to(self.device))
        probs = F.softmax(logits, dim=1)
        k = min(k, probs.shape[1])
        top_conf, top_idx = probs.topk(k, dim=1)
        return [
            [(self.classes[i], float(c)) for i, c in zip(idxs, confs)]
            for idxs, confs in zip(top_idx.tolist(), top_conf.tolist())
        ]

    def predict_topk(self, images: Iterable[Image.Image],
                     k: int = 2) -> list[list[tuple[str, float]]]:
        """
        → les `k` meilleures classes par image (les plus probables d'abord),
        en combinant deux approches complémentaires :

        * **plus-proche-voisin** sur les images déjà classées dans
          `dataset/` — sur un même site, une pièce est un icône rendu quasi
          à l'identique à chaque capture, donc quasi infaillible dès qu'une
          correspondance nette existe (validé à 99,4 % en croisée), et ne
          coûte qu'un produit matriciel numpy ;
        * le **réseau** (`resnet18_chess.pt`), nettement plus coûteux (passe
          avant PyTorch), qui prend le relais pour tout ce que `dataset/` ne
          couvre pas encore (nouveau thème, case surlignée d'une couleur
          inédite, ou plateau recouvert par un élément de l'interface — la
          similarité de gabarit chute nettement dans ce cas, ce qui alerte
          plutôt que de trancher au hasard).

        Le gabarit l'emporte dès que sa similarité dépasse `TEMPLATE_TRUST` ;
        le réseau n'est alors sollicité **que sur les cases restantes**,
        pas sur les 64 à chaque capture — l'essentiel du coût CPU d'une
        lecture une fois le thème du site bien couvert par `dataset/`.
        """
        images = list(images)
        if not images:
            return []

        bank = self._ensure_templates()
        if not bank.available:
            return [entries[:k] for entries in self._cnn_topk(images, k=max(k, 2))]

        tmpl_topk = bank.topk_matches(images, k=max(k, 4))
        results = [entries[:k] for entries in tmpl_topk]

        needs_cnn = [i for i, entries in enumerate(tmpl_topk) if entries[0][1] < TEMPLATE_TRUST]
        if needs_cnn:
            cnn_entries = self._cnn_topk([images[i] for i in needs_cnn], k=max(k, 2))
            for i, entries in zip(needs_cnn, cnn_entries):
                results[i] = entries[:k]
        return results

    def predict(self, images: Iterable[Image.Image]) -> tuple[list[str], list[float]]:
        """
        → (labels, confiances) pour la séquence d'images fournie.
        Les confiances sont dans [0, 1] (probabilité softmax du réseau, ou
        similarité cosinus au gabarit le plus proche — voir `predict_topk`).
        """
        images = list(images)
        if not images:
            return [], []
        topk = self.predict_topk(images, k=1)
        labels = [entries[0][0] for entries in topk]
        conf = [entries[0][1] for entries in topk]
        return labels, conf

    def predict_board(self, squares: Iterable[Image.Image]
                      ) -> tuple[list[str | None], list[float]]:
        """
        Variante « échiquier » : renvoie directement les caractères FEN
        (None pour une case vide) au lieu des noms de classes.
        """
        labels, conf = self.predict(squares)
        return [CLASS_TO_FEN.get(lbl) for lbl in labels], conf

    def predict_board_topk(self, squares: Iterable[Image.Image],
                           k: int = 2) -> list[list[tuple[str | None, float]]]:
        """Variante « échiquier » de `predict_topk` (caractères FEN)."""
        return [
            [(CLASS_TO_FEN.get(lbl), conf) for lbl, conf in entries]
            for entries in self.predict_topk(squares, k=k)
        ]


# ── instance partagée ───────────────────────────────────────────────
_default: PieceClassifier | None = None
_default_lock = threading.Lock()


def get_classifier() -> PieceClassifier:
    """Instance unique — évite de recharger 42 Mo de poids à chaque capture."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = PieceClassifier()
    return _default
