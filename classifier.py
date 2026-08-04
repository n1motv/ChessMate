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

    # ── inférence ───────────────────────────────────────────────────
    @torch.inference_mode()
    def predict(self, images: Iterable[Image.Image]) -> tuple[list[str], list[float]]:
        """
        → (labels, confiances) pour la séquence d'images fournie.
        Les confiances sont des probabilités softmax dans [0, 1].
        """
        images = list(images)
        if not images:
            return [], []

        net = self._ensure_loaded()
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        logits = net(batch.to(self.device))
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

        labels = [self.classes[i] for i in idx.tolist()]
        return labels, [float(c) for c in conf.tolist()]

    def predict_board(self, squares: Iterable[Image.Image]
                      ) -> tuple[list[str | None], list[float]]:
        """
        Variante « échiquier » : renvoie directement les caractères FEN
        (None pour une case vide) au lieu des noms de classes.
        """
        labels, conf = self.predict(squares)
        return [CLASS_TO_FEN.get(lbl) for lbl in labels], conf


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
