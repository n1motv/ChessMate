"""
train_resnet.py – entraînement du classifieur de cases.

L'ancienne version tenait en 29 lignes et cumulait les problèmes :

* aucun découpage train / validation → impossible de savoir si le modèle
  apprenait ou récitait ;
* aucune métrique, aucun early stopping, aucune graine aléatoire ;
* pas de `.to(device)` : entraînement sur CPU même avec un GPU disponible ;
* `RandomHorizontalFlip` appliqué à des glyphes de pièces — un miroir change
  l'apparence du cavalier et de plusieurs thèmes, et n'apporte rien ici
  puisque les pièces sont toujours vues de face ;
* l'ordre des classes n'était sauvegardé nulle part, alors que l'inférence en
  dépend entièrement.

⚠️  Le dataset livré ne contient que ~3 images par classe.  C'est très
insuffisant : le modèle appris dessus ne généralise qu'au thème exact utilisé
pour la capture.  Utilisez `python vision.py --dump` sur vos propres parties
pour alimenter `dataset/` — visez au moins 50 images par classe.

    python train_resnet.py --epochs 20 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from classifier import (
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, build_model, pick_device,
)
from paths import CLASSES_PATH, DATASET_DIR, MODEL_PATH


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomSquareHighlight:
    """
    Simule le surlignage coloré qu'appliquent la plupart des sites (dernier
    coup joué, case sélectionnée...) à certaines cases : jaune sur lichess/
    chess.com, parfois vert ou bleu selon le thème.  Le dataset ne contenait
    jusqu'ici aucune case teintée, ce qui faisait déraper la reconnaissance
    dessus (ex. une case vide surlignée en jaune non reconnue comme vide).
    """
    _COLORS = (
        (205, 210, 106),   # jaune clair (lichess, case claire)
        (170, 162, 58),    # jaune foncé (lichess, case sombre)
        (247, 247, 105),   # jaune vif (chess.com)
        (130, 151, 105),   # vert (case sélectionnée)
        (148, 179, 212),   # bleu (case sélectionnée, certains thèmes)
    )

    def __init__(self, p: float = 0.3, alpha_range: tuple[float, float] = (0.15, 0.4)):
        self.p = p
        self.alpha_range = alpha_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        overlay = Image.new("RGB", img.size, random.choice(self._COLORS))
        return Image.blend(img, overlay, random.uniform(*self.alpha_range))


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Augmentations *plausibles* pour des captures d'écran : léger recadrage,
    variations de luminosité/contraste (thèmes clairs et sombres), rotation
    d'un degré ou deux (imprécision de calibration), surlignage occasionnel
    d'une case. Pas de miroir.
    """
    train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        RandomSquareHighlight(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    val = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


def stratified_split(targets: list[int], val_ratio: float, seed: int
                     ) -> tuple[list[int], list[int]]:
    """
    Découpage stratifié : chaque classe est représentée des deux côtés.
    Avec très peu d'images par classe, on garde au moins une image en
    validation dès que la classe en compte au moins deux.
    """
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _label, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_ratio)) if len(indices) > 1 else 0
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


@torch.inference_mode()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device,
             n_classes: int) -> tuple[float, float, np.ndarray]:
    """→ (perte moyenne, exactitude, matrice de confusion)."""
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        total_loss += float(criterion(logits, y)) * y.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        total += y.size(0)
        for t, p in zip(y.tolist(), pred.tolist()):
            confusion[t, p] += 1

    if total == 0:
        return 0.0, 0.0, confusion
    return total_loss / total, correct / total, confusion


def print_confusion(confusion: np.ndarray, classes: list[str]) -> None:
    errors = [(classes[t], classes[p], int(confusion[t, p]))
              for t in range(len(classes))
              for p in range(len(classes))
              if t != p and confusion[t, p]]
    if not errors:
        print("Matrice de confusion : aucune erreur sur la validation ✅")
        return
    print("Confusions (vraie → prédite) :")
    for true, pred, count in sorted(errors, key=lambda e: -e[2]):
        print(f"  {true:<14} → {pred:<14} ×{count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATASET_DIR))
    parser.add_argument("--out", default=str(MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6,
                        help="arrêt anticipé après N époques sans progrès")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    train_tfm, val_tfm = build_transforms()
    base = datasets.ImageFolder(args.data)
    classes = base.classes
    targets = [label for _, label in base.samples]

    counts = Counter(targets)
    print(f"Dataset      : {args.data}")
    print(f"Classes ({len(classes)}) : {classes}")
    print(f"Images       : {len(base)}")
    for i, name in enumerate(classes):
        print(f"  {name:<14} {counts.get(i, 0):>4}")

    if min(counts.values(), default=0) < 10:
        print("\n⚠️  Moins de 10 images pour au moins une classe : le modèle "
              "obtenu sur-apprendra et ne fonctionnera que sur le thème "
              "exact utilisé pour la capture. Enrichissez dataset/ "
              "(python vision.py --dump).\n")

    train_idx, val_idx = stratified_split(targets, args.val_ratio, args.seed)
    train_ds = Subset(datasets.ImageFolder(args.data, train_tfm), train_idx)
    val_ds = Subset(datasets.ImageFolder(args.data, val_tfm), val_idx)
    print(f"Découpage    : {len(train_ds)} entraînement / {len(val_ds)} validation")
    print(f"Appareil     : {device}\n")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers) if len(val_ds) else None

    net = build_model(len(classes), pretrained=True).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    # lissage de labels : utile quand certaines classes se ressemblent beaucoup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state, bad_epochs = -1.0, None, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        running, seen, correct = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()

            running += float(loss) * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            seen += y.size(0)
        scheduler.step()

        train_loss = running / max(1, seen)
        train_acc = correct / max(1, seen)

        if val_dl is not None:
            val_loss, val_acc, _ = evaluate(net, val_dl, device, len(classes))
            print(f"époque {epoch:>3}/{args.epochs}  "
                  f"train {train_loss:.4f} / {train_acc:.1%}   "
                  f"val {val_loss:.4f} / {val_acc:.1%}")
        else:
            val_acc = train_acc
            print(f"époque {epoch:>3}/{args.epochs}  "
                  f"train {train_loss:.4f} / {train_acc:.1%}   (pas de validation)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"\nArrêt anticipé : {args.patience} époques sans progrès.")
                break

    if best_state is None:
        print("❌ Aucun modèle entraîné.")
        return 1

    torch.save(best_state, args.out)
    with open(CLASSES_PATH, "w", encoding="utf-8") as fh:
        json.dump(classes, fh, ensure_ascii=False, indent=2)

    print(f"\nMeilleure exactitude de validation : {best_acc:.1%}")
    print(f"Modèle   → {args.out}")
    print(f"Classes  → {CLASSES_PATH}")
    print(f"Durée    : {time.time() - started:.0f} s")

    if val_dl is not None:
        net.load_state_dict(best_state)
        net.to(device)
        _, _, confusion = evaluate(net, val_dl, device, len(classes))
        print()
        print_confusion(confusion, classes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
