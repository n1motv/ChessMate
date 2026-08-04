"""
paths.py – tous les chemins du projet, ancrés sur l'emplacement du code.

Corrige le bug « chemins relatifs au CWD » : avant, `pathlib.Path("assets")`
ne fonctionnait que si l'on lançait `python main.py` depuis la racine du
dépôt.  Ici tout est dérivé de `__file__`, donc l'application démarre depuis
n'importe quel répertoire de travail.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

ASSETS      = ROOT / "assets"
FLAGS_DIR   = ASSETS / "flags"
LANG_DIR    = ROOT / "lang"
ENGINES_DIR = ROOT / "engines"
DATA_DIR    = ROOT / "data"          # artefacts de capture (debug)
DATASET_DIR = ROOT / "dataset"       # images d'entraînement
SYZYGY_DIR  = ROOT / "syzygy"

MODEL_PATH   = ROOT / "resnet18_chess.pt"
CLASSES_PATH = ROOT / "classes.json"  # écrit par train_resnet.py (optionnel)
CONFIG_PATH  = ROOT / "config.json"

__all__ = [
    "ROOT", "ASSETS", "FLAGS_DIR", "LANG_DIR", "ENGINES_DIR", "DATA_DIR",
    "DATASET_DIR", "SYZYGY_DIR", "MODEL_PATH", "CLASSES_PATH", "CONFIG_PATH",
]
