"""
config.py – réglages persistants (config.json à la racine du projet).

Sert notamment à mémoriser la calibration du plateau **par résolution
d'écran**, ce qui supprime le besoin d'éditer `vision.py` à la main (l'ancien
LEFT / TOP / SIZE documenté dans le README).
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any

from paths import CONFIG_PATH

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    # ── capture écran ───────────────────────────────────────────────
    "monitor": 1,
    # rectangles de plateau calibrés, indexés par "LARGEURxHAUTEUR" de l'écran
    # ex. {"1920x1080": {"left": 441, "top": 237, "size": 1088}}
    "boards": {},
    # ── vision ──────────────────────────────────────────────────────
    # en dessous de ce seuil sur une case, la position est jugée incertaine
    "min_square_confidence": 0.80,
    # nombre de cases douteuses tolérées avant d'abandonner la position
    "max_uncertain_squares": 0,
    # sauvegarde les 64 crops dans data/ (débogage uniquement, coûteux)
    "dump_squares": False,
    # ── moteur ──────────────────────────────────────────────────────
    # "bullet" | "blitz" | "rapid" | "analysis"
    "engine_profile": "blitz",
    "engine_threads": None,     # None → os.cpu_count()
    "engine_hash_mb": None,     # None → valeur du profil
    # ── LLM ─────────────────────────────────────────────────────────
    "llm_enabled": True,
    "llm_timeout": 12.0,
    "llm_retries": 1,
    # ── automatisation ──────────────────────────────────────────────
    "autoplay_style": "click",   # "click" | "drag"
    "autoplay_verify": True,     # recapture l'écran pour confirmer le coup
    # ordre des pièces dans le sélecteur de promotion, du haut vers le bas,
    # tel qu'affiché par chess.com et lichess sur la colonne de promotion
    "promotion_order": ["q", "n", "r", "b"],
    # ── interface ───────────────────────────────────────────────────
    "theme_dark": True,
    "language": None,           # None → détection système
}

_cache: dict[str, Any] | None = None


def load(force: bool = False) -> dict[str, Any]:
    """Charge config.json en le fusionnant avec les valeurs par défaut."""
    global _cache
    if _cache is not None and not force:
        return _cache

    data = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as fh:
                user = json.load(fh)
            if isinstance(user, dict):
                data.update(user)
            else:
                log.warning("config.json ignoré : le contenu n'est pas un objet JSON")
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("config.json illisible (%s) — valeurs par défaut utilisées", exc)

    _cache = data
    return _cache


def save() -> None:
    """Écrit la configuration courante sur disque (best effort)."""
    if _cache is None:
        return
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        log.warning("Impossible d'écrire config.json : %s", exc)


def get(key: str, default: Any = None) -> Any:
    return load().get(key, DEFAULTS.get(key, default))


def set(key: str, value: Any, *, persist: bool = True) -> None:  # noqa: A001
    load()[key] = value
    if persist:
        save()


# ── calibration du plateau, indexée par résolution ──────────────────
def board_key(screen_w: int, screen_h: int) -> str:
    return f"{screen_w}x{screen_h}"


def get_board_rect(screen_w: int, screen_h: int) -> tuple[int, int, int] | None:
    """→ (left, top, size) mémorisé pour cette résolution, ou None."""
    entry = load()["boards"].get(board_key(screen_w, screen_h))
    if not entry:
        return None
    try:
        return int(entry["left"]), int(entry["top"]), int(entry["size"])
    except (KeyError, TypeError, ValueError):
        return None


def set_board_rect(screen_w: int, screen_h: int,
                   left: int, top: int, size: int) -> None:
    load()["boards"][board_key(screen_w, screen_h)] = {
        "left": int(left), "top": int(top), "size": int(size),
    }
    save()


def clear_board_rect(screen_w: int, screen_h: int) -> None:
    load()["boards"].pop(board_key(screen_w, screen_h), None)
    save()
