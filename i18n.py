"""
i18n.py – traductions, désormais **réellement utilisée**.

Avant, ce module n'était importé nulle part : `main.py` réimplémentait son
propre `load_translations()` / `tr()` et `llm.py` avait un dictionnaire de
libellés codé en dur ne couvrant que le français et l'anglais (d'où un
`KeyError` pour es / ru / zh / ar).  Tout passe maintenant par les fichiers
`lang/*.json`, avec repli sur l'anglais puis sur la clé elle-même.
"""
from __future__ import annotations

import json
import locale
import logging
import threading

from paths import LANG_DIR

log = logging.getLogger(__name__)

FALLBACK = "en"

#: langues dont l'écriture va de droite à gauche
RTL_CODES = frozenset({"ar", "he", "fa", "ur"})

#: noms affichés dans le sélecteur
DISPLAY_NAMES = {
    "fr": "Français", "en": "English", "es": "Español",
    "ru": "Русский", "zh": "中文", "ar": "العربية",
}

#: drapeau associé (fichier de assets/flags/)
FLAGS = {
    "fr": "fr.png", "en": "us.png", "es": "es.png",
    "ru": "ru.png", "zh": "cn.png", "ar": "ma.png",
}

_cache: dict[str, dict[str, str]] = {}
_lock = threading.Lock()


def available_codes() -> list[str]:
    """Codes langue disponibles, dans un ordre stable et prévisible."""
    if not LANG_DIR.is_dir():
        return [FALLBACK]
    found = {p.stem for p in LANG_DIR.glob("*.json")}
    preferred = [c for c in ("fr", "en", "es", "ru", "zh", "ar") if c in found]
    return preferred + sorted(found - set(preferred)) or [FALLBACK]


def load(code: str) -> dict[str, str]:
    """Charge (et met en cache) le dictionnaire d'une langue."""
    with _lock:
        if code in _cache:
            return _cache[code]
        path = LANG_DIR / f"{code}.json"
        data: dict[str, str] = {}
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = {k: str(v) for k, v in loaded.items()}
            else:
                log.warning("lang/%s.json : objet JSON attendu", code)
        except FileNotFoundError:
            log.warning("Traduction absente : lang/%s.json", code)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("lang/%s.json illisible : %s", code, exc)
        _cache[code] = data
        return data


def detect_system_lang(default: str = FALLBACK) -> str:
    """
    Langue du système.  Utilise `locale.getlocale()` : `getdefaultlocale()`
    est déprécié depuis Python 3.11 et émet un DeprecationWarning.
    """
    raw = ""
    try:
        raw = (locale.getlocale()[0] or "")
    except (ValueError, TypeError):
        raw = ""
    if not raw:
        try:
            raw = locale.setlocale(locale.LC_CTYPE, "") or ""
        except locale.Error:
            raw = ""

    code = raw.replace("-", "_").split("_")[0].strip().lower()
    # Windows renvoie parfois le nom complet ("French_France")
    _NAMED = {"french": "fr", "english": "en", "spanish": "es",
              "russian": "ru", "chinese": "zh", "arabic": "ar"}
    code = _NAMED.get(code, code)
    return code if code in available_codes() else default


def tr(code: str, key: str, **fmt) -> str:
    """
    Traduit `key`.  Repli : langue demandée → anglais → la clé brute.
    Les `**fmt` sont appliqués via `str.format` si la chaîne les utilise.
    """
    text = load(code).get(key)
    if text is None and code != FALLBACK:
        text = load(FALLBACK).get(key)
    if text is None:
        text = key
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def is_rtl(code: str) -> bool:
    return code in RTL_CODES


def display_name(code: str) -> str:
    return DISPLAY_NAMES.get(code, code.upper())


def flag_file(code: str) -> str | None:
    return FLAGS.get(code)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
