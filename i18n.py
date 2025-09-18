import json, locale, pathlib

_LANG_DIR = pathlib.Path(__file__).with_suffix('').parent / "lang"
_CACHE    = {}

def available_codes():
    return [p.stem for p in _LANG_DIR.glob("*.json")]

def detect_system_lang(default="en"):
    sys = locale.getdefaultlocale()[0] or ""
    sys = sys.split('_')[0].lower()
    return sys if ( _LANG_DIR / f"{sys}.json").exists() else default

def tr(code:str, key:str) -> str:
    if code not in _CACHE:
        with open(_LANG_DIR / f"{code}.json", encoding="utf8") as f:
            _CACHE[code] = json.load(f)
    return _CACHE[code].get(key, key)
