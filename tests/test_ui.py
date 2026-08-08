"""
Tests d'interface, exécutés en mode « offscreen » (aucune fenêtre affichée).

    QT_QPA_PLATFORM=offscreen python tests/test_ui.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import i18n  # noqa: E402
from paths import ASSETS, FLAGS_DIR, LANG_DIR  # noqa: E402


def test_all_languages_share_the_same_keys():
    """Le bug historique : llm.py ne connaissait que « fr » et « en »."""
    reference = set(json.loads((LANG_DIR / "en.json").read_text(encoding="utf-8")))
    for code in i18n.available_codes():
        keys = set(json.loads((LANG_DIR / f"{code}.json").read_text(encoding="utf-8")))
        missing = reference - keys
        assert not missing, f"lang/{code}.json : clés manquantes {sorted(missing)}"


def test_llm_prompt_keys_exist_for_every_language():
    """Sélectionner l'espagnol ou l'arabe levait un KeyError dans llm.py."""
    needed = ["llm_system", "llm_rule", "llm_explain_hint",
              "llm_side_white", "llm_side_black"]
    for code in i18n.available_codes():
        for key in needed:
            value = i18n.tr(code, key)
            assert value != key, f"{code}: « {key} » non traduite"


def test_translation_falls_back_to_english():
    assert i18n.tr("fr", "cle_totalement_inexistante") == "cle_totalement_inexistante"
    assert i18n.tr("zz", "start") == i18n.tr("en", "start")


def test_format_placeholders_are_applied():
    text = i18n.tr("fr", "err_generic", detail="boum")
    assert "boum" in text and "{detail}" not in text


def test_rtl_only_for_arabic():
    assert i18n.is_rtl("ar")
    for code in ("fr", "en", "es", "ru", "zh"):
        assert not i18n.is_rtl(code)


def test_detect_system_lang_returns_available_code():
    assert i18n.detect_system_lang() in i18n.available_codes()


def test_assets_referenced_by_the_ui_exist():
    from utils import PIECE_KEYS
    for key in PIECE_KEYS.values():
        assert (ASSETS / f"{key}.png").exists(), f"assets/{key}.png manquant"
    for code in i18n.available_codes():
        flag = i18n.flag_file(code)
        if flag:
            assert (FLAGS_DIR / flag).exists(), f"assets/flags/{flag} manquant"


def test_ui_labels_are_translated_in_every_language():
    """Les intitulés introduits par la refonte doivent exister partout."""
    keys = ["app_tagline", "white_short", "black_short", "sec_color",
            "sec_mode", "sec_options", "sec_board", "idle_title",
            "idle_hint", "status_ready"]
    for code in i18n.available_codes():
        for key in keys:
            assert i18n.tr(code, key) != key, f"{code}: « {key} » non traduite"


def test_main_window_builds_and_switches_language():
    """Instancie réellement la fenêtre : détecte les erreurs de construction."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import main

    app = QApplication.instance() or QApplication([])
    window = main.ChessHelper()
    try:
        assert window.windowTitle()
        assert not window._running()

        # bascule de langue : libellés et sens d'écriture
        for code in ("fr", "en", "ar", "zh"):
            window.lang = code
            window.apply_language()
            # bouton d'action unique : il porte « start » tant que l'analyse
            # n'a pas démarré, « stop » ensuite.
            assert window.run_btn.text() == i18n.tr(code, "start")
            expected = Qt.RightToLeft if i18n.is_rtl(code) else Qt.LeftToRight
            assert window.layoutDirection() == expected, f"RTL incorrect pour {code}"

        # bascule de thème
        window.apply_theme(False)
        assert window.is_dark is False
        window.apply_theme(True)
        assert window.is_dark is True

        # la feuille est régénérée, jamais concaténée : l'ancien
        # `setStyleSheet(self.styleSheet() + …)` la faisait grossir à chaque
        # bascule, en empilant des règles contradictoires.
        size = len(app.styleSheet())
        window.apply_theme(False)
        window.apply_theme(True)
        assert len(app.styleSheet()) == size

        # une seule page de contenu visible à la fois (avant : le bandeau
        # d'attente et la carte pouvaient l'être en même temps)
        assert window.stack.count() == 2
        assert window.stack.currentWidget() is not None

        # l'effet d'opacité existe bien (l'ancien fondu était un no-op)
        assert window.card_content.graphicsEffect() is window.card_opacity
        assert window.card.graphicsEffect() is not None

        # l'overlay est une fenêtre de premier niveau, pas un enfant
        assert window.overlay.parent() is None
        assert window.overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
    finally:
        window.stop()
        window.overlay.close()
        window.close()
        app.processEvents()


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
