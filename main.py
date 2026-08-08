"""
main.py – interface Qt et orchestration.

Corrections apportées :

* **QThread + signaux** au lieu d'un `QTimer(interval=10)` interrogeant un
  `Future` 100 fois par seconde.  Le worker est réellement asynchrone,
  annulable, et remonte ses erreurs (`failed`) au lieu de les faire
  disparaître.
* **chemins ancrés sur `__file__`** (via `paths.py`) : l'application démarre
  depuis n'importe quel répertoire de travail.
* **i18n centralisée** dans `i18n.py`, qui était jusqu'ici importée nulle
  part pendant que ce fichier réimplémentait sa propre version.
* **arabe en RTL** (`setLayoutDirection`), ce qui n'était pas fait alors que
  la langue était proposée.
* **fondu réellement visible** : `QGraphicsOpacityEffect` sur un conteneur
  interne, l'ombre portée restant sur la carte (un widget Qt ne peut porter
  qu'un seul `QGraphicsEffect`, et animer `windowOpacity` sur un widget
  enfant ne produisait strictement aucun effet).
* **arrêt propre** : plus de `os._exit()`, qui masquait un moteur non fermé.
* **bouton « Calibrer »** : le rectangle du plateau n'est plus figé au
  démarrage du programme.
* `locale.getdefaultlocale()` (déprécié) remplacé par `i18n.detect_system_lang()`.
"""
from __future__ import annotations

import logging
import random
import sys
import threading
import time

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QSize, Qt, QThread, QTimer, Signal,
)
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

import config
import i18n
from autoplay import move_cursor_to, play_move
from calibration import ManualCalibrationOverlay, rect_to_geometry
from overlay import BoardOverlay
from paths import ASSETS, FLAGS_DIR, LANG_DIR
from utils import format_score
from vision import BoardLocator
from worker import Analysis, AnalysisError, Analyzer

log = logging.getLogger(__name__)

# ── palette ─────────────────────────────────────────────────────────
ACCENT = "#00e8ff"
BG_DARK = "#08121d"
BG_LIGHT = "#dce3eb"
TEXT_DARK = "#1e2a38"
CTRL_BG = "#c9d0db"
CTRL_BORDER = "#8d99a6"
CTRL_DISABLED = "#d4d8df"

ICON_SZ = 136


def icon_for(piece_key: str) -> QPixmap:
    """Icône de la pièce ; `piece_key` est déjà canonique (« knight », …)."""
    path = ASSETS / f"{piece_key}.png"
    if not path.exists():
        path = ASSETS / "pawn.png"
    return QPixmap(str(path))


# ───────────────────────────────────────────────────────────────────
#  Thread d'analyse
# ───────────────────────────────────────────────────────────────────
class AnalysisThread(QThread):
    """
    Boucle d'analyse dans un thread dédié.

    Après chaque résultat, le thread **attend** que l'interface ait joué ou
    surligné le coup (`acknowledge()`) avant de repartir : impossible d'avoir
    deux analyses concurrentes ou un état de suivi désynchronisé.
    """

    resultReady = Signal(object)          # Analysis
    failed = Signal(str, str)             # (détail, clé i18n)
    statusChanged = Signal(str)           # clé i18n

    IDLE_POLL = 0.20                      # s entre deux captures si rien ne bouge
    ERROR_PAUSE = 1.50                    # s après une erreur, pour ne pas spammer

    def __init__(self, analyzer: Analyzer, parent=None) -> None:
        super().__init__(parent)
        self._analyzer = analyzer
        self._stop = threading.Event()
        self._ack = threading.Event()
        self._side = "white"
        self._lang = "en"
        self._flipped: bool | None = None
        self._verify_expected: str | None = None
        self._verify_side = "w"

    # ── pilotage depuis l'interface ─────────────────────────────────
    def configure(self, side: str, lang: str, flipped: bool | None = None) -> None:
        """
        `flipped` : orientation réelle du plateau à l'écran, fournie
        explicitement par l'utilisateur (case à cocher) plutôt que déduite
        de la couleur jouée — certains sites n'inversent pas l'affichage
        même quand on joue les Noirs, et deviner mène à lire (et jouer !)
        un plateau en miroir.
        """
        self._side, self._lang, self._flipped = side, lang, flipped

    def acknowledge(self, *, expected_placement: str | None = None,
                    side_after: str = "w") -> None:
        """Débloque le thread une fois le coup joué ou surligné."""
        self._verify_expected = expected_placement
        self._verify_side = side_after
        self._ack.set()

    def stop(self) -> None:
        self._stop.set()
        self._ack.set()

    # ── boucle ──────────────────────────────────────────────────────
    def run(self) -> None:                                     # noqa: C901
        while not self._stop.is_set():
            try:
                analysis = self._analyzer.analyse(
                    self._side, self._lang, flipped=self._flipped)
            except AnalysisError as exc:
                self.failed.emit(str(exc), exc.key)
                self._stop.wait(self.ERROR_PAUSE)
                continue
            except Exception as exc:                            # noqa: BLE001
                log.exception("Erreur inattendue dans le thread d'analyse")
                self.failed.emit(str(exc), "err_generic")
                self._stop.wait(self.ERROR_PAUSE)
                continue

            if analysis is None:                # rien n'a changé à l'écran
                self.statusChanged.emit("status_watching")
                self._stop.wait(self.IDLE_POLL)
                continue

            self._ack.clear()
            self.resultReady.emit(analysis)

            # on attend que l'interface ait fini d'agir
            while not self._ack.wait(0.05):
                if self._stop.is_set():
                    return
            if self._stop.is_set():
                return

            self._verify()

    def _verify(self) -> None:
        """
        Vérifie que le coup a bien été pris en compte par le site.
        C'est ce contrôle qui manquait : l'ancien code écrivait l'état interne
        en supposant que les deux clics avaient réussi.
        """
        expected, self._verify_expected = self._verify_expected, None
        if not expected or not config.get("autoplay_verify", True):
            return

        time.sleep(0.35)                       # laisser l'animation se terminer
        try:
            reading = self._analyzer.read(self._verify_side)
        except AnalysisError as exc:
            log.debug("Vérification impossible : %s", exc)
            return

        if reading.placement != expected:
            log.warning("Coup non confirmé — attendu %s, lu %s",
                        expected, reading.placement)
            self.failed.emit("", "err_autoplay")
            # `resync`, pas `reset` : on garde la certitude que ce n'est
            # plus notre tour. Un reset() traitait la position actuelle
            # (juste après NOTRE coup) comme un premier tour et retentait
            # aussitôt de jouer, avant même que l'adversaire ait répondu —
            # exactement le symptôme "elle essaie de jouer alors que ce
            # n'est pas son tour" remonté par l'utilisateur.
            self._analyzer.resync(reading.placement)


# ───────────────────────────────────────────────────────────────────
#  Fenêtre principale
# ───────────────────────────────────────────────────────────────────
class ChessHelper(QWidget):

    def __init__(self) -> None:
        super().__init__()

        if not ASSETS.is_dir() or not LANG_DIR.is_dir():
            raise FileNotFoundError(
                f"Dossiers requis introuvables : {ASSETS} et {LANG_DIR}"
            )

        self.is_dark = bool(config.get("theme_dark", True))
        self.lang = config.get("language") or i18n.detect_system_lang()
        # pas de détection automatique par contours dans l'interface : c'est
        # toujours l'utilisateur qui entoure l'échiquier à la souris (et donc
        # qui indique du même coup sur quel écran il se trouve).
        self.analyzer = Analyzer(BoardLocator(auto_detect=False))
        self.thread: AnalysisThread | None = None
        self.overlay = BoardOverlay()
        self._manual_overlay: ManualCalibrationOverlay | None = None

        self.sun_icon = self._icon("sun.png")
        self.moon_icon = self._icon("moon.png")

        self._build_ui()
        self._connect()
        self.apply_language()
        QTimer.singleShot(0, lambda: self.apply_theme(self.is_dark))

    # ── construction ────────────────────────────────────────────────
    @staticmethod
    def _icon(name: str) -> QIcon | None:
        path = ASSETS / name
        return QIcon(str(path)) if path.exists() else None

    def _build_ui(self) -> None:
        self.setWindowTitle(self.T("app_title"))
        self.setMinimumWidth(720)
        self._refresh_window_icon()
        self._base_style()

        # ── colonne des raccourcis ─────────────────────────────────
        self.sidebar = QFrame(objectName="sidebar")
        self.sidebar.setFixedWidth(190)
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(12, 12, 12, 12)
        sb.setSpacing(6)
        self.sc_title = QLabel()
        self.sc_title.setFont(QFont("Inter", 12, QFont.Bold))
        sb.addWidget(self.sc_title)
        self.sc_labels: dict[str, QLabel] = {}
        for key, combo in (("sc_startstop", "Space"), ("sc_white", "W"),
                           ("sc_black", "B"), ("sc_mode", "M"),
                           ("sc_theme", "C"), ("sc_calibrate", "R")):
            lbl = QLabel()
            lbl.setWordWrap(True)
            lbl.setFont(QFont("Inter", 9))
            lbl.setProperty("combo", combo)
            sb.addWidget(lbl)
            self.sc_labels[key] = lbl
        sb.addStretch()

        # ── contrôles ───────────────────────────────────────────────
        self.white_rb, self.black_rb = QRadioButton(), QRadioButton()
        self.white_rb.setChecked(True)

        self.mode_box = QComboBox()
        self.mode_box.addItem("", "auto")
        self.mode_box.addItem("", "manual")
        self.mode_box.setFixedWidth(180)

        self.llm_chk = QCheckBox()
        self.llm_chk.setChecked(bool(config.get("llm_enabled", True)))

        # Orientation réelle du plateau à l'écran : par défaut on la déduit
        # de la couleur jouée (convention la plus courante), mais certains
        # sites n'inversent pas l'affichage même en jouant les Noirs — dans
        # ce cas, décocher évite de lire (et jouer !) un plateau en miroir.
        self.flip_chk = QCheckBox()
        self.flip_chk.setChecked(False)
        self._flip_auto = True          # tant que l'utilisateur n'a pas décidé lui-même

        self.lang_box = QComboBox()
        codes = i18n.available_codes()
        for code in codes:
            flag = i18n.flag_file(code)
            path = FLAGS_DIR / flag if flag else None
            icon = QIcon(str(path)) if path and path.exists() else QIcon()
            self.lang_box.addItem(icon, i18n.display_name(code), userData=code)
        if self.lang not in codes:
            self.lang = codes[0]
        self.lang_box.setCurrentIndex(codes.index(self.lang))

        self.theme_btn = QPushButton()
        self.theme_btn.setFixedSize(32, 32)
        self._refresh_theme_icon()

        self.calib_btn = QPushButton()
        self.start_btn, self.stop_btn = QPushButton(), QPushButton()
        self.stop_btn.setEnabled(False)

        # ── bandeau d'attente ───────────────────────────────────────
        self.progress = QProgressBar(alignment=Qt.AlignCenter)
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(320)
        self.progress.setTextVisible(False)
        self.wait_lbl = QLabel(alignment=Qt.AlignCenter)

        wait_layout = QVBoxLayout()
        wait_layout.addWidget(self.progress, alignment=Qt.AlignHCenter)
        wait_layout.addWidget(self.wait_lbl)
        self.load_frame = QFrame()
        self.load_frame.setLayout(wait_layout)
        self.load_frame.hide()

        # ── carte de résultat ───────────────────────────────────────
        self.icon_lbl = QLabel(alignment=Qt.AlignCenter)
        self.icon_lbl.setFixedSize(ICON_SZ, ICON_SZ)
        self.move_lbl = QLabel(alignment=Qt.AlignCenter)
        self.path_lbl = QLabel(alignment=Qt.AlignCenter)
        self.eval_lbl = QLabel(alignment=Qt.AlignCenter)
        self.pv_lbl = QLabel(alignment=Qt.AlignCenter)
        self.pv_lbl.setWordWrap(True)
        self.expl_txt = QTextEdit(readOnly=True)
        self.expl_txt.setFixedHeight(96)
        self.expl_txt.setStyleSheet("background:transparent;")

        inner = QVBoxLayout()
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(6)
        for widget in (self.icon_lbl, self.move_lbl, self.path_lbl,
                       self.eval_lbl, self.pv_lbl, self.expl_txt):
            inner.addWidget(widget, alignment=Qt.AlignHCenter)

        # L'effet d'opacité vit sur ce conteneur ; l'ombre portée reste sur la
        # carte. Un widget Qt ne peut porter qu'un seul QGraphicsEffect.
        self.card_content = QWidget()
        self.card_content.setLayout(inner)
        self.card_opacity = QGraphicsOpacityEffect(self.card_content)
        self.card_opacity.setOpacity(1.0)
        self.card_content.setGraphicsEffect(self.card_opacity)

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.addWidget(self.card_content)
        self.card = QFrame(objectName="info_card")
        self.card.setLayout(card_layout)
        # L'effet est parenté à la carte : sans référence côté Python, il est
        # détruit dès la fin de l'expression et l'ombre disparaît.
        self.card_shadow = QGraphicsDropShadowEffect(self.card)
        self.card_shadow.setBlurRadius(25)
        self.card_shadow.setOffset(0, 3)
        self.card_shadow.setColor(QColor(0, 0, 0, 180))
        self.card.setGraphicsEffect(self.card_shadow)
        self.card.hide()

        # ── barre d'état ────────────────────────────────────────────
        self.status_lbl = QLabel(objectName="status_bar", alignment=Qt.AlignCenter)
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setMinimumHeight(24)

        # ── crédits ─────────────────────────────────────────────────
        self.credits_bar = QFrame(objectName="credits_bar")
        self.credits_bar.setFixedHeight(64)
        self.credits_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid = QGridLayout(self.credits_bar)
        grid.setContentsMargins(20, 2, 20, 2)
        grid.setHorizontalSpacing(6)
        txt_lbl = QLabel("créé par n1motv 😎")
        txt_lbl.setFont(QFont("Inter", 18, QFont.DemiBold))
        self.credits_icon = QLabel(objectName="credits_icon")
        self.credits_icon.setFixedSize(60, 60)
        self.credits_icon.setAlignment(Qt.AlignCenter)
        grid.addWidget(txt_lbl, 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(self.credits_icon, 0, 2,
                       alignment=Qt.AlignRight | Qt.AlignVCenter)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)

        # ── assemblage ──────────────────────────────────────────────
        head = QHBoxLayout()
        head.addWidget(self.sidebar)
        head.addWidget(self.white_rb)
        head.addWidget(self.black_rb)
        head.addStretch()
        head.addWidget(self.mode_box)
        head.addWidget(self.flip_chk)
        head.addWidget(self.llm_chk)
        head.addWidget(self.lang_box)
        head.addWidget(self.calib_btn)
        head.addWidget(self.theme_btn)

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)
        root.addLayout(head)
        root.addWidget(self.start_btn)
        root.addWidget(self.stop_btn)
        root.addWidget(self.load_frame, alignment=Qt.AlignHCenter)
        root.addWidget(self.card)
        root.addWidget(self.status_lbl)
        root.addWidget(self.credits_bar, alignment=Qt.AlignBottom)

        self.setFocusPolicy(Qt.StrongFocus)

    def _connect(self) -> None:
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.calib_btn.clicked.connect(self.calibrate)
        self.lang_box.currentIndexChanged.connect(self.on_lang_change)
        self.llm_chk.toggled.connect(self.on_llm_toggle)
        self.flip_chk.toggled.connect(self.on_flip_toggle)
        self.white_rb.toggled.connect(self._sync_flip_default)
        self.black_rb.toggled.connect(self._sync_flip_default)

        # Les raccourcis passent par QShortcut : ils fonctionnent même quand
        # le focus est sur un QComboBox ou le QTextEdit.
        for keys, slot in (
            ("Space", self._toggle_run),
            ("W", lambda: self.white_rb.setChecked(True)),
            ("B", lambda: self.black_rb.setChecked(True)),
            ("M", self._toggle_mode),
            ("C", self.toggle_theme),
            ("R", self.calibrate),
        ):
            QShortcut(QKeySequence(keys), self, activated=slot)

    # ── i18n ────────────────────────────────────────────────────────
    def T(self, key: str, **fmt) -> str:
        return i18n.tr(self.lang, key, **fmt)

    def on_lang_change(self, index: int) -> None:
        self.lang = self.lang_box.itemData(index)
        config.set("language", self.lang)
        if self.thread is not None:
            self.thread.configure(self._side(), self.lang, self.flip_chk.isChecked())
        self.apply_language()

    def on_llm_toggle(self, checked: bool) -> None:
        config.set("llm_enabled", checked)

    def on_flip_toggle(self, _checked: bool) -> None:
        # dès que l'utilisateur touche la case lui-même, on arrête de la
        # réécrire automatiquement quand il change de couleur.
        self._flip_auto = False

    def _sync_flip_default(self) -> None:
        """Suggestion par défaut : Noirs → plateau retourné (convention la
        plus courante). Ne s'applique que tant que l'utilisateur n'a pas
        lui-même corrigé la case, pour les sites qui ne retournent pas
        l'affichage."""
        if self._flip_auto:
            self.flip_chk.setChecked(self.black_rb.isChecked())

    def apply_language(self) -> None:
        self.setLayoutDirection(
            Qt.RightToLeft if i18n.is_rtl(self.lang) else Qt.LeftToRight
        )
        self.setWindowTitle(self.T("app_title"))

        self.white_rb.setText(self.T("white"))
        self.black_rb.setText(self.T("black"))
        for rb in (self.white_rb, self.black_rb):
            rb.setFont(QFont("Inter", 13, QFont.Bold))

        self.start_btn.setText(self.T("start"))
        self.stop_btn.setText(self.T("stop"))
        self.calib_btn.setText(self.T("calibrate"))
        self.wait_lbl.setText(self.T("waiting"))
        self.mode_box.setItemText(0, self.T("mode_auto"))
        self.mode_box.setItemText(1, self.T("mode_manual"))
        self.llm_chk.setText(self.T("llm_toggle"))
        self.llm_chk.setToolTip(self.T("llm_toggle_hint"))
        self.flip_chk.setText(self.T("flip_board"))
        self.flip_chk.setToolTip(self.T("flip_board_hint"))

        self.sc_title.setText(self.T("shortcuts"))
        for key, lbl in self.sc_labels.items():
            lbl.setText(f"<b>{lbl.property('combo')} :</b> {self.T(key)}")

        self.start_btn.setToolTip(f"{self.T('sc_startstop')} (Space)")
        self.stop_btn.setToolTip(f"{self.T('sc_startstop')} (Space)")
        self.white_rb.setToolTip(f"{self.T('sc_white')} (W)")
        self.black_rb.setToolTip(f"{self.T('sc_black')} (B)")
        self.mode_box.setToolTip(f"{self.T('sc_mode')} (M)")
        self.theme_btn.setToolTip(f"{self.T('sc_theme')} (C)")
        self.calib_btn.setToolTip(f"{self.T('sc_calibrate')} (R)")

        self.move_lbl.setFont(QFont("Inter", 30, QFont.Black))
        self.path_lbl.setFont(QFont("Inter", 20, QFont.DemiBold))
        self.eval_lbl.setFont(QFont("Inter", 16))
        self.pv_lbl.setFont(QFont("Inter", 11))

    # ── thème ───────────────────────────────────────────────────────
    def _base_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{BG_DARK}; color:#e0e0e0; }}
            QPushButton {{ background:{ACCENT}; color:{BG_DARK};
                           border:none; border-radius:6px; padding:4px 10px; }}
            QProgressBar {{ border:1px solid {ACCENT}; border-radius:6px; }}
        """)

    def _refresh_window_icon(self) -> None:
        logo = "chessmate.png" if self.is_dark else "chessmate_light.png"
        path = ASSETS / logo
        if path.exists():
            self.setWindowIcon(QIcon(str(path)))

    def _refresh_theme_icon(self) -> None:
        icon = self.moon_icon if self.is_dark else self.sun_icon
        if icon and not icon.isNull():
            self.theme_btn.setIcon(icon)
            self.theme_btn.setIconSize(QSize(20, 20))
            self.theme_btn.setText("")
        else:
            self.theme_btn.setIcon(QIcon())
            self.theme_btn.setText("☾" if self.is_dark else "☀")

    def toggle_theme(self) -> None:
        self.apply_theme(not self.is_dark)

    def apply_theme(self, dark: bool) -> None:
        self.is_dark = dark
        config.set("theme_dark", dark)
        self._refresh_window_icon()
        self._refresh_theme_icon()

        logo = "chessmate.png" if dark else "chessmate_light.png"
        path = ASSETS / logo
        if path.exists():
            self.credits_icon.setPixmap(
                QPixmap(str(path)).scaled(60, 60, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            )

        try:
            import qt_material
            qt_material.apply_stylesheet(
                self,
                theme="dark_teal.xml" if dark else "light_blue.xml",
                extra={
                    "primaryColor": ACCENT,
                    "accentLightColor": ACCENT,
                    "secondaryTextColor": "#cfd8dc" if dark else "#5f6f8c",
                    "background": BG_DARK if dark else BG_LIGHT,
                },
            )
        except Exception as exc:                                # noqa: BLE001
            log.warning("qt-material indisponible (%s) — thème simplifié", exc)

        text_color = "#e0e0e0" if dark else TEXT_DARK
        bg = BG_DARK if dark else BG_LIGHT
        panel_bg = "rgba(255,255,255,0.03)" if dark else CTRL_BG
        panel_border = ("1px solid rgba(255,255,255,0.08)" if dark
                        else f"1px solid {CTRL_BORDER}")
        button_border = "none" if dark else f"1px solid {CTRL_BORDER}"

        self.setStyleSheet(self.styleSheet() + f"""
            QWidget {{ background:{bg}; color:{text_color}; }}
            QPushButton {{
                background:{ACCENT}; color:{BG_DARK}; font-weight:bold;
                border:{button_border}; border-radius:8px; padding:6px 14px;
            }}
            QPushButton:disabled {{
                background:{CTRL_DISABLED}; color:#8a8a8a; border:{button_border};
            }}
            QFrame#info_card, QFrame#credits_bar, QFrame#sidebar {{
                background:{panel_bg}; border:{panel_border}; border-radius:8px;
            }}
            QFrame#credits_bar QLabel {{
                background:transparent; font-size:18px; font-weight:600;
            }}
            QLabel#status_bar {{ background:transparent; font-size:12px; }}
        """)

    # ── démarrage / arrêt ───────────────────────────────────────────
    def _side(self) -> str:
        return "white" if self.white_rb.isChecked() else "black"

    def _mode(self) -> str:
        return self.mode_box.currentData()

    def _toggle_run(self) -> None:
        self.stop() if self._running() else self.start()

    def _running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def _toggle_mode(self) -> None:
        self.mode_box.setCurrentIndex(1 - self.mode_box.currentIndex())

    def start(self) -> None:
        if self._running():
            return
        self.analyzer.reset()
        self.set_status("")
        self.card.hide()
        self.load_frame.show()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = AnalysisThread(self.analyzer, self)
        self.thread.configure(self._side(), self.lang, self.flip_chk.isChecked())
        self.thread.resultReady.connect(self.on_result)
        self.thread.failed.connect(self.on_failed)
        self.thread.statusChanged.connect(self.on_status_key)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()

    def stop(self) -> None:
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.stop()
            if not thread.wait(3000):
                log.warning("Le thread d'analyse n'a pas répondu — terminaison")
                thread.terminate()
                thread.wait(500)
        self.overlay.clear()
        self.load_frame.hide()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_thread_finished(self) -> None:
        self.load_frame.hide()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ── calibration ─────────────────────────────────────────────────
    def calibrate(self) -> None:
        """
        Aucune détection automatique par contours dans l'interface : trop
        peu fiable (thèmes à faible contraste, autres damiers à l'écran,
        fenêtre ChessMate qui recouvre le plateau...). C'est toujours
        l'utilisateur qui entoure l'échiquier à la souris — ce qui indique
        du même coup sur quel écran il se trouve.
        """
        was_running = self._running()
        self.stop()
        self.set_status(self.T("calibrate_manual_hint"))
        overlay = ManualCalibrationOverlay(self.T("calibrate_manual_hint"))
        self._manual_overlay = overlay          # référence forte : sinon Qt la détruit

        overlay.selected.connect(
            lambda rect: self._on_manual_calibrated(rect, was_running))
        overlay.cancelled.connect(self._on_manual_calibration_cancelled)
        overlay.show()

    def _on_manual_calibrated(self, rect, was_running: bool) -> None:
        self._manual_overlay = None
        try:
            monitor, left, top, size = rect_to_geometry(rect)
            geom = self.analyzer.locator.set_manual(left, top, size, monitor=monitor)
        except Exception as exc:                                # noqa: BLE001
            self.set_status(self.T("err_vision", detail=str(exc)), error=True)
            return
        self.analyzer.reset()
        self.set_status(self.T("calibrated", size=geom.size,
                               left=geom.left, top=geom.top))
        if was_running:
            self.start()

    def _on_manual_calibration_cancelled(self) -> None:
        self._manual_overlay = None
        self.set_status(self.T("calibrate_manual_cancelled"), error=True)

    # ── réception des résultats ─────────────────────────────────────
    def on_status_key(self, key: str) -> None:
        if not self.status_lbl.property("sticky"):
            self.set_status(self.T(key), transient=True)

    def set_status(self, text: str, *, error: bool = False,
                   transient: bool = False) -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setProperty("sticky", error)
        color = "#ff6b6b" if error else ("#8fa3b8" if transient else ACCENT)
        self.status_lbl.setStyleSheet(f"color:{color};background:transparent;")

    def on_failed(self, detail: str, key: str) -> None:
        message = self.T(key, detail=detail) if detail else self.T(key)
        self.set_status(message, error=True)
        log.warning("Analyse : %s", message)

    def on_result(self, analysis: Analysis) -> None:
        """
        Exécuté dans le thread graphique.  Le thread d'analyse est en pause
        jusqu'à `acknowledge()`, donc aucune course n'est possible ici.

        En mode automatique, le coup n'est pas joué tout de suite : un clic
        instantané dès que le moteur a fini de réfléchir n'a rien d'humain
        et se repère facilement. On affiche le coup trouvé immédiatement
        (transparence pour l'utilisateur), mais on ne clique/glisse
        réellement qu'après un délai aléatoire — `QTimer.singleShot` plutôt
        qu'un `time.sleep`, pour ne pas geler l'interface pendant l'attente.
        """
        self._render(analysis)

        if self._mode() == "auto":
            delay = random.uniform(
                float(config.get("autoplay_delay_min", 3.0)),
                float(config.get("autoplay_delay_max", 7.0)),
            )
            self.set_status(self.T("status_delay"), transient=True)
            thread = self.thread
            QTimer.singleShot(int(delay * 1000),
                              lambda: self._execute_result(analysis, thread))
        else:
            self._execute_result(analysis, self.thread)

    def _execute_result(self, analysis: Analysis,
                        thread: AnalysisThread | None) -> None:
        """
        Joue (ou surligne) le coup puis débloque le thread d'analyse.
        Séparé de `on_result` pour pouvoir être différé par le délai
        aléatoire ci-dessus sans bloquer l'interface entre-temps.

        `thread` est celui qui attendait CE résultat précisément : si
        l'utilisateur a arrêté (ou recalibré, ce qui redémarre un nouveau
        thread) pendant l'attente, `thread` n'est plus `self.thread` — le
        coup n'a alors plus lieu d'être et on l'abandonne silencieusement,
        plutôt que de cliquer sur un plateau qu'on ne surveille plus.
        """
        if thread is None or thread is not self.thread or not thread.isRunning():
            return

        played = False
        try:
            if self._mode() == "auto":
                self.set_status(self.T("status_playing"), transient=True)
                QApplication.processEvents()
                play_move(analysis.move, analysis.geometry,
                          flipped=analysis.flipped, board=analysis.board_before)
                played = True
            else:
                self.overlay.show_move(analysis.geometry, analysis.src,
                                       analysis.dst, flipped=analysis.flipped)

            self._return_cursor()
            self.analyzer.commit(analysis, played=played)

        except Exception as exc:                                # noqa: BLE001
            log.exception("Échec de l'exécution du coup")
            self.set_status(self.T("err_generic", detail=str(exc)), error=True)
            self.analyzer.commit_placement(analysis.placement)
        finally:
            board = analysis.board_after if played else analysis.board_before
            thread.acknowledge(
                expected_placement=board.board_fen() if played else None,
                side_after="w" if board.turn else "b",
            )

    def _render(self, a: Analysis) -> None:
        self.load_frame.hide()
        self.icon_lbl.setPixmap(icon_for(a.piece).scaled(
            ICON_SZ, ICON_SZ, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.move_lbl.setText(f"{self.T('bestMove')} : <b>{a.san}</b>")
        self.path_lbl.setText(f"{self.T('bestPath')} : <b>{a.src} → {a.dst}</b>")

        board_after = a.board_after
        if board_after.is_checkmate():
            self.eval_lbl.setText(self.T("checkmate"))
        elif board_after.is_stalemate():
            self.eval_lbl.setText(self.T("stalemate"))
        else:
            self.eval_lbl.setText(
                f"{self.T('eval')} : {format_score(a.score_pawns, a.mate_in)}"
                f"   ·   {self.T('confidence')} : {a.confidence:.0%}"
            )

        self.pv_lbl.setText(a.pv_san)
        self.pv_lbl.setVisible(bool(a.pv_san))

        if a.explanation.strip():
            self.expl_txt.setPlainText(a.explanation)
            self.expl_txt.show()
        else:
            self.expl_txt.hide()

        if a.llm_error:
            self.set_status(self.T("err_llm", detail=a.llm_error), transient=True)
        else:
            # un résultat exploitable veut dire que la lecture a réussi :
            # ça efface toute erreur "sticky" laissée par un échec transitoire
            # précédent (case incertaine, plateau non trouvé un instant...),
            # sans quoi le bandeau rouge restait affiché indéfiniment même
            # une fois l'analyse repartie normalement.
            self.set_status("")

        self._fade_in(self.card)

    def _fade_in(self, widget: QWidget) -> None:
        widget.show()
        anim = QPropertyAnimation(self.card_opacity, b"opacity", self)
        anim.setDuration(320)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    def _return_cursor(self) -> None:
        """Ramène le curseur sur l'interface pour que l'utilisateur reprenne la main."""
        top_left = self.mapToGlobal(self.rect().topLeft())
        move_cursor_to(top_left.x() + self.width() // 2, top_left.y() + 40)

    # ── fermeture ───────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        self.stop()
        self.overlay.close()
        try:
            self.analyzer.close()
        except Exception:                                       # noqa: BLE001
            log.exception("Erreur lors de la fermeture du moteur")
        config.save()
        super().closeEvent(event)


# ───────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("ChessMate")

    window = ChessHelper()
    window.show()
    # Plus d'os._exit() : le moteur est fermé par closeEvent + atexit, donc
    # une sortie normale suffit et les tampons sont bien vidés.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
