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

Refonte de l'interface :

* **une seule source de style** (`theme.py`). `qt-material` posait un thème
  complet par-dessus lequel deux feuilles maison étaient *concaténées*
  (`setStyleSheet(self.styleSheet() + …)`) : chaque bascule de thème empilait
  une règle de plus sans retirer la précédente, d'où des contrôles restés
  aux couleurs de l'ancien thème. La feuille est maintenant régénérée
  entièrement et posée sur le `QApplication`, ce qui couvre aussi les
  fenêtres détachées (liste déroulante, infobulle).
* **une seule zone de contenu** (`QStackedWidget`). Le bandeau d'attente et
  la carte de résultat étaient deux widgets empilés dans le même layout,
  affichés/masqués indépendamment : la fenêtre sautait à chaque analyse et
  les deux pouvaient se retrouver visibles en même temps.
* **un seul bouton d'action** au lieu de « Démarrer » et « Arrêter » côte à
  côte dont l'un était toujours grisé.
* **la barre latérale n'est plus dans la barre d'outils** : elle occupait une
  colonne du `QHBoxLayout` d'en-tête, ce qui écrasait les contrôles voisins.
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
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame,
    QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QGridLayout,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

import config
import i18n
import theme
from autoplay import move_cursor_to, play_move
from calibration import ManualCalibrationOverlay, rect_to_geometry
from overlay import BoardOverlay
from paths import ASSETS, FLAGS_DIR, LANG_DIR
from utils import format_score
from vision import BoardLocator
from worker import Analysis, AnalysisError, Analyzer

log = logging.getLogger(__name__)

ICON_SZ = 76
#: la barre latérale ne se réduit pas librement : les QCheckBox et QPushButton
#: de Qt tronquent leur texte au lieu de l'abréger. Mesuré sur les six
#: langues, le libellé le plus long en réclame 224 ; on garde une marge.
SIDEBAR_W = 240
HINT_W = 290          # largeur du texte d'accueil, en px logiques

#: (touche affichée, clé i18n) — sert au panneau des raccourcis ET aux
#: infobulles, pour qu'un ajout ne puisse plus être oublié d'un côté.
SHORTCUTS = (
    ("Space", "sc_startstop"),
    ("W", "sc_white"),
    ("B", "sc_black"),
    ("M", "sc_mode"),
    ("C", "sc_theme"),
    ("R", "sc_calibrate"),
)


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
        # dernier état de la barre d'état : (texte, ton). Mémorisé pour être
        # redessiné aux bonnes couleurs après un changement de thème.
        self._status: tuple[str, str] = ("", "muted")

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

    def _section_label(self, key: str) -> QLabel:
        """Intitulé de section de la barre latérale (traduit et capitalisé)."""
        lbl = QLabel(objectName="section")
        self._sections[key] = lbl
        return lbl

    @staticmethod
    def _separator() -> QFrame:
        return QFrame(objectName="separator")

    def _build_ui(self) -> None:
        self.setObjectName("root")
        self.setWindowTitle(self.T("app_title"))
        self.setMinimumSize(640, 620)
        self.resize(714, 700)
        self._refresh_window_icon()
        self._sections: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(18, 18, 18, 14)
        body.setSpacing(18)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_content(), 1)
        root.addLayout(body, 1)

        root.addWidget(self._build_statusbar())
        self.setFocusPolicy(Qt.StrongFocus)
        # sans ça, le focus initial tombe sur le sélecteur de langue, qui
        # s'affiche alors entouré d'accent au démarrage.
        self.run_btn.setFocus()

    # ── en-tête ─────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        header = QFrame(objectName="header")
        header.setFixedHeight(62)
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 0, 14, 0)
        row.setSpacing(12)

        self.logo_lbl = QLabel()
        self.logo_lbl.setFixedSize(30, 30)
        self.logo_lbl.setScaledContents(True)

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        self.title_lbl = QLabel("ChessMate", objectName="app_title")
        self.tagline_lbl = QLabel(objectName="app_tagline")
        titles.addWidget(self.title_lbl)
        titles.addWidget(self.tagline_lbl)

        self.lang_box = QComboBox()
        self.lang_box.setFixedWidth(148)
        codes = i18n.available_codes()
        for code in codes:
            flag = i18n.flag_file(code)
            path = FLAGS_DIR / flag if flag else None
            icon = QIcon(str(path)) if path and path.exists() else QIcon()
            self.lang_box.addItem(icon, i18n.display_name(code), userData=code)
        if self.lang not in codes:
            self.lang = codes[0]
        self.lang_box.setCurrentIndex(codes.index(self.lang))

        self.theme_btn = QPushButton(objectName="icon")
        self.theme_btn.setFixedSize(38, 38)
        self._refresh_theme_icon()

        row.addWidget(self.logo_lbl)
        row.addLayout(titles)
        row.addStretch()
        row.addWidget(self.lang_box)
        row.addWidget(self.theme_btn)
        return header

    # ── barre latérale ──────────────────────────────────────────────
    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(SIDEBAR_W)
        col = QVBoxLayout(sidebar)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(10)

        # couleur jouée — contrôle segmenté : deux boutons bascule réunis dans
        # un QButtonGroup exclusif, ce qui garde l'API `isChecked()`/`toggled`
        # des anciens QRadioButton sans en traîner la pastille.
        self.white_rb = QPushButton(objectName="segment", checkable=True)
        self.black_rb = QPushButton(objectName="segment", checkable=True)
        self.side_group = QButtonGroup(self)
        self.side_group.setExclusive(True)
        for rb in (self.white_rb, self.black_rb):
            rb.setCursor(Qt.PointingHandCursor)
            rb.setMinimumHeight(40)
            self.side_group.addButton(rb)
        self.white_rb.setChecked(True)
        segment = QHBoxLayout()
        segment.setSpacing(8)
        segment.addWidget(self.white_rb)
        segment.addWidget(self.black_rb)

        col.addWidget(self._section_label("sec_color"))
        col.addLayout(segment)

        # mode
        self.mode_box = QComboBox()
        self.mode_box.addItem("", "auto")
        self.mode_box.addItem("", "manual")
        col.addSpacing(4)
        col.addWidget(self._section_label("sec_mode"))
        col.addWidget(self.mode_box)

        # options
        self.llm_chk = QCheckBox()
        self.llm_chk.setChecked(bool(config.get("llm_enabled", True)))
        self.flip_chk = QCheckBox()
        self.flip_chk.setChecked(False)
        for chk in (self.llm_chk, self.flip_chk):
            chk.setCursor(Qt.PointingHandCursor)
        # Orientation réelle du plateau à l'écran : par défaut on la déduit
        # de la couleur jouée (convention la plus courante), mais certains
        # sites n'inversent pas l'affichage même en jouant les Noirs — dans
        # ce cas, décocher évite de lire (et jouer !) un plateau en miroir.
        self._flip_auto = True          # tant que l'utilisateur n'a pas décidé

        col.addSpacing(4)
        col.addWidget(self._section_label("sec_options"))
        col.addWidget(self.llm_chk)
        col.addWidget(self.flip_chk)

        # plateau
        self.calib_btn = QPushButton()
        self.calib_btn.setCursor(Qt.PointingHandCursor)
        col.addSpacing(4)
        col.addWidget(self._section_label("sec_board"))
        col.addWidget(self.calib_btn)

        col.addStretch()

        # action principale : un seul bouton qui bascule, au lieu de deux
        # boutons dont l'un était en permanence grisé.
        self.run_btn = QPushButton(objectName="primary")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(48)
        col.addWidget(self.run_btn)

        col.addSpacing(6)
        col.addWidget(self._separator())
        col.addSpacing(2)
        col.addWidget(self._section_label("shortcuts"))
        col.addLayout(self._build_shortcuts())
        return sidebar

    def _build_shortcuts(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)
        self.sc_labels: dict[str, QLabel] = {}
        for row, (combo, key) in enumerate(SHORTCUTS):
            kbd = QLabel(combo, objectName="kbd", alignment=Qt.AlignCenter)
            kbd.setFixedHeight(19)
            desc = QLabel(objectName="shortcut")
            desc.setWordWrap(True)
            grid.addWidget(kbd, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(desc, row, 1)
            self.sc_labels[key] = desc
        grid.setColumnStretch(1, 1)
        return grid

    # ── zone centrale ───────────────────────────────────────────────
    def _build_content(self) -> QStackedWidget:
        """
        Les trois états possibles occupent la **même** case d'un
        `QStackedWidget` : plus de widgets qui se superposent ou de fenêtre
        qui saute de hauteur quand l'analyse démarre.
        """
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_placeholder())   # 0 – au repos
        self.stack.addWidget(self._build_card())          # 1 – résultat
        self.stack.setCurrentIndex(0)
        return self.stack

    def _build_placeholder(self) -> QFrame:
        frame = QFrame(objectName="placeholder")
        col = QVBoxLayout(frame)
        col.setContentsMargins(32, 32, 32, 32)
        col.setSpacing(14)
        col.addStretch()

        # Ni le logo ni les images de pièces ne conviennent ici : ce sont des
        # PNG à fond opaque (imagettes découpées sur un plateau), qui posent
        # un rectangle beige ou gris au milieu du panneau. Un glyphe Unicode
        # se colore avec le thème et n'a pas de fond.
        self.idle_icon = QLabel("♞", objectName="idle_glyph",
                                alignment=Qt.AlignCenter)
        self.idle_icon.setFixedHeight(96)
        self.idle_title = QLabel(objectName="idle_title", alignment=Qt.AlignCenter)
        self.idle_hint = QLabel(objectName="hint", alignment=Qt.AlignCenter)
        self.idle_hint.setWordWrap(True)
        # largeur figée : sans elle, le layout calcule la hauteur du libellé
        # sur une seule ligne alors qu'il en occupe deux, et le texte
        # débordait par-dessus le titre.
        self.idle_hint.setFixedWidth(HINT_W)

        # bandeau d'attente : la barre indéterminée n'apparaît que pendant
        # une analyse, à la place du texte d'accueil.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(240)
        self.progress.setTextVisible(False)
        self.progress.hide()

        col.addWidget(self.idle_icon, alignment=Qt.AlignHCenter)
        col.addWidget(self.idle_title)
        col.addWidget(self.idle_hint, alignment=Qt.AlignHCenter)
        col.addSpacing(6)
        col.addWidget(self.progress, alignment=Qt.AlignHCenter)
        col.addStretch()
        return frame

    def _build_card(self) -> QFrame:
        self.icon_lbl = QLabel(alignment=Qt.AlignCenter)
        self.icon_lbl.setFixedSize(ICON_SZ, ICON_SZ)
        self.move_lbl = QLabel(objectName="move")
        self.path_lbl = QLabel(objectName="path")
        self.eval_chip = QLabel(objectName="chip")
        self.eval_chip.setProperty("tone", "accent")
        self.conf_chip = QLabel(objectName="chip")

        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.setContentsMargins(0, 4, 0, 0)
        chips.addWidget(self.eval_chip)
        chips.addWidget(self.conf_chip)
        chips.addStretch()

        # aucun ressort ici : un espaceur extensible dans une sous-disposition
        # se partage l'espace libre avec celui du bas de la carte, ce qui
        # creusait un large trou entre les puces et le séparateur.
        summary = QVBoxLayout()
        summary.setContentsMargins(0, 0, 0, 0)
        summary.setSpacing(2)
        summary.addWidget(self.move_lbl)
        summary.addWidget(self.path_lbl)
        summary.addLayout(chips)

        head = QHBoxLayout()
        head.setSpacing(20)
        head.addWidget(self.icon_lbl, 0, Qt.AlignTop)
        head.addLayout(summary, 1)

        self.pv_lbl = QLabel(objectName="pv")
        self.pv_lbl.setWordWrap(True)
        self.expl_txt = QTextEdit(objectName="explain", readOnly=True)
        self.expl_txt.setFixedHeight(104)

        inner = QVBoxLayout()
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(12)
        inner.addLayout(head)
        inner.addWidget(self._separator())
        inner.addWidget(self.pv_lbl)
        inner.addWidget(self.expl_txt)
        inner.addStretch()

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
        self.card = QFrame(objectName="card")
        self.card.setLayout(card_layout)
        # L'effet est parenté à la carte : sans référence côté Python, il est
        # détruit dès la fin de l'expression et l'ombre disparaît.
        self.card_shadow = QGraphicsDropShadowEffect(self.card)
        self.card_shadow.setBlurRadius(34)
        self.card_shadow.setOffset(0, 6)
        self.card.setGraphicsEffect(self.card_shadow)
        return self.card

    # ── barre d'état ────────────────────────────────────────────────
    def _build_statusbar(self) -> QFrame:
        bar = QFrame(objectName="statusbar")
        bar.setFixedHeight(40)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(9)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_lbl = QLabel(objectName="status_text")
        self.status_lbl.setWordWrap(False)

        self.credits_lbl = QLabel("créé par n1motv", objectName="credits")
        self.credits_icon = QLabel()
        self.credits_icon.setFixedSize(18, 18)
        self.credits_icon.setScaledContents(True)

        row.addWidget(self.status_dot)
        row.addWidget(self.status_lbl, 1)
        row.addWidget(self.credits_lbl)
        row.addWidget(self.credits_icon)
        return bar

    def _connect(self) -> None:
        self.run_btn.clicked.connect(self._toggle_run)
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
        self.tagline_lbl.setText(self.T("app_tagline"))

        self.white_rb.setText(self.T("white_short"))
        self.black_rb.setText(self.T("black_short"))
        self.calib_btn.setText(self.T("calibrate"))
        self.mode_box.setItemText(0, self.T("mode_auto"))
        self.mode_box.setItemText(1, self.T("mode_manual"))
        self.llm_chk.setText(self.T("llm_toggle"))
        self.llm_chk.setToolTip(self.T("llm_toggle_hint"))
        self.flip_chk.setText(self.T("flip_board"))
        self.flip_chk.setToolTip(self.T("flip_board_hint"))

        for key, lbl in self._sections.items():
            lbl.setText(self.T(key).upper())
        for _combo, key in SHORTCUTS:
            self.sc_labels[key].setText(self.T(key))

        self.idle_title.setText(self.T("idle_title"))
        self.idle_hint.setText(self.T("idle_hint"))
        # la traduction peut occuper une ligne de plus (ru, ar) : on réserve
        # la hauteur réellement nécessaire à cette largeur.
        self.idle_hint.setMinimumHeight(self.idle_hint.heightForWidth(HINT_W))

        self.run_btn.setToolTip(f"{self.T('sc_startstop')} · Space")
        self.white_rb.setToolTip(f"{self.T('white')} · W")
        self.black_rb.setToolTip(f"{self.T('black')} · B")
        self.mode_box.setToolTip(f"{self.T('sc_mode')} · M")
        self.theme_btn.setToolTip(f"{self.T('sc_theme')} · C")
        self.calib_btn.setToolTip(f"{self.T('sc_calibrate')} · R")

        self._refresh_run_button()

    # ── thème ───────────────────────────────────────────────────────
    def _refresh_window_icon(self) -> None:
        logo = "chessmate.png" if self.is_dark else "chessmate_light.png"
        path = ASSETS / logo
        if path.exists():
            self.setWindowIcon(QIcon(str(path)))

    def _refresh_theme_icon(self) -> None:
        icon = self.moon_icon if self.is_dark else self.sun_icon
        if icon and not icon.isNull():
            self.theme_btn.setIcon(icon)
            self.theme_btn.setIconSize(QSize(18, 18))
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

        logo_path = ASSETS / ("chessmate.png" if dark else "chessmate_light.png")
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            self.logo_lbl.setPixmap(pix)
            self.credits_icon.setPixmap(pix)

        # feuille régénérée intégralement (jamais concaténée) et posée sur
        # l'application : les popups de QComboBox et les infobulles sont des
        # fenêtres à part, elles ne verraient pas un style posé sur `self`.
        app = QApplication.instance()
        (app or self).setStyleSheet(theme.stylesheet(dark))

        self.card_shadow.setColor(QColor(0, 0, 0, 150 if dark else 45))
        # la barre d'état est la seule à porter un style « inline » (couleur
        # variable selon le ton du message) : elle se repeint à la main.
        # Le reste est déjà repoli par `setStyleSheet`, y compris les
        # sélecteurs à propriété — un unpolish/polish manuel sur toute la
        # hiérarchie était non seulement inutile mais décalait la carte.
        self._apply_status_style()

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

    def _refresh_run_button(self) -> None:
        running = self._running()
        self.run_btn.setText(self.T("stop") if running else self.T("start"))
        self.run_btn.setProperty("running", "true" if running else "false")
        style = self.run_btn.style()
        style.unpolish(self.run_btn)
        style.polish(self.run_btn)

    def _set_busy(self, busy: bool) -> None:
        """Bandeau d'attente : visible seulement tant qu'aucun coup n'est affiché."""
        self.progress.setVisible(busy)
        self.idle_hint.setVisible(not busy)
        self.idle_title.setText(self.T("status_thinking") if busy
                                else self.T("idle_title"))

    def start(self) -> None:
        if self._running():
            return
        self.analyzer.reset()
        self.set_status("")
        self.stack.setCurrentIndex(0)
        self._set_busy(True)

        self.thread = AnalysisThread(self.analyzer, self)
        self.thread.configure(self._side(), self.lang, self.flip_chk.isChecked())
        self.thread.resultReady.connect(self.on_result)
        self.thread.failed.connect(self.on_failed)
        self.thread.statusChanged.connect(self.on_status_key)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()
        self._refresh_run_button()

    def stop(self) -> None:
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.stop()
            if not thread.wait(3000):
                log.warning("Le thread d'analyse n'a pas répondu — terminaison")
                thread.terminate()
                thread.wait(500)
        self.overlay.clear()
        self._set_busy(False)
        self._refresh_run_button()

    def _on_thread_finished(self) -> None:
        self._set_busy(False)
        self._refresh_run_button()

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
                               left=geom.left, top=geom.top), ok=True)
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
                   transient: bool = False, ok: bool = False) -> None:
        tone = "danger" if error else ("success" if ok
                                       else ("muted" if transient else "accent"))
        self._status = (text, tone)
        self.status_lbl.setProperty("sticky", error)
        self._apply_status_style()

    def _apply_status_style(self) -> None:
        text, tone = self._status
        t = theme.tokens(self.is_dark)
        color = {"danger": t["danger"], "success": t["success"],
                 "muted": t["text_muted"], "accent": t["accent"]}[tone]
        self.status_lbl.setText(text or self.T("status_ready"))
        self.status_lbl.setStyleSheet(f"color:{color if text else t['text_faint']};")
        self.status_dot.setStyleSheet(
            f"background:{color if text else t['text_faint']}; border-radius:4px;")

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
        self._set_busy(False)
        self.icon_lbl.setPixmap(theme.rounded(icon_for(a.piece).scaled(
            ICON_SZ, ICON_SZ, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self.move_lbl.setText(a.san)
        self.path_lbl.setText(f"{a.src}  →  {a.dst}")

        board_after = a.board_after
        if board_after.is_checkmate():
            self.eval_chip.setText(self.T("checkmate"))
        elif board_after.is_stalemate():
            self.eval_chip.setText(self.T("stalemate"))
        else:
            self.eval_chip.setText(
                f"{self.T('eval')} {format_score(a.score_pawns, a.mate_in)}")
        self.conf_chip.setText(f"{self.T('confidence')} {a.confidence:.0%}")

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

        self.stack.setCurrentWidget(self.card)
        self._fade_in()

    def _fade_in(self) -> None:
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
    app.setFont(theme.font(10))

    window = ChessHelper()
    window.show()
    # Plus d'os._exit() : le moteur est fermé par closeEvent + atexit, donc
    # une sortie normale suffit et les tampons sont bien vidés.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
