# main.py  – UI + orchestration  (all-in-one, colours refreshed, dual theme)
import sys, locale, pathlib, traceback, json, math, concurrent.futures, chess, os
import pyautogui
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QRadioButton, QFrame, QProgressBar,
    QComboBox, QGraphicsDropShadowEffect
)
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QGridLayout   # ↖️  ajout d’import tout en haut
from PySide6.QtGui  import (
    QPixmap, QFont, QIcon, QColor, QPainter, QPen
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QPoint, QPointF, QCoreApplication, QSize
)
import qt_material
import worker
from vision    import detect_board_region
from autoplay  import play_move

# ───────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ───────────────────────────────────────────────────────────────
ACCENT    = "#00e8ff"     # turquoise from logo
BG_DARK   = "#08121d"     # dark background
TEXT_DARK = "#1e2a38"     # for light theme text
BG_LIGHT  = "#dce3eb"     # less harsh light background (attenuated)
CTRL_BG      = "#c9d0db"   # + foncé qu’avant
CTRL_BORDER  = "#8d99a6"   # gris moyen
CTRL_DISABLED = "#d4d8df"  # pour le bouton STOP


ICON_SZ   = 136
ASSETS    = pathlib.Path("assets")
FLAGS_DIR = ASSETS / "flags"
LANG_DIR  = pathlib.Path("lang")
if not ASSETS.exists() or not LANG_DIR.exists():
    raise FileNotFoundError("Dossiers assets/ et lang/ requis.")

# high-dpi acceleration
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# piece icons
piece_icon_map = {
    "pawn":"pawn","pion":"pawn", "knight":"knight","cavalier":"knight",
    "bishop":"bishop","fou":"bishop", "rook":"rook","tour":"rook",
    "queen":"queen","dame":"queen", "king":"king","roi":"king",
}
def icon_for(name:str)->QPixmap:
    return QPixmap(str(ASSETS/(piece_icon_map.get(name,"pawn")+".png")))

# ───────────────── i18n helper (lazy load) ───────────────────────
_translation_cache: dict[str, dict] = {}
def load_translations(code: str) -> dict:
    if code not in _translation_cache:
        path = LANG_DIR / f"{code}.json"
        try:
            with open(path, encoding="utf-8") as f:
                _translation_cache[code] = json.load(f)
        except Exception:
            _translation_cache[code] = {}
    return _translation_cache[code]

def tr(code: str, key: str) -> str:
    return load_translations(code).get(key, key)

# ─────────────────── overlay highlight ──────────────────────────
class HighlightOverlay(QWidget):
    def __init__(self, parent, square:int):
        super().__init__(parent)
        self.square = square
        self.src_c  = self.dst_c = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        self.hide()

    def highlight(self, src:str, dst:str, duration=2000):
        f0,r0 = ord(src[0])-97, 8-int(src[1])
        f1,r1 = ord(dst[0])-97, 8-int(dst[1])
        self.src_c = QPointF(f0*self.square + self.square/2,
                             r0*self.square + self.square/2)
        self.dst_c = QPointF(f1*self.square + self.square/2,
                             r1*self.square + self.square/2)
        self.show(); self.update()
        QTimer.singleShot(duration, self.hide)

    def paintEvent(self, _):
        if not (self.src_c and self.dst_c): return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT), 4, Qt.SolidLine, Qt.RoundCap))
        rad = self.square * 0.4
        p.drawEllipse(self.src_c, rad, rad)
        p.drawLine(self.src_c, self.dst_c)
        ang  = math.atan2(self.dst_c.y()-self.src_c.y(),
                          self.dst_c.x()-self.src_c.x())
        L = self.square * 0.3
        for s in (1, -1):
            t = ang + s * math.radians(25)
            x = self.dst_c.x() - L * math.cos(t)
            y = self.dst_c.y() - L * math.sin(t)
            p.drawLine(self.dst_c, QPointF(x, y))

# ─────────────────── fenêtre principale ─────────────────────────
class ChessHelper(QWidget):
    pool = concurrent.futures.ThreadPoolExecutor(1)
    task = None

    def __init__(self):
        super().__init__()

        # état du thème
        self.is_dark = True  # démarrage sombre

        # icônes thème (sun / moon) — fallback aux symboles
        self.sun_icon = QIcon(str(ASSETS/"sun.png")) if (ASSETS/"sun.png").exists() else None
        self.moon_icon = QIcon(str(ASSETS/"moon.png")) if (ASSETS/"moon.png").exists() else None

        # style initial léger pour accélérer affichage
        self.base_quick_style()

        # i18n ----------------------------------------------------
        codes = ["fr","en","es","ru","zh","ar"]
        sys_loc = (locale.getdefaultlocale()[0] or "").split("_")[0].lower()
        self.lang = sys_loc if sys_loc in codes else "en"

        # fenêtre -------------------------------------------------
        self.setWindowTitle("ChessMate · your AI friend on every move")
        self.setFixedWidth(800)
        logo_path = "chessmate_light.png" if not self.is_dark else "chessmate.png"
        self.setWindowIcon(QIcon(str(ASSETS / logo_path)))

        # overlay -------------------------------------------------
        _, bx, by, bw, _ = detect_board_region()
        square = bw // 8
        top_left = self.mapFromGlobal(QPoint(bx, by))
        self.overlay = HighlightOverlay(self, square)
        self.overlay.setGeometry(top_left.x(), top_left.y(), bw, bw)

        # contrôles ----------------------------------------------
        self.white_rb, self.black_rb = QRadioButton(), QRadioButton()
        self.white_rb.setChecked(True)
        for rb in (self.white_rb, self.black_rb):
            rb.setStyleSheet("QRadioButton{spacing:4px;}")

        self.mode_box = QComboBox()
        self.mode_box.addItem(tr(self.lang, "mode_auto"), "auto")
        self.mode_box.addItem(tr(self.lang, "mode_manual"), "manual")
        self.mode_box.setFixedWidth(140)

        self.lang_box = QComboBox()
        flag_map = {
            "en": "us.png",
            "fr": "fr.png",
            "ar": "ma.png",
            "es": "es.png",
            "ru": "ru.png",
            "zh": "cn.png"
        }
        names = {"fr":"Français","en":"English","es":"Español","ru":"Русский","zh":"中文","ar":"العربية"}
        for code in ["fr","en","es","ru","zh","ar"]:
            display = names.get(code, code)
            icon_path = FLAGS_DIR / flag_map.get(code, "")
            icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
            self.lang_box.addItem(icon, display, userData=code)
        self.lang_box.setCurrentIndex(["fr","en","es","ru","zh","ar"].index(self.lang))
        self.lang_box.currentIndexChanged.connect(self.on_lang_change)

        # thème switch
        self.theme_btn = QPushButton()
        self.theme_btn.setFixedSize(32, 32)
        self.theme_btn.setToolTip("Toggle theme (C)")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.update_theme_icon()

        self.start_btn, self.stop_btn = QPushButton(), QPushButton()
        self.start_btn.setToolTip("Start analysis (Space)")
        self.stop_btn.setToolTip("Stop analysis (Space)")
        self.white_rb.setToolTip("Play as White (W)")
        self.black_rb.setToolTip("Play as Black (B)")
        self.mode_box.setToolTip("Toggle mode (M)")

        self.stop_btn.setEnabled(False)

        # spinner ----------------------------------------------------
        self.progress = QProgressBar(alignment=Qt.AlignCenter)
        self.progress.setRange(0,0); self.progress.setFixedWidth(300)
        self.progress.setTextVisible(False)

        self.wait_lbl = QLabel(alignment=Qt.AlignCenter)
        lf = QVBoxLayout(); lf.addWidget(self.progress); lf.addWidget(self.wait_lbl)
        self.load_frame = QFrame(); self.load_frame.setLayout(lf); self.load_frame.hide()

        # info card --------------------------------------------------
        self.icon_lbl = QLabel(alignment=Qt.AlignCenter); self.icon_lbl.setFixedSize(ICON_SZ, ICON_SZ)
        self.move_lbl = QLabel(alignment=Qt.AlignCenter)
        self.path_lbl = QLabel(alignment=Qt.AlignCenter)
        self.eval_lbl = QLabel(alignment=Qt.AlignCenter)
        self.expl_txt = QTextEdit(readOnly=True); self.expl_txt.setFixedHeight(100)
        self.expl_txt.setStyleSheet("background:transparent;")

        card_layout = QVBoxLayout(); card_layout.setContentsMargins(24,24,24,24); card_layout.setSpacing(6)
        for w in (self.icon_lbl, self.move_lbl, self.path_lbl, self.eval_lbl, self.expl_txt):
            card_layout.addWidget(w, alignment=Qt.AlignHCenter)
        self.card = QFrame(); self.card.setLayout(card_layout)
        self.card.setObjectName("info_card")
        self.card.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=25, xOffset=0, yOffset=3,
                                                              color=QColor(0,0,0,180)))
        self.card.hide()
        # ───────── barre "créé par…" ─────────
        self.credits_bar = QFrame()
        self.credits_bar.setObjectName("credits_bar")

        BAR_H   = 64          # 60 px d’icône + 2 px marges haut/bas + 2 px jeu

        self.credits_bar.setFixedHeight(BAR_H)
        self.credits_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        grid = QGridLayout(self.credits_bar)
        grid.setContentsMargins(20, 2, 20, 2)        # garde 2 px haut/bas
        grid.setHorizontalSpacing(6)

        txt_lbl = QLabel("créé par n1motv 😎")
        txt_lbl.setFont(QFont("Inter", 20, QFont.DemiBold))

        icon_lbl = QLabel(objectName="credits_icon")
        icon_lbl.setFixedSize(60, 60)
        icon_lbl.setAlignment(Qt.AlignCenter)        # ← centre le pixmap dans son QLabel

        # IMPORTANT : ajout d’AlignVCenter
        grid.addWidget(txt_lbl,  0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(icon_lbl, 0, 2, alignment=Qt.AlignRight | Qt.AlignVCenter)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)



        # sidebar raccourcis
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(160)
        sidebar.setStyleSheet("QFrame{background:rgba(255,255,255,0.03);border-radius:8px;}")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(12,12,12,12)
        sb_layout.setSpacing(6)
        title = QLabel("Shortcuts")
        title.setFont(QFont("Inter",12,QFont.Bold))
        sb_layout.addWidget(title)
        def make_item(key, desc):
            lbl = QLabel(f"<b>{key}</b> {desc}")
            lbl.setWordWrap(True)
            lbl.setFont(QFont("Inter",9))
            sb_layout.addWidget(lbl)
        make_item("Space :", "Start / Stop")
        make_item("W :", "Blancs")
        make_item("B :", "Noirs")
        make_item("M :", "Auto / Highlight")
        make_item("C :", "Changer thème")
        sb_layout.addStretch()

        # root layout --------------------------------------------
        root = QVBoxLayout(self); root.setSpacing(12); root.setContentsMargins(14,14,14,14)
        head = QHBoxLayout()
        head.addWidget(sidebar)
        head.addWidget(self.white_rb); head.addWidget(self.black_rb)
        head.addStretch(); head.addWidget(self.mode_box); head.addWidget(self.lang_box); head.addWidget(self.theme_btn)
        root.addLayout(head)
        root.addWidget(self.start_btn); root.addWidget(self.stop_btn)
        root.addWidget(self.load_frame, alignment=Qt.AlignHCenter)
        root.addWidget(self.card)
        root.addWidget(self.credits_bar, alignment=Qt.AlignBottom)

        # signaux -------------------------------------------------
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.timer = QTimer(self, interval=10); self.timer.timeout.connect(self.tick)

        self.apply_language()
        QTimer.singleShot(0, self.finish_setup)  # thème complet post-init

        # raccourcis clavier
        self.setFocusPolicy(Qt.StrongFocus)

    def base_quick_style(self):
        """Style minimal pour accélérer le premier rendu."""
        self.setStyleSheet(f"""
            QWidget {{ background:{BG_DARK}; color:#e0e0e0; }}
            QPushButton {{ background:{ACCENT}; color:{BG_DARK}; border:none; border-radius:6px; padding:4px 10px; }}
            QProgressBar {{ border:1px solid {ACCENT}; border-radius:6px; }}
        """)

    def finish_setup(self):
        """Applique thème complet (Qt-Material + retouches)."""
        self.apply_theme(self.is_dark)

    def update_theme_icon(self):
        if self.is_dark:
            if self.moon_icon and not self.moon_icon.isNull():
                self.theme_btn.setIcon(self.moon_icon)
                self.theme_btn.setIconSize(QSize(20,20))
                self.theme_btn.setText("")
            else:
                self.theme_btn.setText("☾")
        else:
            if self.sun_icon and not self.sun_icon.isNull():
                self.theme_btn.setIcon(self.sun_icon)
                self.theme_btn.setIconSize(QSize(20,20))
                self.theme_btn.setText("")
            else:
                self.theme_btn.setText("☀")

    def apply_theme(self, dark: bool):
        self.is_dark = dark
        self.refresh_logos()
        if dark:
            qt_material.apply_stylesheet(
                self, theme="dark_teal.xml",
                extra={
                    "primaryColor"      : ACCENT,
                    "accentLightColor"  : ACCENT,
                    "secondaryTextColor": "#cfd8dc",
                    "background"        : BG_DARK
                },
            )
            text_color = "#e0e0e0"
            bg = BG_DARK
            sidebar_bg = "rgba(255,255,255,0.03)"
            sidebar_border = f"1px solid rgba(255,255,255,0.08)"
            button_border = "none"
        else:          #  ← branche thème clair
            qt_material.apply_stylesheet(
                self, theme="light_blue.xml" if hasattr(qt_material, "build_stylesheet") else "dark_teal.xml",
                extra={
                    "primaryColor"      : ACCENT,
                    "accentLightColor"  : ACCENT,
                    "secondaryTextColor": "#5f6f8c",
                    "background"        : BG_LIGHT
                }
            )
            text_color   = TEXT_DARK
            bg           = BG_LIGHT
            sidebar_bg     = CTRL_BG
            sidebar_border = f"1px solid {CTRL_BORDER}"
            button_border  = f"1px solid {CTRL_BORDER}"

        # mise à jour icône thème
        self.update_theme_icon()

        # overrides communs
        # overrides communs
        self.setStyleSheet(self.styleSheet() + f"""
            QWidget {{ background:{bg}; color:{text_color}; }}

            /* Boutons actifs */
            QPushButton {{
                background:{ACCENT}; color:{BG_DARK}; font-weight:bold;
                border:{button_border}; border-radius:8px; padding:6px 14px;
            }}
            /* Boutons désactivés */
            QPushButton:disabled {{
                background:{CTRL_DISABLED};
                color:#8a8a8a;
                border:{button_border};
            }}

            /* Cadres / cartes / barre crédits / panneau shortcuts */
            QFrame#info_card,
            QFrame#credits_bar,
            QFrame#sidebar {{
                background:{sidebar_bg};
                border:{sidebar_border};
                border-radius:8px;
            }}

            /* Texte dans la barre crédits */
            QFrame#credits_bar QLabel {{
                background:transparent;
                font-size:20px;
                font-weight:600;
            }}
        """)



    def toggle_theme(self):
        self.apply_theme(not self.is_dark)

    def refresh_logos(self):
        # chemin selon le thème
        logo_file = "chessmate.png" if self.is_dark else "chessmate_light.png"

        # icône de fenêtre
        self.setWindowIcon(QIcon(str(ASSETS / logo_file)))

        # logo dans la barre crédits
        label = self.findChild(QLabel, "credits_icon")
        if label:
            label.setPixmap(
                QPixmap(str(ASSETS / logo_file))
                .scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # i18n -----------------------------------------------------------
    def on_lang_change(self, i):
        self.lang = self.lang_box.itemData(i)
        self.apply_language()

    def T(self, k):
        return tr(self.lang, k)

    def apply_language(self):
        for rb,key in ((self.white_rb,"white"),(self.black_rb,"black")):
            rb.setText(self.T(key)); rb.setFont(QFont("Inter",14,QFont.Bold))
        self.start_btn.setText(self.T("start")); self.stop_btn.setText(self.T("stop"))
        self.wait_lbl.setText(self.T("waiting"))
        self.mode_box.setItemText(0, self.T("mode_auto"))
        self.mode_box.setItemText(1, self.T("mode_manual"))
        self.move_lbl.setFont(QFont("Inter",34,QFont.Black))
        self.path_lbl.setFont(QFont("Inter",22,QFont.DemiBold)); self.eval_lbl.setFont(QFont("Inter",18))

    # start / stop ---------------------------------------------------
    def start(self):
        worker.reset()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.card.hide(); self.load_frame.show()
        self.task = None; self.timer.start(); self._refresh_size()

    def stop(self):
        self.timer.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.load_frame.hide(); self.task = None ; self._refresh_size()

    def _refresh_size(self):
        """Force la fenêtre à se recalc- uler à la taille minimale utile."""
        self.layout().invalidate()          # force le recalcul des hints
        self.adjustSize()                   # adapte la taille de la fenêtre

    def _toggle(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.start()

    def _select_color(self, color: str):
        if color == "white":
            self.white_rb.setChecked(True)
        else:
            self.black_rb.setChecked(True)

    def _toggle_mode(self):
        current = self.mode_box.currentIndex()
        self.mode_box.setCurrentIndex(1 if current == 0 else 0)

    # boucle UI ↔ worker --------------------------------------------
    def tick(self):
        if self.task is None:
            side = "white" if self.white_rb.isChecked() else "black"
            self.task = worker.submit(side)
        elif self.task.done():
            res = self.task.result(); self.task = None
            if not res: return
            fen, mv, piece, src, dst, score, expl = res

            board_after = chess.Board(fen); board_after.push_san(mv)
            mode = self.mode_box.currentData()
            if mode == "auto":         # Auto-play
                play_move(src, dst)
                self.return_cursor_to_ui()
                worker._prev_layout = board_after.fen().split()[0]
            else:                     # Highlight only (manuel)
                self.overlay.highlight(src, dst)
                self.return_cursor_to_ui()
                worker._prev_layout = fen.split()[0]
                self.stop()

            if board_after.is_checkmate():
                self.stop(); self.eval_lbl.setText("Check-mate!")
            else:
                self.eval_lbl.setText(f"{self.T('eval')} : {score:+.2f}")

            self.icon_lbl.setPixmap(icon_for(piece).scaled(
                ICON_SZ, ICON_SZ, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.move_lbl.setText(f"{self.T('bestMove')} : <b>{mv}</b>")
            self.path_lbl.setText(f"{self.T('bestPath')} : <b>{src} → {dst}</b>")
            if expl.strip():
                self.expl_txt.setPlainText(expl); self.expl_txt.show()
            else:
                self.expl_txt.hide()

            self.load_frame.hide(); self._fade(self.card)

    def _fade(self,w):
        w.setVisible(True); w.setWindowOpacity(0)
        QPropertyAnimation(w,b"windowOpacity",duration=350,
            startValue=0,endValue=1,easingCurve=QEasingCurve.OutCubic
        ).start(QPropertyAnimation.DeleteWhenStopped)

    def return_cursor_to_ui(self):
        """Replace la souris sur l'UI pour que l'utilisateur puisse cliquer."""
        top_left = self.mapToGlobal(QPoint(0, 0))
        cx = top_left.x() + self.width() // 2
        cy = top_left.y() + 40
        try:
            pyautogui.moveTo(cx, cy, duration=0.12)
        except Exception:
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._toggle()
        elif event.key() == Qt.Key_W:
            self._select_color("white")
        elif event.key() == Qt.Key_B:
            self._select_color("black")
        elif event.key() == Qt.Key_M:
            self._toggle_mode()
        elif event.key() == Qt.Key_C:
            self.toggle_theme()
        else:
            super().keyPressEvent(event)

    def closeEvent(self,e):
        self.stop()
        try:
            worker.shutdown()
        except Exception:
            pass
        super().closeEvent(e)

# ─────────────────────────── run ─────────────────────────────────
if __name__=="__main__":
    app = QApplication(sys.argv)
    win = ChessHelper()
    win.show()
    try:
        exit_code = app.exec()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    # nettoyage final (au cas où)
    try:
        worker.shutdown()
    except Exception:
        
        pass
    os._exit(exit_code)
