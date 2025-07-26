# ───────────  main.py  ────────────────────────────────────────────────
import sys, pathlib, concurrent.futures, traceback
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QRadioButton, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect, QProgressBar, QComboBox
)
from PySide6.QtGui  import QPixmap, QFont, QIcon, QColor
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
import qt_material

import worker                      # ← thread + Stockfish/LLM

ICON_SZ = 136
ASSETS  = pathlib.Path("assets")
if not ASSETS.exists():
    raise FileNotFoundError("Place PNG icons in ./assets/")

def icon_for(piece: str) -> QPixmap:
    fname = {
        "pawn":"pawn",  "knight":"knight", "bishop":"bishop",
        "rook":"rook",  "queen":"queen",   "king":"king"
    }.get(piece, "pawn") + ".png"
    return QPixmap(str(ASSETS / fname))


# ─────────────────────────────────────────────────────────────────────
class ChessHelper(QWidget):
    pool  = concurrent.futures.ThreadPoolExecutor(1)
    task  = None
    lang  = "fr"                     # 'fr' ou 'en'

    # ───────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        qt_material.apply_stylesheet(self, theme="dark_teal.xml")

        self.setWindowTitle("ChessMate · Your AI friend on every move")
        self.setFixedWidth(560)
        self.setWindowIcon(QIcon("assets/chessmate.png"))

        # ===== Sélecteur de langue  =================================
        self.lang_box = QComboBox()
        self.lang_box.addItems(["Français", "English"])
        self.lang_box.currentIndexChanged.connect(self.on_lang_change)

        # ===== Choix du côté ========================================
        self.white_rb = QRadioButton()
        self.black_rb = QRadioButton()
        self.white_rb.setChecked(True)

        # ===== Boutons ==============================================
        self.start_btn = QPushButton()
        self.stop_btn  = QPushButton(); self.stop_btn.setEnabled(False)

        # ===== Loader ===============================================
        self.progress  = QProgressBar(alignment=Qt.AlignCenter)
        self.progress.setRange(0, 0); self.progress.setTextVisible(False)
        self.progress.setFixedWidth(300)
        self.progress.setStyleSheet(
            "QProgressBar{background:rgba(255,255,255,0.06);"
            "border:1px solid #00e8ff;border-radius:7px;}"
            "QProgressBar::chunk{background:#00e8ff;border-radius:6px;}"
        )
        self.wait_lbl  = QLabel(alignment=Qt.AlignCenter)
        load_lay = QVBoxLayout(); load_lay.addWidget(self.progress); load_lay.addWidget(self.wait_lbl)
        self.load_frame = QFrame(); self.load_frame.setLayout(load_lay); self.load_frame.setVisible(False)

        # ===== Carte résultat =======================================
        self.card      = QFrame()
        self.card.setStyleSheet(
            "QFrame{background:rgba(255,255,255,0.04);"
            "border:1px solid rgba(0,232,255,0.4);border-radius:14px;}")
        self.card.setGraphicsEffect(QGraphicsDropShadowEffect(
            blurRadius=25, xOffset=0, yOffset=3, color=QColor(0,0,0,180)))

        self.icon_lbl  = QLabel(alignment=Qt.AlignCenter); self.icon_lbl.setFixedSize(ICON_SZ, ICON_SZ)
        self.move_lbl  = QLabel(alignment=Qt.AlignCenter)
        self.path_lbl  = QLabel(alignment=Qt.AlignCenter)
        self.eval_lbl  = QLabel(alignment=Qt.AlignCenter)
        self.expl_txt  = QTextEdit(readOnly=True); self.expl_txt.setFixedHeight(100); self.expl_txt.setStyleSheet("background:transparent;")

        card_lay = QVBoxLayout(self.card); card_lay.setContentsMargins(24,24,24,24); card_lay.setSpacing(6)
        for w in (self.icon_lbl, self.move_lbl, self.path_lbl, self.eval_lbl, self.expl_txt):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); card_lay.addWidget(w, alignment=Qt.AlignHCenter)
        self.card.setVisible(False)

        # ===== Layout racine ========================================
        root = QVBoxLayout(self); root.setSpacing(12); root.setContentsMargins(14,14,14,14)

        head = QHBoxLayout(); head.addWidget(self.white_rb); head.addWidget(self.black_rb); head.addStretch(); head.addWidget(self.lang_box)
        root.addLayout(head)
        root.addWidget(self.start_btn); root.addWidget(self.stop_btn)
        root.addWidget(self.load_frame, alignment=Qt.AlignHCenter)
        root.addWidget(self.card)

        # signaux
        self.start_btn.clicked.connect(self.start)
        self.stop_btn .clicked.connect(self.stop)

        self.timer = QTimer(self, interval=1600); self.timer.timeout.connect(self.tick)

        # applique le texte initial (français)
        self.apply_language()

    # ===== INTERNATIONALISATION =====================================
    def on_lang_change(self, idx:int):
        self.lang = "fr" if idx==0 else "en"
        self.apply_language()

    def apply_language(self):
        tr_fr = {
            "white"     : "Je joue les Blancs",
            "black"     : "Je joue les Noirs",
            "start"     : "⏵  LANCER L’ANALYSE",
            "stop"      : "⏹  STOPPER",
            "waiting"   : "Analyse en cours…",
            "bestMove"  : "Meilleur coup",
            "bestPath"  : "Meilleur déplacement",
            "eval"      : "Évaluation",
            "error"     : "Erreur",
        }
        tr_en = {
            "white"     : "I play White",
            "black"     : "I play Black",
            "start"     : "⏵  START ANALYSIS",
            "stop"      : "⏹  STOP",
            "waiting"   : "Analysing…",
            "bestMove"  : "Best move",
            "bestPath"  : "Best path",
            "eval"      : "Eval",
            "error"     : "Error",
        }
        T = tr_fr if self.lang=="fr" else tr_en

        self.white_rb.setText(T["white"]); self.black_rb.setText(T["black"])
        for rb in (self.white_rb, self.black_rb): rb.setFont(QFont("Inter", 14, QFont.Bold))
        self.start_btn.setText(T["start"]); self.stop_btn.setText(T["stop"])
        self.wait_lbl.setText(T["waiting"])
        self.move_lbl.setProperty("labelKey", "bm")     # mémorise pour plus tard
        self.path_lbl.setProperty("labelKey", "bp")
        self.eval_lbl.setProperty("labelKey", "ev")

        # refresh fonts (plus gros)
        self.move_lbl.setFont(QFont("Inter", 34, QFont.Black))
        self.path_lbl.setFont(QFont("Inter", 22, QFont.DemiBold))
        self.eval_lbl.setFont(QFont("Inter", 18))

    # ===== Contrôles start / stop ====================================
    def start(self):
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.card.setVisible(False); self.load_frame.setVisible(True)
        self.task = None; self.timer.start()

    def stop(self):
        self.timer.stop(); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.load_frame.setVisible(False); self.task = None

    # ===== Thread & affichage ========================================
    def tick(self):
        if self.task is None:                 # lancement
            side = "white" if self.white_rb.isChecked() else "black"
            self.task = self.pool.submit(worker.analyse, side, self.lang)
        elif self.task.done():                # résultat prêt
            try:
                fen, mv, piece, src, dst, score, expl = self.task.result()
                self.display(mv, piece, src, dst, score, expl)
            except Exception as err:
                self.display_error(str(err))
            finally:
                self.stop()

    def display(self, mv, piece, src, dst, score, expl):
        T = {"fr": ("Meilleur coup", "Meilleur déplacement", "Évaluation"),
             "en": ("Best move", "Best path", "Eval")}[self.lang]
        self.icon_lbl.setPixmap(icon_for(piece).scaled(
            ICON_SZ, ICON_SZ, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.move_lbl.setText(f"{T[0]} : <b>{mv}</b>")
        self.path_lbl.setText(f"{T[1]} : <b>{src}  →  {dst}</b>")
        self.eval_lbl.setText(f"{T[2]} : {score:+.2f}")

        if expl.strip():
            self.expl_txt.setPlainText(expl)
            self.expl_txt.show()
        else:
            self.expl_txt.hide()

        self.fade_in(self.card)

    def display_error(self, txt):
        self.icon_lbl.clear()
        self.move_lbl.setText({"fr":"Erreur", "en":"Error"}[self.lang])
        self.path_lbl.clear(); self.eval_lbl.clear()
        self.expl_txt.setPlainText(txt); self.expl_txt.show()
        self.fade_in(self.card)

    @staticmethod
    def fade_in(widget: QWidget):
        widget.setVisible(True); widget.setWindowOpacity(0)
        QPropertyAnimation(widget, b"windowOpacity",
                           duration=350, startValue=0, endValue=1,
                           easingCurve=QEasingCurve.OutCubic
                           ).start(QPropertyAnimation.DeleteWhenStopped)

    def closeEvent(self, ev): self.stop(); super().closeEvent(ev)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = ChessHelper(); win.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
