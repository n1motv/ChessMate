from PySide6.QtWidgets import QWidget
from PySide6.QtGui     import QPainter, QPen, QBrush, QColor
from PySide6.QtCore    import Qt, QRectF

class ArrowOverlay(QWidget):
    """Widget transparent placé au‑dessus du plateau."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.src = self.dst = None          # en pixels

    def show_arrow(self, p1: tuple[int,int], p2: tuple[int,int]):
        self.src, self.dst = p1, p2
        self.repaint()

    def paintEvent(self, e):
        if not self.src or not self.dst:
            return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#00E87A"), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(*self.src, *self.dst)

        # petite tête de flèche
        p.setBrush(QBrush(QColor("#00E87A")))
        head = QRectF(self.dst[0]-6, self.dst[1]-6, 12, 12)
        p.drawEllipse(head)
