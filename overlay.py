"""
overlay.py – flèche de surlignage dessinée par-dessus le plateau.

Remplace `ui_overlay.py` (`ArrowOverlay`), qui n'était importé nulle part, et
corrige le `HighlightOverlay` de l'ancien `main.py` :

* l'ancien widget recevait `self` (la fenêtre principale) comme **parent**
  tout en se voyant appliquer `Qt.FramelessWindowHint | Qt.Tool`.  Ces
  drapeaux en font une fenêtre de premier niveau, dont `setGeometry()` attend
  des coordonnées **globales** — or on lui passait le résultat de
  `mapFromGlobal()`, calculé qui plus est avant que la fenêtre soit affichée.
  La flèche tombait donc systématiquement à côté du plateau ;
* la géométrie est désormais recalculée à chaque coup à partir du rectangle
  courant du plateau, et convertie en pixels logiques Qt (écrans HiDPI).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

import theme

#: la flèche se pose sur le plateau réel, pas sur l'interface : on prend
#: toujours l'accent « sombre », le plus lumineux des deux, pour rester
#: lisible quel que soit le thème du site d'échecs.
ARROW_COLOR = theme.DARK["accent"]


class BoardOverlay(QWidget):
    """Fenêtre transparente, sans bordure, qui ne capte jamais la souris."""

    def __init__(self) -> None:
        super().__init__(None)                       # ← aucun parent : top-level
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self._square = 0.0
        self._src: QPointF | None = None
        self._dst: QPointF | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    # ── placement ───────────────────────────────────────────────────
    def _device_ratio(self) -> float:
        screen = self.screen() or (self.windowHandle().screen()
                                   if self.windowHandle() else None)
        return float(screen.devicePixelRatio()) if screen else 1.0

    def show_move(self, geometry, src: str, dst: str,
                  *, flipped: bool = False, duration_ms: int = 2500) -> None:
        """
        Affiche une flèche de `src` vers `dst` sur le plateau décrit par
        `geometry` (pixels physiques, coordonnées écran absolues).
        """
        ratio = self._device_ratio()
        left = int(geometry.left / ratio)
        top = int(geometry.top / ratio)
        size = int(geometry.size / ratio)
        self._square = size / 8.0

        self.setGeometry(left, top, size, size)      # coordonnées globales
        self._src = self._center(src, flipped)
        self._dst = self._center(dst, flipped)

        self.show()
        self.raise_()
        self.update()
        self._timer.start(max(200, duration_ms))

    def _center(self, name: str, flipped: bool) -> QPointF:
        col = ord(name[0].lower()) - ord("a")
        row = 8 - int(name[1])
        if flipped:
            col, row = 7 - col, 7 - row
        return QPointF((col + 0.5) * self._square, (row + 0.5) * self._square)

    def clear(self) -> None:
        self._timer.stop()
        self._src = self._dst = None
        self.hide()

    # ── rendu ───────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:
        if self._src is None or self._dst is None or self._square <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        width = max(3.0, self._square * 0.09)
        pen = QPen(QColor(ARROW_COLOR), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        radius = self._square * 0.40
        painter.drawEllipse(self._src, radius, radius)

        angle = math.atan2(self._dst.y() - self._src.y(),
                           self._dst.x() - self._src.x())
        # on arrête le trait au bord du cercle pour ne pas le traverser
        start = QPointF(self._src.x() + radius * math.cos(angle),
                        self._src.y() + radius * math.sin(angle))
        painter.drawLine(start, self._dst)

        head = self._square * 0.32
        for sign in (1, -1):
            theta = angle + sign * math.radians(28)
            painter.drawLine(
                self._dst,
                QPointF(self._dst.x() - head * math.cos(theta),
                        self._dst.y() - head * math.sin(theta)),
            )
