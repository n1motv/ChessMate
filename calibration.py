"""
calibration.py – sélection manuelle du plateau à la souris.

Avant ce module, le bouton « Calibrer » et le message d'erreur de
`vision.VisionError` promettaient une calibration manuelle qui n'existait
nulle part : le bouton se contentait de relancer `detect_board_region()`,
c'est-à-dire exactement l'algorithme qui venait d'échouer. Si la détection
automatique ne trouve pas le plateau (p. ex. la fenêtre ChessMate le
recouvre, thème à faible contraste, plateau partiellement hors écran), il
n'y avait aucun filet de sécurité.

`ManualCalibrationOverlay` affiche une fenêtre transparente par-dessus
l'écran réel (comme un outil de capture d'écran) et laisse l'utilisateur
entourer lui-même l'échiquier à la souris.
"""
from __future__ import annotations

import mss
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QKeyEvent, QMouseEvent,
    QPainter, QPen, QRegion,
)
from PySide6.QtWidgets import QWidget

import theme

ACCENT = theme.DARK["accent"]     # tracé sur un voile sombre : accent lumineux
MIN_SELECTION = 20      # px logiques ; en dessous, on ignore (clic accidentel)


class ManualCalibrationOverlay(QWidget):
    """
    Fenêtre transparente couvrant tous les écrans : l'utilisateur clique-glisse
    pour entourer l'échiquier. `selected` renvoie le rectangle tracé en pixels
    logiques Qt, coordonnées globales du bureau (à convertir via
    `rect_to_geometry`). `cancelled` est émis sur Échap, clic droit, ou
    sélection trop petite.
    """
    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self, hint: str) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._hint = hint
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setGeometry(QGuiApplication.primaryScreen().virtualGeometry())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.activateWindow()
        self.raise_()

    # ── interaction ─────────────────────────────────────────────────
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self._finish(cancel=True)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._finish(cancel=True)
            return
        self._origin = self._current = event.globalPosition().toPoint()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._origin is not None:
            self._current = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._origin is None:
            return
        rect = QRect(self._origin, event.globalPosition().toPoint()).normalized()
        if rect.width() < MIN_SELECTION or rect.height() < MIN_SELECTION:
            self._finish(cancel=True)
        else:
            self._finish(cancel=False, rect=rect)

    def _finish(self, *, cancel: bool, rect: QRect | None = None) -> None:
        self._origin = self._current = None
        self.close()
        if cancel:
            self.cancelled.emit()
        else:
            self.selected.emit(rect)

    # ── rendu ───────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        dim = QColor(0, 0, 0, 90)

        if self._origin is not None and self._current is not None:
            rect = QRect(self._origin, self._current).normalized()
            local = QRect(self.mapFromGlobal(rect.topLeft()), rect.size())
            # on assombrit tout sauf la sélection en cours, pour qu'elle
            # reste bien lisible par-dessus le plateau réel
            clip = QRegion(self.rect()).subtracted(QRegion(local))
            painter.setClipRegion(clip)
            painter.fillRect(self.rect(), dim)
            painter.setClipping(False)

            painter.setPen(QPen(QColor(ACCENT), 2))
            painter.drawRect(local)
            painter.setFont(theme.font(10, QFont.DemiBold))
            painter.setPen(QColor(ACCENT))
            painter.drawText(local.bottomLeft() + QPoint(4, 18),
                             f"{rect.width()} × {rect.height()} px")
        else:
            painter.fillRect(self.rect(), dim)

        self._draw_hint(painter)

    def _draw_hint(self, painter: QPainter) -> None:
        """
        Consigne posée dans une pastille sombre plutôt qu'en texte blanc nu :
        par-dessus un plateau clair, l'ancien libellé était illisible.
        """
        font = theme.font(12, QFont.Bold)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text = metrics.elidedText(self._hint, Qt.ElideRight,
                                  int(self.width() * 0.7))
        box = metrics.boundingRect(text).adjusted(-18, -10, 18, 10)
        box.moveCenter(QPoint(self.rect().center().x(), self.rect().top() + 54))

        painter.setPen(QPen(QColor(theme.DARK["border_strong"]), 1))
        painter.setBrush(QColor(11, 17, 32, 235))       # theme.DARK["bg"] opaque
        painter.drawRoundedRect(box, 12, 12)
        painter.setPen(QColor(theme.DARK["text"]))
        painter.drawText(box, Qt.AlignCenter, text)


def rect_to_geometry(rect: QRect) -> tuple[int, int, int, int]:
    """
    Convertit un rectangle en pixels logiques Qt (coordonnées globales du
    bureau) vers `(monitor, left, top, size)` en pixels *physiques*,
    globaux (coin haut-gauche du bureau, tous écrans confondus) — le même
    repère que celui produit par `vision.detect_board_region`. `monitor`
    identifie simplement l'écran mss sur lequel se trouve la sélection, pour
    que la capture d'écran suivante cible le bon.
    """
    screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio()

    phys_left = round(rect.left() * ratio)
    phys_top = round(rect.top() * ratio)
    phys_w = round(rect.width() * ratio)
    phys_h = round(rect.height() * ratio)
    cx, cy = phys_left + phys_w / 2, phys_top + phys_h / 2

    with mss.mss() as sct:
        monitors = sct.monitors

    monitor_idx = 1
    for idx in range(1, len(monitors)):
        m = monitors[idx]
        if (m["left"] <= cx < m["left"] + m["width"]
                and m["top"] <= cy < m["top"] + m["height"]):
            monitor_idx = idx
            break
    else:
        best_dist = None
        for idx in range(1, len(monitors)):
            m = monitors[idx]
            mcx = m["left"] + m["width"] / 2
            mcy = m["top"] + m["height"] / 2
            dist = (mcx - cx) ** 2 + (mcy - cy) ** 2
            if best_dist is None or dist < best_dist:
                monitor_idx, best_dist = idx, dist

    size = round((phys_w + phys_h) / 2)
    return monitor_idx, phys_left, phys_top, size
