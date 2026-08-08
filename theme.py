"""
theme.py – jetons de design et feuille de style de l'interface.

Auparavant, l'apparence était produite par trois couches qui se battaient :
`qt-material` posait son propre thème complet, `ChessHelper._base_style()`
en écrivait un deuxième, et `apply_theme()` **concaténait** le troisième à
l'existant (`setStyleSheet(self.styleSheet() + …)`), si bien que chaque
changement de thème empilait une règle de plus sans jamais retirer la
précédente. D'où les couleurs incohérentes d'un widget à l'autre et les
contrôles qui gardaient l'ancien fond après un basculement.

Tout part désormais d'ici : une palette par thème, une seule feuille QSS
régénérée intégralement à chaque bascule, et posée sur le `QApplication`
pour que les fenêtres détachées (liste déroulante, infobulle) suivent.
"""
from __future__ import annotations

import functools
import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen, QPixmap,
    QPolygonF,
)

log = logging.getLogger(__name__)

# ── palettes ────────────────────────────────────────────────────────
#: Deux thèmes, **mêmes clés** : aucun appelant n'a à savoir lequel est actif.
DARK: dict[str, str] = {
    "bg":            "#0b1120",   # fond de fenêtre, bleu nuit profond
    "surface":       "#121b2c",   # panneaux (barre latérale, carte)
    "surface_alt":   "#18233a",   # champs, puces, états survolés
    "surface_hi":    "#1f2c47",   # survol d'un élément déjà sur surface_alt
    "border":        "#22304a",
    "border_strong": "#31425f",
    "text":          "#e8eef9",
    "text_muted":    "#93a3bc",
    "text_faint":    "#6b7c96",
    "accent":        "#22d3ee",   # cyan ChessMate, version accessible
    "accent_hover":  "#4fe0f5",
    "accent_press":  "#0aafc9",
    "accent_soft":   "rgba(34, 211, 238, 0.14)",
    "on_accent":     "#04121a",
    "success":       "#34d399",
    "danger":        "#fb7185",
    "warning":       "#fbbf24",
    "shadow":        "rgba(0, 0, 0, 0.55)",
    "disabled_bg":   "#1a2334",
    "disabled_fg":   "#55647d",
}

LIGHT: dict[str, str] = {
    "bg":            "#eef2f9",
    "surface":       "#ffffff",
    "surface_alt":   "#e9eff8",
    "surface_hi":    "#dde7f4",
    "border":        "#d5dfec",
    "border_strong": "#b6c5da",
    "text":          "#0f1b2d",
    "text_muted":    "#55677f",
    "text_faint":    "#7c8ca3",
    "accent":        "#0891b2",   # cyan assombri : lisible sur fond clair
    "accent_hover":  "#0aa5c9",
    "accent_press":  "#0e7490",
    "accent_soft":   "rgba(8, 145, 178, 0.12)",
    "on_accent":     "#ffffff",
    "success":       "#059669",
    "danger":        "#e11d48",
    "warning":       "#d97706",
    "shadow":        "rgba(15, 27, 45, 0.18)",
    "disabled_bg":   "#e4e9f2",
    "disabled_fg":   "#9aa8bd",
}


def tokens(dark: bool) -> dict[str, str]:
    return DARK if dark else LIGHT


def accent(dark: bool = True) -> str:
    return tokens(dark)["accent"]


# ── typographie ─────────────────────────────────────────────────────
#: par ordre de préférence ; l'ancien code demandait « Inter » sans vérifier
#: qu'elle soit installée — sur une machine sans Inter, Qt retombait
#: silencieusement sur une police à empattements qui jurait avec le reste.
_FAMILIES = (
    "Inter", "Inter Display", "Segoe UI Variable Text", "Segoe UI",
    "SF Pro Text", "Noto Sans", "DejaVu Sans",
)


@functools.lru_cache(maxsize=1)
def family() -> str:
    """Première police disponible de `_FAMILIES` (nécessite un QApplication)."""
    available = set(QFontDatabase.families())
    for name in _FAMILIES:
        if name in available:
            return name
    return QFont().defaultFamily()


def font(size: int, weight: QFont.Weight = QFont.Normal, *,
         spacing: float = 0.0) -> QFont:
    fnt = QFont(family(), size)
    fnt.setWeight(weight)
    if spacing:
        fnt.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return fnt


# ── icônes générées (coche des cases à cocher) ──────────────────────
def _icon_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "chessmate-ui"
    path.mkdir(parents=True, exist_ok=True)
    return path


@functools.lru_cache(maxsize=4)
def check_icon(color: str) -> str:
    """
    Coche dessinée à la volée et mise en cache sur disque.

    QSS ne sait pas dessiner de glyphe dans un `::indicator` : sans image, une
    case cochée n'est qu'un carré plein, ambigu au premier coup d'œil. On
    génère donc le pictogramme une fois, dans le dossier temporaire, plutôt
    que d'ajouter un binaire au dépôt. Renvoie une chaîne vide si l'écriture
    échoue — la feuille de style retombe alors sur le carré plein.
    """
    return _stroke_icon(f"check-{color.lstrip('#')}", color, 36, 5.0, [
        (9, 18.5), (15, 25), (27, 11),
    ])


@functools.lru_cache(maxsize=4)
def chevron_icon(color: str) -> str:
    """
    Chevron de la liste déroulante.

    Le triangle en CSS (`border-left/right: transparent`) ne fonctionne pas
    dans QSS : Qt le rendait comme un petit carré plein. `::down-arrow`
    n'accepte réellement qu'une image.
    """
    return _stroke_icon(f"chevron-{color.lstrip('#')}", color, 24, 2.4, [
        (7, 10), (12, 15), (17, 10),
    ])


def _stroke_icon(name: str, color: str, size: int, width: float,
                 points: list[tuple[float, float]]) -> str:
    target = _icon_dir() / f"{name}.png"
    if not target.exists():
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(color), width, Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.end()
        if not pixmap.save(str(target)):
            log.debug("Icône non générée (%s)", target)
            return ""
    return target.as_posix()          # QSS veut des séparateurs « / »


def rounded(pixmap: QPixmap, radius: int = 14) -> QPixmap:
    """
    Arrondit les coins d'une imagette.

    Les icônes de pièces sont des cases découpées sur un plateau : elles ont
    un fond opaque (beige ou brun) qui, posé tel quel dans la carte, y colle
    un rectangle brut. Les coins arrondis les raccordent au reste.
    """
    if pixmap.isNull():
        return pixmap
    out = QPixmap(pixmap.size())
    out.setDevicePixelRatio(pixmap.devicePixelRatio())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(QRectF(pixmap.rect()), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return out


# ── feuille de style ────────────────────────────────────────────────
def stylesheet(dark: bool) -> str:
    """QSS complète du thème demandé. Régénérée entièrement à chaque bascule."""
    t = tokens(dark)
    check = check_icon(t["on_accent"])
    check_rule = (f"image: url({check});" if check else "")
    chevron = chevron_icon(t["text_muted"])
    # sans image utilisable, on laisse Qt dessiner sa flèche native plutôt
    # que d'afficher une zone vide.
    arrow_rule = (f"""
QComboBox::down-arrow {{
    image: url({chevron});
    width: 12px; height: 12px; margin-right: 10px;
}}""" if chevron else "")

    return f"""
/* ── base ──────────────────────────────────────────────────────── */
QWidget {{
    background: transparent;
    color: {t['text']};
    font-family: "{family()}";
    font-size: 13px;
}}
QWidget#root {{ background: {t['bg']}; }}
QToolTip {{
    background: {t['surface_alt']};
    color: {t['text']};
    border: 1px solid {t['border_strong']};
    border-radius: 8px;
    padding: 6px 10px;
}}

/* ── structure ─────────────────────────────────────────────────── */
QFrame#header {{
    background: {t['surface']};
    border: none;
    border-bottom: 1px solid {t['border']};
}}
QFrame#sidebar, QFrame#card, QFrame#placeholder {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 14px;
}}
QFrame#statusbar {{
    background: {t['surface']};
    border: none;
    border-top: 1px solid {t['border']};
}}
QFrame#separator {{
    background: {t['border']};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

/* ── textes ────────────────────────────────────────────────────── */
QLabel#app_title   {{ color: {t['text']}; font-size: 16px; font-weight: 700; }}
QLabel#app_tagline {{ color: {t['text_faint']}; font-size: 12px; }}
QLabel#section     {{ color: {t['text_faint']}; font-size: 10px;
                      font-weight: 700; letter-spacing: 1.2px; }}
QLabel#move        {{ color: {t['text']}; font-size: 30px; font-weight: 800; }}
QLabel#idle_title  {{ color: {t['text']}; font-size: 19px; font-weight: 700; }}
QLabel#idle_glyph  {{ color: {t['border_strong']}; font-size: 76px; }}
QLabel#path        {{ color: {t['text_muted']}; font-size: 15px; font-weight: 600; }}
QLabel#pv          {{ color: {t['text_faint']}; font-size: 12px; }}
QLabel#hint        {{ color: {t['text_muted']}; font-size: 13px; }}
QLabel#credits     {{ color: {t['text_faint']}; font-size: 11px; }}
QLabel#status_text {{ color: {t['text_muted']}; font-size: 12px; }}
QLabel#shortcut    {{ color: {t['text_muted']}; font-size: 11px; }}
QLabel#kbd {{
    background: {t['surface_alt']};
    color: {t['text_muted']};
    border: 1px solid {t['border']};
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#chip {{
    background: {t['surface_alt']};
    color: {t['text_muted']};
    border: 1px solid {t['border']};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
}}
QLabel#chip[tone="accent"] {{
    background: {t['accent_soft']};
    color: {t['accent']};
    border-color: {t['accent']};
}}

/* ── boutons ───────────────────────────────────────────────────── */
QPushButton {{
    background: {t['surface_alt']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover  {{ background: {t['surface_hi']}; border-color: {t['border_strong']}; }}
QPushButton:pressed {{ background: {t['surface_alt']}; }}
QPushButton:disabled {{
    background: {t['disabled_bg']};
    color: {t['disabled_fg']};
    border-color: {t['border']};
}}

QPushButton#primary {{
    background: {t['accent']};
    color: {t['on_accent']};
    border: none;
    border-radius: 12px;
    padding: 13px 18px;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.4px;
}}
QPushButton#primary:hover   {{ background: {t['accent_hover']}; }}
QPushButton#primary:pressed {{ background: {t['accent_press']}; }}
QPushButton#primary[running="true"] {{
    background: {t['danger']};
    color: #ffffff;
}}

QPushButton#icon {{
    background: {t['surface_alt']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 0;
}}
QPushButton#icon:hover {{ background: {t['surface_hi']}; border-color: {t['accent']}; }}

/* segmenté « Blancs / Noirs » : deux boutons bascule dans un QButtonGroup
   exclusif. Un QRadioButton stylé aurait gardé sa pastille (QSS ne sait pas
   la supprimer proprement) et son libellé calé à gauche. */
QPushButton#segment {{
    background: {t['surface_alt']};
    color: {t['text_muted']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 10px 8px;
    font-size: 13px;
    font-weight: 700;
}}
QPushButton#segment:hover {{ color: {t['text']}; background: {t['surface_hi']}; }}
QPushButton#segment:checked {{
    background: {t['accent_soft']};
    color: {t['accent']};
    border: 1px solid {t['accent']};
}}

/* ── cases à cocher ────────────────────────────────────────────── */
QCheckBox {{
    color: {t['text_muted']};
    spacing: 10px;
    font-size: 12px;
    padding: 3px 0;
}}
QCheckBox:hover {{ color: {t['text']}; }}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1.5px solid {t['border_strong']};
    background: {t['surface_alt']};
}}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{
    background: {t['accent']};
    border-color: {t['accent']};
    {check_rule}
}}

/* ── liste déroulante ──────────────────────────────────────────── */
QComboBox {{
    background: {t['surface_alt']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 600;
}}
QComboBox:hover {{ border-color: {t['border_strong']}; }}
QComboBox:focus {{ border-color: {t['accent']}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
{arrow_rule}
QComboBox QAbstractItemView {{
    background: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border_strong']};
    border-radius: 10px;
    padding: 4px;
    outline: none;
    selection-background-color: {t['accent_soft']};
    selection-color: {t['accent']};
}}
QComboBox QAbstractItemView::item {{ padding: 7px 10px; border-radius: 7px; }}

/* ── barre de progression ──────────────────────────────────────── */
QProgressBar {{
    background: {t['surface_alt']};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}

/* ── zone d'explication ────────────────────────────────────────── */
QTextEdit#explain {{
    background: {t['surface_alt']};
    color: {t['text_muted']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
}}

/* ── ascenseurs ────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {t['border_strong']}; border-radius: 4px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {t['text_faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0px; }}
"""
