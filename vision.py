"""
vision.py – capture d'écran, localisation du plateau et lecture de la position.

Corrections apportées par rapport à la version précédente :

* **plus aucun effet de bord à l'import.**  L'ancien module appelait
  `detect_board_region()` au chargement : un simple `import vision` prenait une
  capture d'écran et plantait (`ValueError: max() arg is an empty sequence`)
  si aucun plateau n'était visible.
* **le rectangle du plateau n'est plus figé au démarrage.**  Il est mis en
  cache, persisté par résolution dans `config.json`, et ré-évaluable à tout
  moment via `invalidate()` / `calibrate()`.
* **détection plus sévère** : au lieu de prendre bêtement le plus grand
  contour de l'écran, on ne retient que les contours quasi-carrés, de taille
  plausible, et on note leur « damier-ité ».
* **orientation gérée** : quand on joue les Noirs, le plateau est affiché
  retourné.  Les 64 cases sont réordonnées en conséquence, ce que l'ancienne
  version ne faisait pas du tout.

    python vision.py            → diagnostic : rectangle détecté + FEN lu
    python vision.py --dump     → écrit aussi les 64 imagettes dans data/
"""
from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PIL import Image

import config
from classifier import get_classifier
from fen import BoardReading, FenTracker, flip_chars, index_to_name
from paths import DATA_DIR

log = logging.getLogger(__name__)


class VisionError(RuntimeError):
    """Le plateau n'a pas pu être localisé ou lu de façon fiable."""


# ── géométrie ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class BoardGeometry:
    """Position et taille du plateau, en pixels écran absolus."""
    left: int
    top: int
    size: int          # le plateau est carré
    monitor: int = 1

    @property
    def square(self) -> float:
        return self.size / 8.0

    def square_to_xy(self, name: str, flipped: bool = False) -> tuple[int, int]:
        """Centre en pixels de la case « e4 » (coordonnées écran absolues)."""
        col = ord(name[0].lower()) - ord("a")
        row = 8 - int(name[1])
        if flipped:                       # plateau vu côté Noirs
            col, row = 7 - col, 7 - row
        return (
            int(self.left + (col + 0.5) * self.square),
            int(self.top + (row + 0.5) * self.square),
        )

    def as_tuple(self) -> tuple[int, int, int]:
        return self.left, self.top, self.size


# ── capture ─────────────────────────────────────────────────────────
def grab_screen(monitor: int = 1) -> np.ndarray:
    """Capture l'écran `monitor` → image BGR."""
    with mss.mss() as sct:
        if monitor >= len(sct.monitors):
            raise VisionError(
                f"Écran {monitor} inexistant "
                f"({len(sct.monitors) - 1} écran(s) détecté(s))."
            )
        shot = sct.grab(sct.monitors[monitor])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def screen_size(monitor: int = 1) -> tuple[int, int]:
    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        return int(mon["width"]), int(mon["height"])


def monitor_origin(monitor: int = 1) -> tuple[int, int]:
    """Coin haut-gauche de l'écran dans l'espace de coordonnées global."""
    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        return int(mon["left"]), int(mon["top"])


# ── détection du plateau ────────────────────────────────────────────
def _checkerboard_score(gray: np.ndarray, x: int, y: int, size: int) -> float:
    """
    Mesure à quel point la zone ressemble à un damier : on compare la
    luminance moyenne des cases « claires » et des cases « sombres ».
    Renvoie 0 pour une zone uniforme, ~1 pour un damier bien contrasté.
    """
    step = size / 8.0
    light, dark = [], []
    for r in range(8):
        for c in range(8):
            cy = int(y + (r + 0.5) * step)
            cx = int(x + (c + 0.5) * step)
            half = max(1, int(step * 0.15))
            patch = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            if patch.size == 0:
                return 0.0
            (light if (r + c) % 2 == 0 else dark).append(float(patch.mean()))
    if not light or not dark:
        return 0.0
    return abs(float(np.mean(light)) - float(np.mean(dark))) / 255.0


def detect_board_region(monitor: int = 1,
                        image: np.ndarray | None = None) -> BoardGeometry:
    """
    Localise l'échiquier à l'écran.

    Contrairement à l'ancien `max(cnts, key=cv2.contourArea)`, on filtre les
    candidats sur leur carré-itude et leur taille, puis on départage au score
    de damier — ce qui évite de sélectionner une fenêtre de navigateur ou un
    fond d'écran plus grand que le plateau.
    """
    img = grab_screen(monitor) if image is None else image
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise VisionError(
            "Aucun contour détecté à l'écran. Le plateau est-il bien visible ?"
        )

    min_side = max(120, min(w_img, h_img) // 12)
    max_side = min(w_img, h_img)

    best: tuple[float, BoardGeometry] | None = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if abs(w - h) > 0.06 * max(w, h):          # doit être quasi carré
            continue
        size = min(w, h)
        score = _checkerboard_score(gray, x, y, size)
        if score < 0.05:                            # zone uniforme → pas un damier
            continue
        # à score comparable, on préfère le plus grand candidat
        weighted = score * (size / max_side) ** 0.5
        if best is None or weighted > best[0]:
            best = (weighted, BoardGeometry(x, y, size, monitor))

    if best is None:
        raise VisionError(
            "Plateau introuvable. Vérifiez qu'un échiquier est visible à "
            "l'écran, ou calibrez-le manuellement (bouton « Calibrer »)."
        )

    log.info("Plateau détecté : %s (score %.3f)", best[1], best[0])
    return best[1]


# ── localisateur avec cache + persistance ───────────────────────────
class BoardLocator:
    """
    Conserve la géométrie du plateau entre deux captures et sait la
    ré-évaluer à la demande.  C'est ce qui remplace les anciennes constantes
    globales LEFT / TOP / SQUARE, figées au démarrage du programme.
    """

    def __init__(self, monitor: int | None = None) -> None:
        self.monitor = monitor if monitor is not None else config.get("monitor", 1)
        self._geom: BoardGeometry | None = None
        self._lock = threading.Lock()

    # ── accès ───────────────────────────────────────────────────────
    def get(self, *, refresh: bool = False) -> BoardGeometry:
        with self._lock:
            if self._geom is not None and not refresh:
                return self._geom

            if not refresh:
                saved = config.get_board_rect(*screen_size(self.monitor))
                if saved:
                    self._geom = BoardGeometry(*saved, monitor=self.monitor)
                    log.info("Plateau repris de config.json : %s", self._geom)
                    return self._geom

            self._geom = detect_board_region(self.monitor)
            config.set_board_rect(*screen_size(self.monitor), *self._geom.as_tuple())
            return self._geom

    def invalidate(self) -> None:
        """Oublie la géométrie : la prochaine lecture relancera la détection."""
        with self._lock:
            self._geom = None

    def recalibrate(self) -> BoardGeometry:
        """Force une nouvelle détection et l'enregistre."""
        return self.get(refresh=True)

    def set_manual(self, left: int, top: int, size: int) -> BoardGeometry:
        """Calibration manuelle (rectangle tracé par l'utilisateur)."""
        with self._lock:
            self._geom = BoardGeometry(left, top, size, self.monitor)
            config.set_board_rect(*screen_size(self.monitor), left, top, size)
            return self._geom


# ── découpe en 64 cases ─────────────────────────────────────────────
def slice_squares(img: np.ndarray, geom: BoardGeometry,
                  *, margin: float = 0.0) -> list[Image.Image]:
    """
    Découpe le plateau en 64 imagettes PIL, de a8 (index 0) à h1 (index 63)
    **telles qu'affichées à l'écran** (l'orientation est traitée plus loin).
    """
    step = geom.square
    pad = step * margin
    squares: list[Image.Image] = []
    board_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = board_rgb.shape[:2]

    for r in range(8):
        for c in range(8):
            x0 = int(round(geom.left + c * step + pad))
            y0 = int(round(geom.top + r * step + pad))
            x1 = int(round(geom.left + (c + 1) * step - pad))
            y1 = int(round(geom.top + (r + 1) * step - pad))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w_img, max(x1, x0 + 1)), min(h_img, max(y1, y0 + 1))
            squares.append(Image.fromarray(board_rgb[y0:y1, x0:x1]))
    return squares


def dump_squares(squares: list[Image.Image], out_dir=DATA_DIR) -> None:
    """Écrit les 64 imagettes sur disque (débogage / constitution de dataset)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(squares):
        im.save(out_dir / f"{i:02}.png")


# ── lecture complète ────────────────────────────────────────────────
@dataclass
class ScreenReading:
    """Résultat d'une lecture d'écran."""
    reading: BoardReading
    geometry: BoardGeometry
    flipped: bool
    chars: list[str | None]
    confidence: float                 # confiance minimale sur les 64 cases
    uncertain: list[str]              # noms des cases douteuses

    @property
    def fen(self) -> str:
        return self.reading.fen

    @property
    def placement(self) -> str:
        return self.reading.placement


def read_board(locator: BoardLocator,
               tracker: FenTracker,
               side: str,
               *,
               flipped: bool | None = None,
               commit: bool = False,
               image: np.ndarray | None = None) -> ScreenReading:
    """
    Capture l'écran et renvoie la position lue, FEN complet compris.

    `side`     : "w" ou "b" — trait à jouer.
    `flipped`  : plateau affiché côté Noirs.  None → déduit de `side`.
    `commit`   : met à jour le suivi de partie (droits de roque, compteurs).
    """
    geom = locator.get()
    img = grab_screen(geom.monitor) if image is None else image

    squares = slice_squares(img, geom)
    if config.get("dump_squares"):
        dump_squares(squares)

    chars, confidences = get_classifier().predict_board(squares)

    if flipped is None:
        flipped = (side == "b")
    if flipped:
        chars = flip_chars(chars)
        confidences = list(reversed(confidences))

    threshold = float(config.get("min_square_confidence", 0.80))
    uncertain = [index_to_name(i) for i, c in enumerate(confidences) if c < threshold]

    max_uncertain = int(config.get("max_uncertain_squares", 0))
    if len(uncertain) > max_uncertain:
        raise VisionError(
            f"Position incertaine : {len(uncertain)} case(s) sous le seuil de "
            f"confiance ({', '.join(uncertain[:8])}"
            f"{'…' if len(uncertain) > 8 else ''}). "
            "Recalibrez le plateau ou réentraînez le modèle sur votre thème."
        )

    reading = tracker.build(chars, side, commit=commit)
    return ScreenReading(
        reading=reading,
        geometry=geom,
        flipped=flipped,
        chars=chars,
        confidence=min(confidences) if confidences else 0.0,
        uncertain=uncertain,
    )


# ── diagnostic en ligne de commande ─────────────────────────────────
def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic de la vision ChessMate : localise le plateau "
                    "à l'écran et affiche le FEN lu."
    )
    parser.add_argument("--monitor", type=int, default=None,
                        help="numéro d'écran (1 = principal)")
    parser.add_argument("--side", choices=["w", "b"], default="w",
                        help="trait à jouer (défaut : w)")
    parser.add_argument("--flipped", action="store_true",
                        help="le plateau est affiché côté Noirs")
    parser.add_argument("--dump", action="store_true",
                        help="écrit les 64 imagettes dans data/")
    parser.add_argument("--recalibrate", action="store_true",
                        help="ignore la calibration enregistrée")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    locator = BoardLocator(args.monitor)
    try:
        geom = locator.recalibrate() if args.recalibrate else locator.get()
    except VisionError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"Plateau     : left={geom.left} top={geom.top} size={geom.size} "
          f"(case ≈ {geom.square:.1f} px, écran {geom.monitor})")

    if args.dump:
        dump_squares(slice_squares(grab_screen(geom.monitor), geom))
        print(f"Imagettes   : {DATA_DIR}")

    try:
        result = read_board(locator, FenTracker(), args.side, flipped=args.flipped)
    except Exception as exc:                                   # noqa: BLE001
        print(f"❌ Lecture impossible : {exc}")
        return 1

    print(f"Orientation : {'Noirs en bas' if result.flipped else 'Blancs en bas'}")
    print(f"Confiance   : {result.confidence:.1%} (min sur 64 cases)")
    print(f"FEN         : {result.fen}")
    if result.reading.warnings:
        print(f"Avertis.    : {', '.join(result.reading.warnings)}")
    print()
    print(result.reading.board.unicode(borders=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
