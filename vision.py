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
import functools
import logging
import threading
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PIL import Image

import config
from classifier import get_classifier
from fen import BoardReading, FenError, FenTracker, flip_chars, index_to_name
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
def _resolve_monitor(monitors: list, monitor: int) -> int:
    """
    Ramène `monitor` à un index d'écran réellement branché.

    `config.json` mémorise le dernier écran utilisé, mais ce fichier suit le
    projet d'une machine à l'autre (et un écran peut être débranché entre
    deux lancements) : l'écran n°2 d'hier n'existe pas forcément aujourd'hui.
    Sans ce garde-fou, `sct.monitors[2]` levait un `IndexError` à chaque
    analyse au lieu de retomber sur l'écran principal.
    """
    if len(monitors) < 2:                     # index 0 = « tous les écrans »
        raise VisionError("Aucun écran détecté.")
    if 1 <= monitor < len(monitors):
        return monitor
    log.warning(
        "Écran %s indisponible (%s écran(s) détecté(s)) — repli sur l'écran 1.",
        monitor, len(monitors) - 1,
    )
    return 1


def grab_screen(monitor: int = 1) -> np.ndarray:
    """Capture l'écran `monitor` → image BGR."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[_resolve_monitor(sct.monitors, monitor)])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def monitor_bounds(monitor: int = 1) -> tuple[int, int, int, int]:
    """(left, top, width, height) de l'écran, en coordonnées globales."""
    with mss.mss() as sct:
        mon = sct.monitors[_resolve_monitor(sct.monitors, monitor)]
        return int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"])


def screen_size(monitor: int = 1) -> tuple[int, int]:
    return monitor_bounds(monitor)[2:]


@functools.lru_cache(maxsize=8)
def monitor_origin(monitor: int = 1) -> tuple[int, int]:
    """Coin haut-gauche de l'écran dans l'espace de coordonnées global."""
    return monitor_bounds(monitor)[:2]


def available_monitors() -> list[int]:
    """Indices mss des écrans physiques (hors index 0 = « tous les écrans »)."""
    with mss.mss() as sct:
        return list(range(1, len(sct.monitors)))


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


def _best_candidate(img: np.ndarray, monitor: int) -> tuple[float, BoardGeometry] | None:
    """
    Cherche le meilleur candidat de damier dans `img` (capture d'un seul
    écran). Renvoie `None` si rien de plausible n'y a été trouvé — ce n'est
    pas une erreur en soi quand on balaie plusieurs écrans à la recherche du
    bon.

    Les coordonnées du `BoardGeometry` renvoyé sont **globales** (coin
    haut-gauche de l'écran `monitor` ajouté aux coordonnées locales du
    contour), pour que la géométrie soit directement utilisable par
    `autoplay.py` / `overlay.py` (clics souris, dessin de la flèche) sans
    connaître par ailleurs la disposition des écrans.
    """
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_side = max(120, min(w_img, h_img) // 12)
    max_side = min(w_img, h_img)
    ox, oy = monitor_origin(monitor)

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
            best = (weighted, BoardGeometry(ox + x, oy + y, size, monitor))

    return best


def detect_board_region(monitor: int | None = 1,
                        image: np.ndarray | None = None) -> BoardGeometry:
    """
    Localise l'échiquier à l'écran.

    `monitor=None` balaie **tous** les écrans physiques connectés et retient
    le meilleur candidat toutes écrans confondus — indispensable dès que
    l'échiquier n'est pas forcément sur le même écran que la fenêtre
    ChessMate. `monitor=<n>` restreint la recherche à cet écran (utilisé par
    `--dump` / les tests, ou une fois l'écran déjà connu).

    Contrairement à l'ancien `max(cnts, key=cv2.contourArea)`, on filtre les
    candidats sur leur carré-itude et leur taille, puis on départage au score
    de damier — ce qui évite de sélectionner une fenêtre de navigateur ou un
    fond d'écran plus grand que le plateau.
    """
    monitor_origin.cache_clear()

    if image is not None:
        best = _best_candidate(image, monitor if monitor is not None else 1)
    else:
        monitors = [monitor] if monitor is not None else available_monitors()
        if not monitors:
            raise VisionError("Aucun écran détecté.")
        best = None
        for m in monitors:
            cand = _best_candidate(grab_screen(m), m)
            if cand and (best is None or cand[0] > best[0]):
                best = cand

    if best is None:
        raise VisionError(
            "Plateau introuvable. Vérifiez qu'un échiquier est visible à "
            "l'écran, ou calibrez-le manuellement (bouton « Calibrer »)."
        )

    log.info("Plateau détecté : %s (score %.3f)", best[1], best[0])
    return best[1]


def _fits_monitor(geom: BoardGeometry) -> bool:
    """Le plateau tient-il entièrement dans les limites de son écran ?"""
    left, top, width, height = monitor_bounds(geom.monitor)
    return (
        geom.size > 0
        and geom.left >= left and geom.top >= top
        and geom.left + geom.size <= left + width
        and geom.top + geom.size <= top + height
    )


# ── localisateur avec cache + persistance ───────────────────────────
class BoardLocator:
    """
    Conserve la géométrie du plateau entre deux captures et sait la
    ré-évaluer à la demande.  C'est ce qui remplace les anciennes constantes
    globales LEFT / TOP / SQUARE, figées au démarrage du programme.
    """

    def __init__(self, monitor: int | None = None, *, auto_detect: bool = True) -> None:
        # un écran explicitement demandé (CLI `--monitor`) reste figé ; sinon
        # on part du dernier écran connu (config.json) mais on est prêt à
        # rebalayer tous les écrans si la position mémorisée ne suffit plus.
        # Dans les deux cas l'index est validé : celui de config.json peut
        # venir d'une autre machine, celui de la CLI d'une faute de frappe.
        self._pinned = monitor is not None
        requested = monitor if monitor is not None else config.get("monitor", 1)
        try:
            with mss.mss() as sct:
                self.monitor = _resolve_monitor(sct.monitors, requested)
        except (VisionError, mss.exception.ScreenShotError):
            # aucun écran exploitable au démarrage : on n'empêche pas la
            # fenêtre de s'ouvrir, `get()` signalera l'erreur au moment utile.
            self.monitor = 1
        # l'interface graphique désactive ce balayage : l'utilisateur choisit
        # toujours l'échiquier à la souris (voir calibration.py), la
        # détection par contours reste disponible pour la CLI/les tests.
        self._auto_detect = auto_detect
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
                    geom = BoardGeometry(*saved, monitor=self.monitor)
                    # les rectangles sont indexés par résolution seulement :
                    # celui d'un écran 1920×1080 d'une autre machine (ou d'un
                    # second écran depuis débranché) tombe hors du bureau
                    # actuel. On le jette plutôt que de capturer dans le vide.
                    if _fits_monitor(geom):
                        self._geom = geom
                        log.info("Plateau repris de config.json : %s", geom)
                        return geom
                    log.warning(
                        "Calibration mémorisée hors de l'écran %s (%s) — ignorée.",
                        self.monitor, geom,
                    )
                    config.clear_board_rect(*screen_size(self.monitor))

            if not self._auto_detect:
                raise VisionError(
                    "Plateau non calibré. Cliquez sur « Calibrer » (ou "
                    "appuyez sur R) et entourez l'échiquier à la souris."
                )

            # pas de calibration mémorisée (ou rafraîchissement demandé) :
            # on balaie tous les écrans, sauf si l'appelant a imposé le sien.
            self._geom = detect_board_region(self.monitor if self._pinned else None)
            self.monitor = self._geom.monitor
            config.set("monitor", self.monitor)
            config.set_board_rect(*screen_size(self.monitor), *self._geom.as_tuple())
            return self._geom

    def invalidate(self) -> None:
        """Oublie la géométrie : la prochaine lecture relancera la détection."""
        with self._lock:
            self._geom = None

    def recalibrate(self) -> BoardGeometry:
        """Force une nouvelle détection et l'enregistre."""
        return self.get(refresh=True)

    def set_manual(self, left: int, top: int, size: int,
                   monitor: int | None = None) -> BoardGeometry:
        """Calibration manuelle (rectangle tracé par l'utilisateur)."""
        with self._lock:
            if monitor is not None and monitor != self.monitor:
                self.monitor = monitor
                config.set("monitor", monitor)
            self._geom = BoardGeometry(left, top, size, self.monitor)
            config.set_board_rect(*screen_size(self.monitor), left, top, size)
            return self._geom


def board_region(img: np.ndarray, geom: BoardGeometry) -> np.ndarray:
    """
    Sous-tableau de `img` (capture d'écran brute) correspondant au plateau —
    mêmes coordonnées que `slice_squares`, mais sans le découpage en 64
    cases. Sert notamment à détecter si l'écran a changé sans repasser par
    le classifieur (voir `worker.Analyzer.read`).
    """
    ox, oy = monitor_origin(geom.monitor)
    left, top = geom.left - ox, geom.top - oy
    size = int(round(geom.size))
    h, w = img.shape[:2]
    y0, y1 = max(0, top), min(h, top + size)
    x0, x1 = max(0, left), min(w, left + size)
    return img[y0:y1, x0:x1]


# ── découpe en 64 cases ─────────────────────────────────────────────
def slice_squares(img: np.ndarray, geom: BoardGeometry,
                  *, margin: float = 0.0) -> list[Image.Image]:
    """
    Découpe le plateau en 64 imagettes PIL, de a8 (index 0) à h1 (index 63)
    **telles qu'affichées à l'écran** (l'orientation est traitée plus loin).

    `img` provient de `grab_screen(geom.monitor)` : ses coordonnées sont donc
    relatives à cet écran, alors que `geom.left`/`geom.top` sont désormais
    globales (bureau entier) — d'où la soustraction de l'origine de l'écran.
    """
    ox, oy = monitor_origin(geom.monitor)
    local_left = geom.left - ox
    local_top = geom.top - oy

    step = geom.square
    pad = step * margin
    squares: list[Image.Image] = []
    board_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = board_rgb.shape[:2]

    for r in range(8):
        for c in range(8):
            x0 = int(round(local_left + c * step + pad))
            y0 = int(round(local_top + r * step + pad))
            x1 = int(round(local_left + (c + 1) * step - pad))
            y1 = int(round(local_top + (r + 1) * step - pad))
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


_PROMOTION_RANKS = list(range(0, 8)) + list(range(56, 64))  # rangée 8 et rangée 1


def _repair_illegal_squares(
    chars: list[str | None],
    topk: list[list[tuple[str | None, float]]],
) -> tuple[list[str | None], bool]:
    """
    Corrige les erreurs de classification les plus fréquentes en repêchant
    la 2ᵉ meilleure hypothèse du classifieur sur les cases responsables
    d'une position structurellement impossible, plutôt que d'abandonner
    dès la première lecture erronée :

    * **plusieurs rois d'une même couleur** — on ne garde que la case la
      plus confiante, les autres reprennent leur 2ᵉ hypothèse (un roi
      n'apparaît jamais deux fois sur un plateau réel, c'est forcément une
      confusion avec une autre pièce de même couleur/forme, ex. une dame) ;
    * **aucun roi d'une couleur** — le roi réel a probablement été classé
      juste derrière une autre pièce sur une case : on promeut la case où
      cette hypothèse est la plus probable ;
    * **pion sur la rangée de promotion** — physiquement impossible aux
      échecs (un pion qui atteint le dernier rang est aussitôt promu), donc
      c'est forcément une confusion avec une autre pièce.

    Renvoie la grille corrigée et un booléen indiquant si un correctif a
    effectivement été appliqué (inutile de retenter la construction du FEN
    sinon : ce serait exactement la même erreur).
    """
    chars = list(chars)
    changed = False

    for king, other_king in (("K", "k"), ("k", "K")):
        idxs = [i for i, c in enumerate(chars) if c == king]
        if len(idxs) > 1:
            idxs.sort(key=lambda i: topk[i][0][1], reverse=True)
            for i in idxs[1:]:
                alt = next((lbl for lbl, _ in topk[i][1:] if lbl != king), None)
                chars[i] = alt
                changed = True
        elif not idxs:
            candidates = [
                (i, conf) for i, entries in enumerate(topk)
                for lbl, conf in entries[1:]
                if lbl == king and chars[i] != other_king
            ]
            if candidates:
                i, _ = max(candidates, key=lambda item: item[1])
                chars[i] = king
                changed = True

    for i in _PROMOTION_RANKS:
        if chars[i] in ("P", "p"):
            alt = next((lbl for lbl, _ in topk[i][1:] if lbl not in ("P", "p")), None)
            chars[i] = alt
            changed = True

    return chars, changed


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

    Une lecture peut rater ponctuellement (case survolée par le curseur,
    animation d'un coup en cours, surlignage qui change le contraste d'une
    case au moment précis de la capture...).  Plutôt que de remonter une
    erreur au premier accroc, on tente de corriger les cases fautives avec
    la 2ᵉ hypothèse du classifieur (`_repair_illegal_squares`) et, si ça ne
    suffit pas et qu'on est en capture live (pas une image de test fournie
    explicitement), on retente une seconde capture avant d'abandonner.
    """
    geom = locator.get()
    live_capture = image is None
    threshold = float(config.get("min_square_confidence", 0.75))
    max_uncertain = int(config.get("max_uncertain_squares", 3))

    attempts = 2 if live_capture else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        img = grab_screen(geom.monitor) if image is None else image

        squares = slice_squares(img, geom)
        if config.get("dump_squares"):
            dump_squares(squares)

        topk = get_classifier().predict_board_topk(squares, k=2)
        chars = [entries[0][0] for entries in topk]
        confidences = [entries[0][1] for entries in topk]

        if flipped is None:
            flipped = (side == "b")
        if flipped:
            chars = flip_chars(chars)
            confidences = list(reversed(confidences))
            topk = list(reversed(topk))

        uncertain = [index_to_name(i) for i, c in enumerate(confidences) if c < threshold]
        if len(uncertain) > max_uncertain:
            last_error = VisionError(
                f"Position incertaine : {len(uncertain)} case(s) sous le seuil de "
                f"confiance ({', '.join(uncertain[:8])}"
                f"{'…' if len(uncertain) > 8 else ''}). "
                "Recalibrez le plateau ou réentraînez le modèle sur votre thème."
            )
            continue

        try:
            reading = tracker.build(chars, side, commit=commit)
        except FenError as exc:
            last_error = exc
            repaired, changed = _repair_illegal_squares(chars, topk)
            if changed:
                try:
                    reading = tracker.build(repaired, side, commit=commit)
                    chars = repaired
                except FenError as exc2:
                    last_error = exc2
                    continue
            else:
                continue

        return ScreenReading(
            reading=reading,
            geometry=geom,
            flipped=flipped,
            chars=chars,
            confidence=min(confidences) if confidences else 0.0,
            uncertain=uncertain,
        )

    assert last_error is not None
    raise last_error


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
