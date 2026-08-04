"""
autoplay.py – exécution d'un coup à la souris.

Corrections apportées :

* **la promotion est gérée.**  L'ancien code faisait deux clics secs : sur un
  coup de promotion, le sélecteur de pièce s'ouvrait et n'était jamais cliqué,
  donnant au mieux une dame par défaut, au pire un coup annulé.
* **drag & drop disponible** : certains sites n'acceptent pas le clic-clic.
* **le petit roque est joué roi→case d'arrivée**, avec repli roi→tour pour les
  interfaces qui exigent cette convention.
* **failsafe pyautogui activé** : coin haut-gauche de l'écran = arrêt d'urgence.
* **rien n'est supposé.**  La fonction renvoie ce qu'elle a fait ; c'est
  l'appelant qui recapture l'écran pour confirmer.  L'ancien `main.py` mettait
  à jour l'état interne comme si le clic avait forcément réussi, ce qui
  désynchronisait silencieusement l'application dès qu'il échouait.
"""
from __future__ import annotations

import logging
import time

import chess
import pyautogui

import config
from vision import BoardGeometry

log = logging.getLogger(__name__)

# Coin haut-gauche = interruption d'urgence. Indispensable pour une appli qui
# prend le contrôle de la souris.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

#: décalage vertical du sélecteur de promotion, en nombre de cases, depuis la
#: case d'arrivée et vers l'intérieur du plateau (convention chess.com/lichess)
PROMOTION_STEPS = {"q": 0, "n": 1, "r": 2, "b": 3}

_PROMOTION_CHAR = {
    chess.QUEEN: "q", chess.KNIGHT: "n",
    chess.ROOK: "r", chess.BISHOP: "b",
}


def _click(x: int, y: int, *, duration: float = 0.06) -> None:
    pyautogui.moveTo(x, y, duration=duration)
    pyautogui.click()


def _drag(x1: int, y1: int, x2: int, y2: int, *, duration: float = 0.12) -> None:
    pyautogui.moveTo(x1, y1, duration=duration / 2)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2, y2, duration=duration)
    pyautogui.mouseUp()


def _promotion_target(geom: BoardGeometry, dst: str, flipped: bool,
                      promo_char: str) -> tuple[int, int]:
    """
    Position du bouton de promotion.  Le sélecteur s'affiche sur la colonne
    d'arrivée, en partant de la case de promotion et en descendant vers
    l'intérieur du plateau.
    """
    order = list(config.get("promotion_order", ["q", "n", "r", "b"]))
    steps = order.index(promo_char) if promo_char in order else PROMOTION_STEPS[promo_char]

    x, y = geom.square_to_xy(dst, flipped)
    rank = int(dst[1])
    # promotion sur la 8e rangée → la liste descend ; sur la 1re → elle monte.
    going_down = (rank == 8) != flipped
    y += int(steps * geom.square * (1 if going_down else -1))
    return x, y


def play_move(move: chess.Move,
              geom: BoardGeometry,
              *,
              flipped: bool = False,
              board: chess.Board | None = None,
              style: str | None = None) -> None:
    """
    Joue `move` à la souris sur le plateau situé par `geom`.

    `board` (position *avant* le coup) sert uniquement à détecter le roque ;
    il est facultatif mais recommandé.
    """
    style = style or config.get("autoplay_style", "click")
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)

    # ── roque : on vise la case d'arrivée du roi ────────────────────
    if board is not None and board.is_castling(move):
        _play_castling(move, geom, flipped, board, style)
        return

    x1, y1 = geom.square_to_xy(src, flipped)
    x2, y2 = geom.square_to_xy(dst, flipped)

    if style == "drag":
        _drag(x1, y1, x2, y2)
    else:
        _click(x1, y1)
        _click(x2, y2)

    # ── promotion : cliquer la pièce dans le sélecteur ──────────────
    if move.promotion:
        promo_char = _PROMOTION_CHAR.get(move.promotion, "q")
        time.sleep(0.18)                       # laisser le sélecteur s'ouvrir
        px, py = _promotion_target(geom, dst, flipped, promo_char)
        log.debug("Promotion %s → clic (%d, %d)", promo_char, px, py)
        _click(px, py)


def _play_castling(move: chess.Move, geom: BoardGeometry, flipped: bool,
                   board: chess.Board, style: str) -> None:
    """
    Roque : chess.com et lichess attendent roi → case d'arrivée (e1 → g1).
    python-chess encode déjà le coup ainsi en échecs classiques, il suffit
    donc de ne pas le confondre avec un déplacement de tour.
    """
    del board                       # signature homogène avec play_move()
    king_from = chess.square_name(move.from_square)
    king_to = chess.square_name(move.to_square)

    x1, y1 = geom.square_to_xy(king_from, flipped)
    x2, y2 = geom.square_to_xy(king_to, flipped)
    if style == "drag":
        _drag(x1, y1, x2, y2)
    else:
        _click(x1, y1)
        _click(x2, y2)
    log.debug("Roque joué %s → %s", king_from, king_to)


def highlight_move(src: str, dst: str, geom: BoardGeometry,
                   *, flipped: bool = False) -> None:
    """Déplace simplement le curseur, sans cliquer (mode démonstration)."""
    x1, y1 = geom.square_to_xy(src, flipped)
    x2, y2 = geom.square_to_xy(dst, flipped)
    pyautogui.moveTo(x1, y1, duration=0.08)
    pyautogui.moveTo(x2, y2, duration=0.08)


def move_cursor_to(x: int, y: int) -> None:
    """Ramène le curseur (typiquement sur l'interface) — best effort."""
    try:
        pyautogui.moveTo(x, y, duration=0.10)
    except Exception as exc:                                    # noqa: BLE001
        log.debug("Déplacement du curseur impossible : %s", exc)
