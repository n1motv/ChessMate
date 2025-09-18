"""
worker.py – analyse « pure » (aucune interaction UI).  
Renvoie : (fen, san, piece_name, src, dst, score, explication)
ou None si pas de coup à jouer ou en cas d’erreur/transitoire.
"""
from __future__ import annotations
from chess.engine import EngineTerminatedError
import concurrent.futures, chess

from vision   import screenshot_to_fen
from llm      import choose_move_and_explain
from utils    import describe_move, ensure_san
from engine   import BestMoveEngine

_pool        = concurrent.futures.ThreadPoolExecutor(1)
_sf          : BestMoveEngine | None = None
_prev_layout : str | None = None
_first_run   = True
_START_LAYOUT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def analyse(side: str):
    """
    side in {"white","black"}  → dict complet ou None
    Les exceptions fatales (moteur mort) déclenchent un reset global.
    """
    global _sf, _prev_layout, _first_run

    if _sf is None:
        _sf = BestMoveEngine(depth=28)      # profondeur élevée

    # ---------------- capture & debouncing -----------------
    raw     = screenshot_to_fen()
    layout  = raw.split()[0]

    if _first_run:
        _first_run = False
        if side == "black" and layout == _START_LAYOUT:
            _prev_layout = layout
            return None
    elif layout == _prev_layout:
        return None

    # ---------------- orientation du FEN -------------------
    parts = raw.split()
    parts[1] = 'w' if side == "white" else 'b'
    fen = " ".join(parts)

    # ---------------- choix de coup ------------------------
    try:
        san, expl, score = choose_move_and_explain(fen, side)
        san = ensure_san(fen, san)
    except Exception:
        # fallback : Stockfish costaud
        try:
            san = _sf.best_move(fen, side)
        except EngineTerminatedError:
            reset()                # on purge tout
            return None
        expl  = ""
        score = 0.0
        try:
            san = ensure_san(fen, san)
        except Exception:
            _prev_layout = layout
            return None

    # ---------------- légalité -----------------------------
    board_tmp = chess.Board(fen)
    try:
        mv_obj = board_tmp.parse_san(san)
    except Exception:
        _prev_layout = layout
        return None
    if mv_obj not in board_tmp.legal_moves:
        _prev_layout = layout
        return None

    piece, src, dst = describe_move(fen, san)
    # NB : _prev_layout sera mis à jour par main.py après exécution / highlight
    return fen, san, piece, src, dst, score, expl


# ───────────────────────────────────────────────────────────────
def reset():
    """Remise à zéro globale (appelée par UI ou en cas de crash Stockfish)."""
    global _sf, _prev_layout, _first_run
    _prev_layout = None
    _first_run   = True
    if _sf:
        try:
            _sf.quit()
        except Exception:
            pass
    _sf = None


def submit(side: str, lang: str = "fr"):
    """Interface asynchrone – retourne un Future."""
    return _pool.submit(analyse, side)

def shutdown() -> None:
    """Stoppe le thread-pool et ferme Stockfish (appelé par main.py)."""
    global _pool, _sf
    try:
        _pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    try:
        if _sf:
            _sf.quit()                 # méthode définie dans engine.py
    except Exception:
        pass
    _sf   = None
    _pool = None