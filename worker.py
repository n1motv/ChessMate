# worker.py  ───────────────────────────────────────────────────────────
import concurrent.futures
from vision import screenshot_to_fen
from llm    import choose_move_and_explain
from utils  import describe_move, ensure_san
from engine import BestMoveEngine

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_sf       = BestMoveEngine(depth=18)           # secours

def analyse(color: str, lang: str):
    raw_fen = screenshot_to_fen()
    parts   = raw_fen.split()
    parts[1]= 'w' if color == "white" else 'b'
    fen     = " ".join(parts)

    try:
        move, expl, score = choose_move_and_explain(fen, color, lang)
    except Exception:
        move  = _sf.best_move(fen, color)
        score = 0.00
        expl  = ""                # ← vide → pas d’explication affichée

    piece, src, dst = describe_move(fen, move)
    return fen, move, piece, src, dst, score, expl

def submit(color):
    return _executor.submit(analyse, color)
