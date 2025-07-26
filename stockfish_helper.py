# stockfish_helper.py  ────────────────────────────────────────────────
import chess, chess.engine, pathlib
_STK = str(pathlib.Path("engines/stockfish-windows-x86-64.exe").resolve())
_engine = chess.engine.SimpleEngine.popen_uci(_STK)

def stockfish_top(fen: str, n=3, depth=20):
    """
    Renvoie une liste triée :
      [(san_move, score_cp, pv_san_5coups), …]   (longueur = n)
    score_cp  : >0 ⇒ avantage au trait.
    """
    board = chess.Board(fen)
    infos = _engine.analyse(board,
                            chess.engine.Limit(depth=depth),
                            multipv=n)

    top = []
    for info in infos:
        mv   = board.san(info["pv"][0])
        cp   = info["score"].pov(board.turn).score(mate_score=10_000)/100
        pv5  = " ".join(board.san(m) for m in info["pv"][:5])
        top.append((mv, cp, pv5))
    return top
