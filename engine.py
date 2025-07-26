# engine.py ─────────────────────────────────────────────────────────────
"""
Interface légère entre python‑chess et Stockfish.
• BestMoveEngine.best_move()              → coup recommandé (SAN)
• BestMoveEngine.top_moves(n)             → n premiers coups + évals + PV
"""

import chess, os
import chess.engine
import pathlib
import shutil
from typing import List, Tuple

# ───────────────────────────────────────────────────────────────────────
# 1) Localisation de l’exécutable Stockfish
#
#   placer « stockfish‑windows‑x86‑64.exe » dans le dossier “engines/”
# ----------------------------------------------------------------------

_EXE = (
    os.getenv("STOCKFISH_PATH") or
    shutil.which("stockfish")   or
    str(pathlib.Path("engines/stockfish-windows-x86-64.exe").resolve())
)

if not pathlib.Path(_EXE).exists():
    raise FileNotFoundError(
        "❌ Stockfish introuvable ; placez-le dans ./engines/ ou "
        "définissez la variable d’environnement STOCKFISH_PATH."
    )

print("Chemin Stockfish :", _EXE)

# ───────────────────────────────────────────────────────────────────────
class BestMoveEngine:
    """Encapsulation d’un moteur UCI (Stockfish)."""

    def __init__(self, depth: int = 18):
        """
        depth : profondeur par défaut pour analyse (18 ≈ 0.5‑1 s).
        """
        self.depth = depth
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(_EXE)
        except Exception as e:
            print("⚠️  Échec lancement Stockfish :", e)
            self._engine = None

    # ──────────────────────────────────────────────────────────────────
    def _prepare_board(self, fen: str, color: str) -> chess.Board:
        """
        Crée un objet Board prêt pour l’analyse (ajuste le trait si
        l’utilisateur joue Noir).
        """
        board = chess.Board(fen)
        if color == "black" and board.turn:          # Stockfish doit jouer Noir
            board.push(chess.Move.null())            # coup Null = passer son tour
        return board

    # ──────────────────────────────────────────────────────────────────
    def best_move(self, fen: str, color: str) -> str:
        """Renvoie le **meilleur** coup (SAN) selon Stockfish."""
        if self._engine is None:
            raise RuntimeError("Moteur non initialisé.")

        board = self._prepare_board(fen, color)

        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth)
        )
        move = info["pv"][0]                         # premier coup de la ligne
        return board.san(move)                       # en notation SAN

    # ──────────────────────────────────────────────────────────────────
    def top_moves(
        self,
        fen: str,
        color: str,
        n: int = 3,
        depth: int | None = None
    ) -> List[Tuple[str, float, str]]:
        """
        Renvoie la liste des n meilleurs coups :
        [(coup_SAN, évaluation_cent_pions, PV_san_5_coups), …]

        • évaluation > 0  → avantage au trait.
        • depth : si None → utilise self.depth.
        """
        if self._engine is None:
            raise RuntimeError("Moteur non initialisé.")

        depth = depth or self.depth
        board = self._prepare_board(fen, color)

        infos = self._engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=n
        )

        top = []
        for info in infos:
            mv   = board.san(info["pv"][0])
            cp   = info["score"].relative.score(mate_score=10000) or 0
            cp   = cp / 100.0                    # cent‑pions → pions
            pv5  = " ".join(board.san(m) for m in info["pv"][:5])
            top.append((mv, cp, pv5))

        return top

    # ──────────────────────────────────────────────────────────────────
    def quit(self):
        """Ferme proprement le moteur (à appeler à la fin)."""
        try:
            if self._engine:
                self._engine.quit()
                print("Stockfish arrêté proprement 🛑")
        except chess.engine.EngineTerminatedError:
            print("Stockfish était déjà terminé.")
        except Exception as e:
            print("Erreur à l’arrêt de Stockfish :", e)
