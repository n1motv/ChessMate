from __future__ import annotations
import os, pathlib, shutil, time
from typing import List, Tuple, Optional, Dict, Any
import chess
import chess.engine

# ──────────────────────────────────────────────────────────────
#  Localisation du binaire Stockfish
# ──────────────────────────────────────────────────────────────
_EXE = (
    os.getenv("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or str(pathlib.Path("engines/stockfish-windows-x86-64-bmi2.exe").resolve())
)
if not pathlib.Path(_EXE).exists():
    raise FileNotFoundError(
        f"❌ Stockfish introuvable : {_EXE}\n"
        "Placez le binaire haute-perf. dans ./engines/ "
        "ou renseignez la variable d’environnement STOCKFISH_PATH."
    )
print("Chemin Stockfish :", _EXE)


class BestMoveEngine:
    """
    Wrapper Stockfish orienté perf + réactivité.

    Points clés :
    - Profils prédéfinis (bullet/blitz/rapid/analysis)
    - Limite combinée (time/depth/nodes) + early-exit (streaming)
    - MultiPV uniquement quand demandé
    - Threads/Hash/Syzygy auto, options UCI posées en 'safe set'
    - Rétro-compat: si `analysis_time` None et `depth` fourni => time ≈ depth/8
    """

    # Seuils pour l'early-exit (peuvent être ajustés)
    MATE_EARLY = 5              # stop si mate in <= 5 demi-coups
    MIN_DEPTH_FOR_CP_EXIT = 12  # ne stoppe pas trop tôt
    CP_GOOD_ENOUGH = 80         # ~0.80 pion d'avantage

    def __init__(
        self,
        *,
        # réglages primaires
        profile: str = "blitz",             # "bullet" | "blitz" | "rapid" | "analysis"
        analysis_time: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        # ressources
        threads: Optional[int] = None,
        hash_mb: Optional[int] = None,
        syzygy_path: Optional[str] = "syzygy",
    ):
        # rétro-compat depth -> time
        if analysis_time is None and depth is not None:
            analysis_time = max(0.5, depth / 8)

        self.profile = profile.lower()
        self.analysis_time = analysis_time
        self.depth = depth
        self.nodes = nodes
        self.threads = threads or (os.cpu_count() or 1)
        self.hash_mb = hash_mb
        self.syzygy_path = syzygy_path

        # cache léger (FEN -> (ts, best_san, score_cp, depth))
        self._cache: Dict[tuple, tuple] = {}
        self._engine: Optional[chess.engine.SimpleEngine] = None

        self._start()
        self._apply_profile_defaults()

    # ──────────────────────────────────────────────────────────
    # Boot + options
    # ──────────────────────────────────────────────────────────
    def _safe_set(self, name: str, value: Any) -> None:
        try:
            if name in self._engine.options:  # type: ignore[union-attr]
                self._engine.configure({name: str(value)})  # type: ignore[union-attr]
        except Exception:
            pass

    def _start(self) -> None:
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(_EXE)
            # Options robustes + rapides (sans planter si absentes)
            self._safe_set("Threads", self.threads)
            self._safe_set("Hash", self.hash_mb or 1024)
            self._safe_set("Ponder", "false")
            self._safe_set("Skill Level", 20)
            self._safe_set("UCI_LimitStrength", "false")
            self._safe_set("Move Overhead", 30)
            self._safe_set("Minimum Thinking Time", 10)
            self._safe_set("Slow Mover", 80)      # <100 = joue plus vite
            self._safe_set("Contempt", 0)
            self._safe_set("Use NNUE", "true")    # si dispo

            # Syzygy auto si présent
            if self.syzygy_path:
                tb = pathlib.Path(self.syzygy_path).resolve()
                if tb.exists():
                    self._safe_set("SyzygyPath", str(tb))
        except Exception as e:
            print("⚠️  Échec démarrage Stockfish :", e)
            self._engine = None

    def _apply_profile_defaults(self) -> None:
        """Remplit les valeurs manquantes selon le profil pour vitesse/force."""
        if self.profile not in {"bullet", "blitz", "rapid", "analysis"}:
            self.profile = "blitz"

        if self.profile == "bullet":
            self.analysis_time = self.analysis_time or 0.25
            self.nodes        = self.nodes or 300_000
            self.hash_mb      = self.hash_mb or 512
            self._safe_set("Slow Mover", 60)
        elif self.profile == "blitz":
            self.analysis_time = self.analysis_time or 0.60
            self.nodes        = self.nodes or 800_000
            self.hash_mb      = self.hash_mb or 1024
            self._safe_set("Slow Mover", 75)
        elif self.profile == "rapid":
            self.analysis_time = self.analysis_time or 2.0
            self.nodes        = self.nodes or 2_500_000
            self.hash_mb      = self.hash_mb or 1536
            self._safe_set("Slow Mover", 90)
        else:  # analysis
            self.analysis_time = self.analysis_time or 8.0
            self.nodes        = self.nodes or None  # illimité
            self.hash_mb      = self.hash_mb or 4096
            self._safe_set("Slow Mover", 100)

        # reflète Hash si on l’a fixé ici
        self._safe_set("Hash", self.hash_mb)

    def _ensure_alive(self) -> None:
        if self._engine is None:
            self._start()
            return
        try:
            self._engine.ping()
        except chess.engine.EngineTerminatedError:
            self._restart()
        except Exception:
            self._restart()

    def _restart(self) -> None:
        try:
            if self._engine:
                self._engine.quit()
        except Exception:
            pass
        self._start()

    # ──────────────────────────────────────────────────────────
    # Outils
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _orient_board(fen: str, color: str) -> chess.Board:
        bd = chess.Board(fen)
        if color == "black" and bd.turn:
            bd.push(chess.Move.null())
        return bd

    def _limit(
        self,
        time_s: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
    ) -> chess.engine.Limit:
        return chess.engine.Limit(
            time = time_s  if time_s  is not None else self.analysis_time,
            depth= depth   if depth   is not None else self.depth,
            nodes= nodes   if nodes   is not None else self.nodes,
        )

    # ──────────────────────────────────────────────────────────
    # API
    # ──────────────────────────────────────────────────────────
    def best_move(
        self,
        fen: str,
        color: str,
        *,
        time_s: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        cache_ttl: float = 0.4,           # pour spam UI : réutilise résultat récent
    ) -> str:
        self._ensure_alive()
        bd = self._orient_board(fen, color)
        key = (bd.board_fen(), bd.turn)

        # cache ultra-court
        now = time.time()
        if key in self._cache and (now - self._cache[key][0]) < cache_ttl:
            return self._cache[key][1]

        limit = self._limit(time_s, depth, nodes)

        last_info = None
        best_pv = None

        # Streaming = possibilité d'early-exit (gros gain de vitesse sur positions simples)
        with self._engine.analysis(bd, limit=limit, multipv=1) as analysis:  # type: ignore[arg-type]
            for info in analysis:
                last_info = info
                if "pv" in info and info["pv"]:
                    best_pv = info["pv"]

                sc = info.get("score")
                dp = int(info.get("depth", 0) or 0)

                if sc is not None:
                    # mate direct → stop
                    try:
                        mate = sc.relative.mate()
                        if mate is not None and abs(mate) <= self.MATE_EARLY:
                            analysis.stop()
                            break
                    except Exception:
                        pass

                    # avantage suffisant à profondeur décente → stop
                    try:
                        cp = sc.relative.score(mate_score=10000)
                        if (
                            cp is not None
                            and dp >= self.MIN_DEPTH_FOR_CP_EXIT
                            and abs(cp) >= self.CP_GOOD_ENOUGH
                        ):
                            analysis.stop()
                            break
                    except Exception:
                        pass

        if best_pv:
            best = bd.san(best_pv[0])
        else:
            # fallback : une passe d’analyse bloquante si jamais
            info = self._engine.analyse(bd, limit)  # type: ignore[arg-type]
            best = bd.san(info["pv"][0])

        # alimente le cache
        self._cache[key] = (time.time(), best, 0, int(last_info.get("depth", 0)) if last_info else 0)  # type: ignore[union-attr]
        return best

    def top_moves(
        self,
        fen: str,
        color: str,
        n: int = 3,
        *,
        time_s: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
    ) -> List[Tuple[str, float, str]]:
        self._ensure_alive()
        bd = self._orient_board(fen, color)
        limit = self._limit(time_s, depth, nodes)

        # MultiPV uniquement ici (sinon vitesse max pour best_move)
        infos = self._engine.analyse(bd, limit, multipv=max(1, n))  # type: ignore[arg-type]
        out: List[Tuple[str, float, str]] = []
        for inf in infos:
            san = bd.san(inf["pv"][0])
            score_cp = (inf["score"].relative.score(mate_score=10000) or 0) / 100.0
            pv5 = " ".join(bd.san(m) for m in inf["pv"][:5])
            out.append((san, score_cp, pv5))
        return out

    # ──────────────────────────────────────────────────────────
    # Fin propre
    # ──────────────────────────────────────────────────────────
    def quit(self) -> None:
        try:
            if self._engine:
                self._engine.quit()
                print("Stockfish arrêté 🛑")
        except Exception:
            pass