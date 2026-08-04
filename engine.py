"""
engine.py – pilotage de Stockfish.

Corrections apportées :

* **une seule instance** de moteur pour tout le processus (`get_engine()`).
  Avant, `llm.py` en créait une au niveau module et `worker.py` une autre :
  deux processus Stockfish, chacun avec Hash=1024 Mo et Threads=cpu_count(),
  soit ~2 Go de RAM et deux moteurs se disputant tous les cœurs.  Celui de
  `llm.py` n'était jamais fermé et survivait à l'application.
* **une seule limite primaire.**  Avant, `depth=28` était converti en
  `analysis_time=3.5 s` puis combiné au `nodes=800 000` du profil blitz : les
  trois limites partaient ensemble à `chess.engine.Limit` et la première
  atteinte (les nœuds) coupait tout.  La « profondeur élevée » annoncée
  n'existait pas.
* **early-exit sur la stabilité du meilleur coup**, plus sur la valeur de
  l'évaluation.  L'ancien critère (`|cp| >= 80` dès la profondeur 12) coupait
  l'analyse dans toute position à +1.0, c'est-à-dire en permanence.
* **plus de coup nul illégal.**  `_orient_board()` faisait
  `board.push(chess.Move.null())`, illégal en échec et destructeur pour
  l'en passant.
* **pas d'exception à l'import.**  Le binaire est résolu paresseusement : si
  Stockfish manque, l'interface s'ouvre quand même et affiche l'erreur.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
import pathlib
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import chess
import chess.engine

import config
from paths import ENGINES_DIR, SYZYGY_DIR

log = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """Stockfish est introuvable ou refuse de démarrer."""


# ── localisation du binaire ─────────────────────────────────────────
_BUNDLED_NAMES = (
    "stockfish-windows-x86-64-bmi2.exe",
    "stockfish-windows-x86-64-avx2.exe",
    "stockfish-windows-x86-64.exe",
    "stockfish.exe",
    "stockfish",
)


def find_stockfish() -> pathlib.Path:
    """
    Résolution paresseuse du binaire, par ordre de priorité :
    STOCKFISH_PATH → PATH système → binaire livré dans engines/.
    Lève `EngineUnavailable` avec un message actionnable si rien n'est trouvé.
    """
    override = os.getenv("STOCKFISH_PATH")
    if override:
        p = pathlib.Path(override).expanduser()
        if p.is_file():
            return p.resolve()
        raise EngineUnavailable(
            f"STOCKFISH_PATH pointe sur un fichier inexistant : {override}"
        )

    on_path = shutil.which("stockfish")
    if on_path:
        return pathlib.Path(on_path).resolve()

    for name in _BUNDLED_NAMES:
        candidate = ENGINES_DIR / name
        if candidate.is_file():
            return candidate.resolve()

    raise EngineUnavailable(
        "Stockfish introuvable.\n"
        f"Placez le binaire dans {ENGINES_DIR}, ajoutez-le au PATH, "
        "ou renseignez la variable d'environnement STOCKFISH_PATH."
    )


# ── profils ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Profile:
    """
    Un profil définit **une** limite primaire, jamais plusieurs.
    `depth_cap` est un garde-fou facultatif, pas un objectif.
    """
    name: str
    movetime: float          # secondes allouées à la recherche
    hash_mb: int
    depth_cap: Optional[int] = None
    slow_mover: int = 80


PROFILES: dict[str, Profile] = {
    "bullet":   Profile("bullet",   movetime=0.25, hash_mb=256,  depth_cap=20, slow_mover=60),
    "blitz":    Profile("blitz",    movetime=0.80, hash_mb=512,  depth_cap=26, slow_mover=75),
    "rapid":    Profile("rapid",    movetime=2.50, hash_mb=1024, depth_cap=32, slow_mover=90),
    "analysis": Profile("analysis", movetime=8.00, hash_mb=2048, depth_cap=None, slow_mover=100),
}
DEFAULT_PROFILE = "blitz"


@dataclass
class MoveSuggestion:
    """Un coup candidat avec son évaluation."""
    san: str
    move: chess.Move
    score_pawns: float           # positif = bon pour le camp au trait
    mate_in: Optional[int]
    pv_san: str                  # variante principale, 5 coups
    depth: int

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None


class BestMoveEngine:
    """
    Enveloppe Stockfish orientée réactivité.

    Politique de limite : `movetime` est le critère unique, éventuellement
    complété d'un plafond de profondeur.  On ne mélange plus temps, profondeur
    et nœuds dans le même `Limit`.
    """

    # Early-exit : on s'arrête quand le meilleur coup ne bouge plus.
    STABLE_ITERATIONS = 4        # itérations consécutives avec le même coup
    MIN_DEPTH_FOR_EXIT = 14      # ...pas avant cette profondeur
    MATE_EARLY = 3               # mat annoncé en ≤ 3 → inutile de chercher

    def __init__(self,
                 profile: str | None = None,
                 *,
                 movetime: float | None = None,
                 depth: int | None = None,
                 threads: int | None = None,
                 hash_mb: int | None = None,
                 syzygy_path: str | pathlib.Path | None = None) -> None:
        name = (profile or config.get("engine_profile", DEFAULT_PROFILE)).lower()
        self.profile = PROFILES.get(name, PROFILES[DEFAULT_PROFILE])
        if name not in PROFILES:
            log.warning("Profil moteur inconnu « %s » → %s", name, self.profile.name)

        self.movetime = movetime if movetime is not None else self.profile.movetime
        self.depth_cap = depth if depth is not None else self.profile.depth_cap
        self.threads = threads or config.get("engine_threads") or (os.cpu_count() or 1)
        self.hash_mb = hash_mb or config.get("engine_hash_mb") or self.profile.hash_mb
        self.syzygy_path = pathlib.Path(syzygy_path) if syzygy_path else SYZYGY_DIR

        self._engine: chess.engine.SimpleEngine | None = None
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, bool], tuple[float, MoveSuggestion]] = {}

    # ── cycle de vie ────────────────────────────────────────────────
    def _safe_set(self, name: str, value: Any) -> None:
        """Pose une option UCI sans planter si le moteur ne la connaît pas."""
        engine = self._engine
        if engine is None:
            return
        try:
            if name in engine.options:
                engine.configure({name: value})
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            log.debug("Option UCI « %s » refusée : %s", name, exc)

    def _start(self) -> None:
        exe = find_stockfish()
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(exe))
        except Exception as exc:                                  # noqa: BLE001
            raise EngineUnavailable(f"Démarrage de Stockfish impossible : {exc}") from exc

        log.info("Stockfish démarré : %s (%d threads, %d Mo de hash)",
                 exe.name, self.threads, self.hash_mb)

        self._safe_set("Threads", int(self.threads))
        self._safe_set("Hash", int(self.hash_mb))
        self._safe_set("Ponder", False)
        self._safe_set("Skill Level", 20)
        self._safe_set("UCI_LimitStrength", False)
        self._safe_set("Move Overhead", 30)
        self._safe_set("Slow Mover", self.profile.slow_mover)
        self._safe_set("Use NNUE", True)

        if self.syzygy_path and self.syzygy_path.is_dir():
            self._safe_set("SyzygyPath", str(self.syzygy_path.resolve()))

    def _ensure_alive(self) -> chess.engine.SimpleEngine:
        with self._lock:
            if self._engine is None:
                self._start()
            else:
                try:
                    self._engine.ping()
                except chess.engine.EngineTerminatedError:
                    log.warning("Stockfish s'est arrêté — redémarrage")
                    self._engine = None
                    self._cache.clear()
                    self._start()
            assert self._engine is not None
            return self._engine

    def quit(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
            self._cache.clear()
        if engine is None:
            return
        try:
            engine.quit()
            log.info("Stockfish arrêté")
        except Exception:                                          # noqa: BLE001
            with contextlib.suppress(Exception):
                engine.close()

    # ── position ────────────────────────────────────────────────────
    @staticmethod
    def _board_for(fen: str, color: str | None) -> chess.Board:
        """
        Construit le plateau et vérifie que le trait correspond bien au camp
        demandé.  Aucune manipulation du plateau (l'ancien `push(Move.null())`
        était illégal en échec et effaçait l'en passant).
        """
        board = chess.Board(fen)
        if color is not None:
            want_white = color.lower().startswith("w")
            if board.turn != want_white:
                raise ValueError(
                    f"Le FEN est au trait des {'Blancs' if board.turn else 'Noirs'} "
                    f"alors que « {color} » est demandé : {fen}"
                )
        return board

    def _limit(self, movetime: float | None, depth: int | None) -> chess.engine.Limit:
        """Un seul critère primaire (le temps) + un plafond de profondeur."""
        return chess.engine.Limit(
            time=movetime if movetime is not None else self.movetime,
            depth=depth if depth is not None else self.depth_cap,
        )

    @staticmethod
    def _to_suggestion(board: chess.Board, info: chess.engine.InfoDict) -> MoveSuggestion:
        pv = list(info.get("pv") or [])
        if not pv:
            raise chess.engine.EngineError("Le moteur n'a renvoyé aucune variante")
        score = info.get("score")
        mate_in = score.relative.mate() if score else None
        cp = score.relative.score(mate_score=10_000) if score else 0
        return MoveSuggestion(
            san=board.san(pv[0]),
            move=pv[0],
            score_pawns=(cp or 0) / 100.0,
            mate_in=mate_in,
            pv_san=board.variation_san(pv[:5]),
            depth=int(info.get("depth") or 0),
        )

    # ── API ─────────────────────────────────────────────────────────
    def analyse_best(self,
                     fen: str,
                     color: str | None = None,
                     *,
                     movetime: float | None = None,
                     depth: int | None = None,
                     cache_ttl: float = 0.4) -> MoveSuggestion:
        """
        Meilleur coup, avec arrêt anticipé dès que la recherche se stabilise.
        """
        board = self._board_for(fen, color)
        if board.is_game_over():
            raise chess.engine.EngineError(
                f"Partie terminée ({board.result()}) — aucun coup à jouer"
            )

        key = (board.fen(), board.turn)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[0]) < cache_ttl:
            return cached[1]

        engine = self._ensure_alive()
        limit = self._limit(movetime, depth)

        best: MoveSuggestion | None = None
        stable = 0
        last_move: chess.Move | None = None

        with self._lock, engine.analysis(board, limit=limit) as analysis:
            for info in analysis:
                if not info.get("pv"):
                    continue
                try:
                    current = self._to_suggestion(board, info)
                except (chess.engine.EngineError, ValueError):
                    continue
                best = current

                # mat court : rien de mieux à trouver
                if current.mate_in is not None and abs(current.mate_in) <= self.MATE_EARLY:
                    analysis.stop()
                    break

                # stabilité : le meilleur coup ne change plus
                if current.move == last_move:
                    stable += 1
                else:
                    stable, last_move = 0, current.move

                if (stable >= self.STABLE_ITERATIONS
                        and current.depth >= self.MIN_DEPTH_FOR_EXIT):
                    analysis.stop()
                    break

        if best is None:                       # flux vide : passe bloquante
            info = engine.analyse(board, limit)
            best = self._to_suggestion(board, info)

        self._cache[key] = (time.time(), best)
        return best

    def best_move(self, fen: str, color: str | None = None, **kw) -> str:
        """Compatibilité : renvoie uniquement la SAN du meilleur coup."""
        return self.analyse_best(fen, color, **kw).san

    def top_moves(self,
                  fen: str,
                  color: str | None = None,
                  n: int = 3,
                  *,
                  movetime: float | None = None,
                  depth: int | None = None) -> list[MoveSuggestion]:
        """Les `n` meilleures lignes (MultiPV), triées de la meilleure à la pire."""
        board = self._board_for(fen, color)
        legal = board.legal_moves.count()
        if legal == 0:
            raise chess.engine.EngineError(
                f"Aucun coup légal ({'mat' if board.is_check() else 'pat'})"
            )

        engine = self._ensure_alive()
        limit = self._limit(movetime, depth)
        with self._lock:
            infos = engine.analyse(board, limit, multipv=max(1, min(n, legal)))

        if isinstance(infos, dict):            # MultiPV=1 → dict, pas liste
            infos = [infos]

        out: list[MoveSuggestion] = []
        for info in infos:
            try:
                out.append(self._to_suggestion(board, info))
            except (chess.engine.EngineError, ValueError):
                continue
        if not out:
            raise chess.engine.EngineError("Le moteur n'a proposé aucun coup exploitable")
        return out


# ── instance partagée ───────────────────────────────────────────────
_engine: BestMoveEngine | None = None
_engine_lock = threading.Lock()


def get_engine(**kwargs) -> BestMoveEngine:
    """
    Instance unique de Stockfish pour tout le processus.
    Les `kwargs` ne sont pris en compte qu'à la première création.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = BestMoveEngine(**kwargs)
    return _engine


def shutdown_engine() -> None:
    """Ferme le moteur partagé — idempotent, appelé aussi via atexit."""
    global _engine
    with _engine_lock:
        engine, _engine = _engine, None
    if engine is not None:
        engine.quit()


atexit.register(shutdown_engine)
