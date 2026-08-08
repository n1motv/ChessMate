"""
worker.py – analyse « pure » : aucune dépendance à Qt, aucun état global.

Changements notables :

* `submit(side, lang)` **transmet réellement** la langue.  L'ancienne version
  déclarait `lang="fr"` puis appelait `_pool.submit(analyse, side)` : le
  paramètre était purement décoratif et les explications étaient toujours en
  français, quelle que soit la langue choisie dans l'interface.
* plus de variables de module (`_sf`, `_prev_layout`, `_first_run`) tripotées
  depuis `main.py` : tout est encapsulé dans `Analyzer`.
* le moteur est le **singleton** de `engine.get_engine()`, plus une instance
  privée qui doublonnait celle de `llm.py`.
* la validation de légalité passe par `utils.parse_move`, et les erreurs sont
  typées au lieu de disparaître dans un `except Exception:` muet.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import chess
import chess.engine

import engine as engine_mod
import llm
from classifier import ClassifierError
from fen import FenError, FenTracker
from utils import MoveError, describe_move, parse_move
from vision import (
    BoardGeometry, BoardLocator, ScreenReading, VisionError,
    board_region, grab_screen, read_board,
)

log = logging.getLogger(__name__)


class AnalysisError(RuntimeError):
    """Erreur exploitable par l'interface (message déjà lisible)."""

    def __init__(self, message: str, key: str = "err_generic") -> None:
        super().__init__(message)
        self.key = key          # clé i18n : err_engine / err_vision / …


@dataclass
class Analysis:
    """Résultat complet d'une analyse, prêt à être affiché."""
    fen: str
    placement: str
    san: str
    move: chess.Move
    piece: str                 # clé canonique : "pawn", "knight", …
    src: str
    dst: str
    score_pawns: float
    mate_in: int | None
    pv_san: str
    explanation: str
    geometry: BoardGeometry
    flipped: bool
    confidence: float
    from_llm: bool
    llm_error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def board_before(self) -> chess.Board:
        return chess.Board(self.fen)

    @property
    def board_after(self) -> chess.Board:
        board = self.board_before
        board.push(self.move)
        return board

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None


class Analyzer:
    """
    Orchestre vision → FEN → moteur → LLM.

    Cycle d'utilisation :
        a = Analyzer()
        res = a.analyse("white", "fr")     # ne modifie pas l'état de suivi
        ...  # on joue ou on surligne le coup
        a.commit(res)                      # alors seulement on avance
    """

    # nombre de lectures identiques d'affilée exigées avant d'agir sur un
    # changement de position — voir `analyse()`.
    _STABLE_READS = 2

    def __init__(self, locator: BoardLocator | None = None) -> None:
        self.locator = locator or BoardLocator()
        self.tracker = FenTracker()
        self.user_side = "white"          # couleur jouée par l'utilisateur
        self._last_placement: str | None = None
        self._pending_placement: str | None = None
        self._pending_count = 0
        self._first_run = True
        # cache de la dernière capture d'écran : évite de redécouper et de
        # reclassifier les 64 cases quand rien n'a changé à l'écran (voir
        # `read()`), ce qui est le cas la quasi-totalité du temps pendant
        # qu'on attend l'adversaire.
        self._frame_hash: bytes | None = None
        self._cached_reading: ScreenReading | None = None

    # ── état ────────────────────────────────────────────────────────
    @property
    def flipped(self) -> bool:
        """
        Le plateau est retourné quand l'utilisateur joue les Noirs.

        C'est bien la couleur *jouée* qui compte, pas le trait : l'affichage
        ne pivote pas d'un coup à l'autre.  L'ancienne version ne gérait pas
        du tout l'orientation et lisait toujours le plateau côté Blancs.
        """
        return self.user_side == "black"

    def reset(self) -> None:
        self.tracker.reset()
        self._last_placement = None
        self._pending_placement = None
        self._pending_count = 0
        self._first_run = True
        self._frame_hash = None
        self._cached_reading = None

    def resync(self, placement: str) -> None:
        """
        Recale le suivi sur ce qui est réellement affiché à l'écran, sans
        repartir de zéro comme `reset()` : on garde la certitude que ce
        n'est plus notre tour. Utilisé après un coup dont l'exécution n'a
        pas pu être confirmée (`AnalysisThread._verify`) — sans ça, un
        `reset()` traitait la position actuelle (juste après NOTRE coup)
        comme un tout premier tour et retentait aussitôt de jouer, avant
        même que l'adversaire ait répondu.
        """
        self._last_placement = placement
        self._pending_placement = None
        self._pending_count = 0

    def recalibrate(self) -> BoardGeometry:
        geom = self.locator.recalibrate()
        self.reset()
        return geom

    # ── lecture ─────────────────────────────────────────────────────
    def read(self, side: str, *, flipped: bool | None = None) -> ScreenReading:
        """
        Lecture brute de l'écran, sans analyse moteur.

        Le plateau ne change qu'après un coup — le reste du temps, chaque
        capture est pixel pour pixel identique à la précédente. On compare
        donc d'abord un hachage léger de la zone du plateau : si rien n'a
        bougé, on renvoie la dernière lecture telle quelle au lieu de
        redécouper les 64 cases et de les repasser dans le classifieur —
        l'essentiel du coût CPU/RAM de la boucle de surveillance.
        """
        if flipped is None:
            flipped = self.flipped
        try:
            geom = self.locator.get()
            frame = grab_screen(geom.monitor)
            frame_hash = hashlib.blake2b(
                board_region(frame, geom).tobytes(), digest_size=16
            ).digest()
            if frame_hash == self._frame_hash and self._cached_reading is not None:
                return self._cached_reading

            reading = read_board(self.locator, self.tracker, side, flipped=flipped)
            self._frame_hash = frame_hash
            self._cached_reading = reading
            return reading
        except VisionError as exc:
            raise AnalysisError(str(exc), "err_vision") from exc
        except ClassifierError as exc:
            raise AnalysisError(str(exc), "err_model") from exc
        except FenError as exc:
            raise AnalysisError(str(exc), "err_vision") from exc

    # ── analyse ─────────────────────────────────────────────────────
    def analyse(self,
                side: str,
                lang: str = "en",
                *,
                flipped: bool | None = None,
                force: bool = False) -> Analysis | None:
        """
        → `Analysis` si un nouveau coup est à jouer, `None` si la position n'a
        pas changé depuis la dernière validation (anti-rebond).

        Lève `AnalysisError` avec une clé i18n en cas de problème réel.
        """
        self.user_side = side
        chess_side = "w" if side == "white" else "b"
        reading = self.read(chess_side, flipped=flipped)

        # ── anti-rebond ─────────────────────────────────────────────
        if not force:
            if reading.placement == self._last_placement:
                self._pending_placement = None
                self._pending_count = 0
                return None
            # Au tout premier tour en jouant les Noirs, la position de départ
            # n'est pas la nôtre : on attend le coup des Blancs.
            if (self._first_run and chess_side == "b"
                    and reading.placement == chess.STARTING_BOARD_FEN):
                self._last_placement = reading.placement
                self.tracker.commit(reading.chars, chess_side)
                self._first_run = False
                return None

            # Un changement de position ne suffit pas à lui seul : une case
            # mal lue en cours d'animation (glisser-déposer de l'adversaire,
            # surlignage transitoire...) produit exactement le même symptôme
            # qu'un coup réellement joué, et faisait auparavant calculer et
            # jouer un coup **avant que l'adversaire ait fini de jouer le
            # sien** — l'app perdait ainsi la main sur le trait réel. On
            # n'agit donc que sur une position revue identique deux captures
            # d'affilée (~`IDLE_POLL` secondes d'écart) : un vrai coup reste
            # stable, un artefact d'animation ne l'est pas.
            if reading.placement != self._pending_placement:
                self._pending_placement = reading.placement
                self._pending_count = 1
                return None
            self._pending_count += 1
            if self._pending_count < self._STABLE_READS:
                return None

        self._pending_placement = None
        self._pending_count = 0
        self._first_run = False

        board = reading.reading.board
        if board.is_game_over():
            raise AnalysisError(
                f"Partie terminée ({board.result()})", "game_over"
            )

        # ── choix du coup ───────────────────────────────────────────
        try:
            choice = llm.choose_move_and_explain(reading.fen, side, lang=lang)
        except engine_mod.EngineUnavailable as exc:
            raise AnalysisError(str(exc), "err_engine") from exc
        except chess.engine.EngineError as exc:
            raise AnalysisError(str(exc), "err_engine") from exc
        except ValueError as exc:          # FEN au mauvais trait
            raise AnalysisError(str(exc), "err_vision") from exc

        # ── validation finale ───────────────────────────────────────
        try:
            move = parse_move(reading.fen, choice.san)
            piece, src, dst = describe_move(reading.fen, choice.san)
        except MoveError as exc:
            raise AnalysisError(str(exc), "err_engine") from exc

        return Analysis(
            fen=reading.fen,
            placement=reading.placement,
            san=choice.san,
            move=move,
            piece=piece,
            src=src,
            dst=dst,
            score_pawns=choice.score_pawns,
            mate_in=choice.mate_in,
            pv_san=choice.pv_san,
            explanation=choice.explanation,
            geometry=reading.geometry,
            flipped=reading.flipped,
            confidence=reading.confidence,
            from_llm=choice.from_llm,
            llm_error=choice.error,
            warnings=list(reading.reading.warnings),
        )

    # ── validation ──────────────────────────────────────────────────
    def commit(self, analysis: Analysis, *, played: bool) -> None:
        """
        Enregistre l'avancement de la partie.

        `played=True`  : le coup a été exécuté → on avance jusqu'à la position
                         qui suit le coup.
        `played=False` : on s'est contenté de surligner → la position à
                         l'écran est encore celle d'avant le coup.
        """
        board = analysis.board_after if played else analysis.board_before
        placement = board.board_fen()
        side_after = "w" if board.turn == chess.WHITE else "b"
        try:
            self.tracker.commit_placement(placement, side_after)
        except FenError as exc:
            log.warning("Suivi de partie désynchronisé : %s", exc)
            self.tracker.reset()
        self._last_placement = placement

    def commit_placement(self, placement: str) -> None:
        """Force la position de référence (utilisé après un échec d'exécution)."""
        self._last_placement = placement

    # ── arrêt ───────────────────────────────────────────────────────
    def close(self) -> None:
        engine_mod.shutdown_engine()


# ── compatibilité : API de module ───────────────────────────────────
_default: Analyzer | None = None


def get_analyzer() -> Analyzer:
    global _default
    if _default is None:
        _default = Analyzer()
    return _default


def reset() -> None:
    get_analyzer().reset()


def analyse(side: str, lang: str = "en") -> Analysis | None:
    return get_analyzer().analyse(side, lang)


def shutdown() -> None:
    global _default
    if _default is not None:
        _default.close()
        _default = None
    else:
        engine_mod.shutdown_engine()


__all__ = [
    "Analysis", "AnalysisError", "Analyzer",
    "analyse", "get_analyzer", "reset", "shutdown",
]
