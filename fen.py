"""
fen.py – construction d'un FEN **complet** à partir d'une simple grille de
64 cases reconnues à l'écran.

Avant, `vision.py` renvoyait  ``f"{placement} {side} - - 0 1"`` : les droits
de roque étaient toujours vides et l'en passant jamais renseigné.  Stockfish
ne pouvait donc **jamais** proposer O-O / O-O-O ni une prise en passant.

Ici, `FenTracker` conserve l'historique des positions observées et en déduit :

* **les droits de roque** — disponibles tant que le roi et la tour concernés
  n'ont jamais quitté leur case d'origine sur l'ensemble des captures vues ;
* **la case d'en passant** — détectée en comparant deux captures successives
  (un pion qui avance de deux rangées) ;
* **les compteurs** de demi-coups et de coups.

Limite assumée : si l'application démarre en milieu de partie, on ne connaît
pas l'historique.  On part alors de l'hypothèse optimiste « le roque est
encore possible si les pièces sont sur leurs cases d'origine », ce qui est
juste dans l'immense majorité des cas et jamais pire que l'ancien « - ».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chess

log = logging.getLogger(__name__)

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# index de case → 0 = a8 (coin haut-gauche vu des Blancs) … 63 = h1
E1, A1, H1 = 60, 56, 63
E8, A8, H8 = 4, 0, 7

# droit de roque → (case du roi, case de la tour, pièces attendues)
_CASTLING_HOME: dict[str, tuple[int, int, str, str]] = {
    "K": (E1, H1, "K", "R"),
    "Q": (E1, A1, "K", "R"),
    "k": (E8, H8, "k", "r"),
    "q": (E8, A8, "k", "r"),
}


class FenError(ValueError):
    """La grille observée ne forme pas une position d'échecs exploitable."""


# ── conversions grille ↔ FEN ────────────────────────────────────────
def index_to_square(idx: int) -> int:
    """index écran (0 = a8) → case python-chess (0 = a1)."""
    return chess.square(idx % 8, 7 - idx // 8)


def index_to_name(idx: int) -> str:
    return chess.square_name(index_to_square(idx))


def name_to_index(name: str) -> int:
    sq = chess.parse_square(name)
    return (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)


def chars_to_placement(chars: list[str | None]) -> str:
    """
    64 caractères FEN (ou None pour une case vide), rangés de a8 à h1,
    → champ « placement » d'un FEN.
    """
    if len(chars) != 64:
        raise FenError(f"64 cases attendues, {len(chars)} reçues")

    rows: list[str] = []
    for r in range(8):
        row, empty = "", 0
        for c in range(8):
            piece = chars[r * 8 + c]
            if piece is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def placement_to_chars(placement: str) -> list[str | None]:
    """Opération inverse de `chars_to_placement`."""
    chars: list[str | None] = []
    for row in placement.split("/"):
        for ch in row:
            if ch.isdigit():
                chars.extend([None] * int(ch))
            else:
                chars.append(ch)
    if len(chars) != 64:
        raise FenError(f"Placement invalide : {placement!r}")
    return chars


def flip_chars(chars: list[str | None]) -> list[str | None]:
    """Rotation de 180° — utile quand le plateau est affiché côté Noirs."""
    return list(reversed(chars))


# ── inférences ──────────────────────────────────────────────────────
def possible_castling(chars: list[str | None]) -> set[str]:
    """Droits de roque physiquement plausibles pour cette grille."""
    return {
        right
        for right, (k_sq, r_sq, k_piece, r_piece) in _CASTLING_HOME.items()
        if chars[k_sq] == k_piece and chars[r_sq] == r_piece
    }


def detect_en_passant(prev: list[str | None],
                      curr: list[str | None]) -> str | None:
    """
    Compare deux grilles successives et renvoie la case d'en passant si le
    dernier coup était une avancée de pion de deux cases, sinon None.
    """
    if prev is None:
        return None

    changed = [i for i in range(64) if prev[i] != curr[i]]
    # une avancée simple touche exactement 2 cases (départ vidé, arrivée
    # remplie) ; plus que ça = roque, capture multiple ou détection bruitée
    if len(changed) != 2:
        return None

    for src in changed:
        for dst in changed:
            if src == dst:
                continue
            # Blancs : rangée 2 → rangée 4 (indices 48-55 → 32-39)
            if (prev[src] == "P" and curr[dst] == "P"
                    and 48 <= src <= 55 and dst == src - 16
                    and curr[src] is None):
                return index_to_name(src - 8)
            # Noirs : rangée 7 → rangée 5 (indices 8-15 → 24-31)
            if (prev[src] == "p" and curr[dst] == "p"
                    and 8 <= src <= 15 and dst == src + 16
                    and curr[src] is None):
                return index_to_name(src + 8)
    return None


def _piece_count(chars: list[str | None]) -> int:
    return sum(1 for c in chars if c is not None)


def _pawns_moved(prev: list[str | None], curr: list[str | None]) -> bool:
    prev_pawns = {i for i, c in enumerate(prev) if c in ("P", "p")}
    curr_pawns = {i for i, c in enumerate(curr) if c in ("P", "p")}
    return prev_pawns != curr_pawns


# ── résultat ────────────────────────────────────────────────────────
@dataclass
class BoardReading:
    """Une lecture d'échiquier, FEN validé compris."""
    fen: str
    placement: str
    side: str                     # "w" | "b"
    warnings: list[str] = field(default_factory=list)

    @property
    def board(self) -> chess.Board:
        return chess.Board(self.fen)


# ── suivi de partie ─────────────────────────────────────────────────
class FenTracker:
    """
    Mémorise ce qui ne se lit pas sur une simple photo de l'échiquier :
    droits de roque perdus, dernier coup joué, compteurs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev: list[str | None] | None = None
        self._lost_castling: set[str] = set()
        self._halfmove = 0
        self._fullmove = 1
        self._seen_first = False

    # ── état ────────────────────────────────────────────────────────
    @property
    def previous_placement(self) -> str | None:
        return chars_to_placement(self._prev) if self._prev else None

    def _update_castling_memory(self, chars: list[str | None]) -> None:
        """Un droit perdu une fois l'est définitivement."""
        available = possible_castling(chars)
        self._lost_castling |= set(_CASTLING_HOME) - available

    # ── API principale ──────────────────────────────────────────────
    def build(self, chars: list[str | None], side: str,
              *, commit: bool = False) -> BoardReading:
        """
        Assemble un FEN complet et **validé** pour la grille `chars`.

        `commit=False` (défaut) n'altère pas l'état interne : on peut donc
        analyser une position sans polluer le suivi si l'utilisateur annule.
        Appeler `commit()` une fois le coup réellement joué.
        """
        if side not in ("w", "b"):
            raise FenError(f"Trait invalide : {side!r}")

        placement = chars_to_placement(chars)
        warnings: list[str] = []

        # ── cohérence élémentaire avant de déranger python-chess ────
        kings = {"K": chars.count("K"), "k": chars.count("k")}
        if kings["K"] != 1 or kings["k"] != 1:
            raise FenError(
                f"Position impossible : {kings['K']} roi(s) blanc(s), "
                f"{kings['k']} roi(s) noir(s). La reconnaissance a échoué."
            )

        if commit:
            self._update_castling_memory(chars)

        # Un droit est offert s'il est physiquement présent *et* n'a jamais
        # été perdu au cours des positions déjà observées.
        rights = possible_castling(chars) - self._lost_castling
        castling = "".join(r for r in "KQkq" if r in rights) or "-"

        ep = detect_en_passant(self._prev, chars) if self._prev else None

        halfmove, fullmove = self._halfmove, self._fullmove
        if self._prev is not None:
            if _pawns_moved(self._prev, chars) or _piece_count(chars) != _piece_count(self._prev):
                halfmove = 0
            else:
                halfmove += 1
            if side == "w":
                fullmove += 1

        fen = self._assemble(placement, side, castling, ep, halfmove, fullmove, warnings)

        if commit:
            self._prev = list(chars)
            self._halfmove, self._fullmove = halfmove, fullmove
            self._seen_first = True

        return BoardReading(fen=fen, placement=placement, side=side, warnings=warnings)

    @staticmethod
    def _assemble(placement: str, side: str, castling: str, ep: str | None,
                  halfmove: int, fullmove: int, warnings: list[str]) -> str:
        """
        Assemble le FEN puis le fait valider par python-chess, en dégradant
        progressivement les champs déduits si la position est refusée.
        Objectif : ne **jamais** renvoyer un FEN que le moteur rejettera.
        """
        attempts = [
            (castling, ep or "-"),
            (castling, "-"),          # l'en passant est le plus fragile
            ("-", "-"),               # dernier recours : ancien comportement
        ]
        last_error = ""
        for cast, ep_field in attempts:
            candidate = f"{placement} {side} {cast} {ep_field} {halfmove} {fullmove}"
            try:
                board = chess.Board(candidate)
            except ValueError as exc:
                last_error = str(exc)
                continue

            status = board.status()

            # Anomalies structurelles : aucune variante de roque ou d'en
            # passant ne les corrigera, inutile d'insister.
            if status & _FATAL_STATUS:
                raise FenError(
                    f"Position illisible (status={status}) pour {placement!r} "
                    f"au trait « {side} ». Vérifiez la couleur sélectionnée "
                    "et la calibration du plateau."
                )

            # Anomalies imputables aux champs déduits : on dégrade.
            if status & _FIXABLE_STATUS:
                last_error = f"status={status}"
                continue

            # Une case d'en passant qu'aucun pion ne peut réellement capturer
            # est inutile *et* nuisible : elle modifie la clé Zobrist et fausse
            # la détection de répétition. On la retire (même convention que
            # python-chess, dont `Board.fen()` filtre par défaut).
            if ep_field != "-" and not board.has_legal_en_passant():
                last_error = "en passant non capturable"
                continue

            if cast != castling:
                warnings.append("castling_dropped")
            if ep and ep_field == "-":
                warnings.append("en_passant_dropped")
            if status != chess.STATUS_VALID:
                warnings.append(f"status:{status}")
            return candidate

        raise FenError(
            f"Impossible de construire un FEN valide pour {placement!r} "
            f"({last_error})."
        )

    def commit(self, chars: list[str | None], side: str) -> None:
        """Valide définitivement une lecture (appelé après exécution du coup)."""
        self.build(chars, side, commit=True)

    def commit_placement(self, placement: str, side: str) -> None:
        """Variante prenant un placement FEN (utile après `board.push()`)."""
        self.commit(placement_to_chars(placement), side)


def _status_flag(name: str) -> int:
    """Certains drapeaux n'existent pas sur les vieilles versions de python-chess."""
    return getattr(chess, name, 0)


# Anomalies structurelles : la reconnaissance a échoué, mieux vaut le dire.
_FATAL_STATUS = (
    _status_flag("STATUS_NO_WHITE_KING")
    | _status_flag("STATUS_NO_BLACK_KING")
    | _status_flag("STATUS_TOO_MANY_KINGS")
    | _status_flag("STATUS_PAWNS_ON_BACKRANK")
    | _status_flag("STATUS_OPPOSITE_CHECK")
    | _status_flag("STATUS_TOO_MANY_CHECKERS")
    | _status_flag("STATUS_IMPOSSIBLE_CHECK")
)

# Anomalies imputables aux champs que l'on déduit : on peut les retirer.
_FIXABLE_STATUS = (
    _status_flag("STATUS_BAD_CASTLING_RIGHTS")
    | _status_flag("STATUS_INVALID_EP_SQUARE")
)
