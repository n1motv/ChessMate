"""
utils.py – petits utilitaires autour de python-chess.

Changements :

* les noms de pièces ne sont plus figés en français (`_NAMES_FR`) : on renvoie
  la clé canonique anglaise, que l'interface traduit et qui sert directement
  de nom de fichier d'icône ;
* `ensure_san()` lève désormais un `MoveError` (sous-classe de `ValueError`)
  au lieu d'un `RuntimeError` générique, ce qui permet de le rattraper
  précisément sans masquer d'autres bugs.
"""
from __future__ import annotations

import chess

#: type de pièce → clé canonique (= nom du PNG dans assets/)
PIECE_KEYS: dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


class MoveError(ValueError):
    """Coup mal formé ou illégal dans la position donnée."""


def piece_key(piece_type: int) -> str:
    return PIECE_KEYS.get(piece_type, "pawn")


def describe_move(fen: str, san: str) -> tuple[str, str, str]:
    """
    (clé_pièce, case_départ, case_arrivée) pour le coup `san` dans `fen`.

    >>> describe_move(chess.STARTING_FEN, "e4")
    ('pawn', 'e2', 'e4')
    """
    board = chess.Board(fen)
    try:
        move = board.parse_san(san)
    except ValueError as exc:
        raise MoveError(f"Coup illisible : {san!r} dans {fen}") from exc

    piece = board.piece_at(move.from_square)
    if piece is None:                      # ne devrait pas arriver après parse_san
        raise MoveError(f"Aucune pièce sur {chess.square_name(move.from_square)}")

    return (piece_key(piece.piece_type),
            chess.square_name(move.from_square),
            chess.square_name(move.to_square))


def ensure_san(fen: str, move_txt: str) -> str:
    """
    Normalise `move_txt` (SAN, UCI ou LAN) en SAN **légale** pour `fen`.
    Lève `MoveError` si le coup est mal formé ou illégal.
    """
    board = chess.Board(fen)
    text = (move_txt or "").strip()
    if not text:
        raise MoveError("Coup vide")

    # ① déjà de la SAN valide ?
    try:
        move = board.parse_san(text)
        return board.san(move)             # renormalise (« Nf3 » vs « Ng1f3 »)
    except ValueError:
        pass

    # ② UCI / LAN → SAN
    try:
        move = chess.Move.from_uci(text.lower())
    except ValueError as exc:
        raise MoveError(f"Coup mal formé : {move_txt!r}") from exc

    if move not in board.legal_moves:
        raise MoveError(f"Coup illégal : {move_txt} dans {fen}")
    return board.san(move)


def parse_move(fen: str, san: str) -> chess.Move:
    """SAN → objet `chess.Move` validé."""
    board = chess.Board(fen)
    try:
        move = board.parse_san(san)
    except ValueError as exc:
        raise MoveError(f"Coup illisible : {san!r} dans {fen}") from exc
    if move not in board.legal_moves:
        raise MoveError(f"Coup illégal : {san} dans {fen}")
    return move


def format_score(score_pawns: float, mate_in: int | None = None) -> str:
    """Évaluation lisible : « +1.34 » ou « #3 »."""
    if mate_in is not None:
        return f"#{'-' if mate_in < 0 else ''}{abs(mate_in)}"
    return f"{score_pawns:+.2f}"
