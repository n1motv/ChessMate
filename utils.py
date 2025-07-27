# utils.py-------------------------------------
import chess

_NAMES_FR = {
    chess.PAWN:   "pion",
    chess.KNIGHT: "cavalier",
    chess.BISHOP: "fou",
    chess.ROOK:   "tour",
    chess.QUEEN:  "dame",
    chess.KING:   "roi",
}

def describe_move(fen: str, san: str):
    """
    Prend la position `fen` et le coup en notation SAN (`san`)
    → retourne (nom_pièce, case_from, case_to).

    >>> describe_move(START_FEN, "e4")
    ('pion', 'e2', 'e4')
    """
    board = chess.Board(fen)
    move  = board.parse_san(san)       # SAN → Move
    piece = board.piece_at(move.from_square)

    name  = _NAMES_FR[piece.piece_type]
    src   = chess.square_name(move.from_square)
    dst   = chess.square_name(move.to_square)
    return name, src, dst


def ensure_san(fen: str, move_txt: str) -> str:
    """
    Garantit que `move_txt` est la SAN *légale* correspondant à la
    position FEN.  - Accepte SAN, UCI ou LAN.
    - Lève RuntimeError si le coup est illégal ou mal formé.
    """
    board = chess.Board(fen)

    # ① SAN déjà correct ?
    try:
        board.parse_san(move_txt)
        return move_txt.strip()
    except ValueError:
        pass

    # ② UCI / LAN → SAN
    try:
        mv = chess.Move.from_uci(move_txt.lower())
    except ValueError:
        raise RuntimeError(f"Coup mal formé : {move_txt!r}")

    if mv not in board.legal_moves:
        raise RuntimeError(f"Coup illégal : {move_txt} dans {fen}")

    return board.san(mv)
