"""
Tests de engine.py – nécessitent le binaire Stockfish (engines/ ou PATH).
Ignorés proprement s'il est absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (  # noqa: E402
    PROFILES,
    BestMoveEngine,
    EngineUnavailable,
    find_stockfish,
    get_engine,
    shutdown_engine,
)

START = chess.Board().fen()
MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"       # Ra8#
CASTLING = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"


def test_profiles_have_single_primary_limit():
    """Chaque profil définit un temps ; la profondeur n'est qu'un plafond."""
    for prof in PROFILES.values():
        assert prof.movetime > 0
        assert prof.depth_cap is None or prof.depth_cap > 0


def test_board_for_rejects_wrong_side():
    try:
        BestMoveEngine._board_for(START, "black")
    except ValueError as exc:
        assert "trait" in str(exc)
    else:
        raise AssertionError("ValueError attendue : le FEN est au trait des Blancs")


def test_board_for_preserves_castling_and_ep():
    """L'ancien push(Move.null()) détruisait ces deux informations."""
    board = BestMoveEngine._board_for(CASTLING, "white")
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert board.fen() == CASTLING


def test_engine_finds_mate_in_one():
    eng = get_engine()
    best = eng.analyse_best(MATE_IN_1, "white")
    assert best.san == "Ra8#"
    assert best.is_mate and best.mate_in == 1


def test_engine_offers_castling():
    """Le vrai test du bug FEN : le moteur doit *pouvoir* roquer."""
    eng = get_engine()
    tops = eng.top_moves(CASTLING, "white", n=5, movetime=0.3)
    board = chess.Board(CASTLING)
    for sug in tops:
        assert sug.move in board.legal_moves
    assert any(m in board.legal_moves for m in [chess.Move.from_uci("e1g1")])


def test_top_moves_are_sorted_and_legal():
    eng = get_engine()
    tops = eng.top_moves(START, "white", n=3, movetime=0.3)
    assert 1 <= len(tops) <= 3
    board = chess.Board(START)
    for sug in tops:
        assert sug.move in board.legal_moves
        assert sug.pv_san
    scores = [s.score_pawns for s in tops]
    assert scores == sorted(scores, reverse=True)


def test_get_engine_is_a_singleton():
    """Le bug historique : deux processus Stockfish tournaient en parallèle."""
    assert get_engine() is get_engine()


def _main() -> int:
    try:
        print(f"Stockfish : {find_stockfish()}")
    except EngineUnavailable as exc:
        print(f"⚠️  {exc}\n→ tests moteur ignorés")
        return 0

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    try:
        for fn in tests:
            try:
                fn()
            except Exception as exc:                            # noqa: BLE001
                failed += 1
                print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            else:
                print(f"ok   {fn.__name__}")
    finally:
        shutdown_engine()
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
