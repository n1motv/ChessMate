"""
Tests de fen.py – exécutables sans écran, sans GPU et sans Stockfish.

    python -m pytest tests/            (ou simplement : python tests/test_fen.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fen import (  # noqa: E402
    START_PLACEMENT,
    FenError,
    FenTracker,
    chars_to_placement,
    detect_en_passant,
    flip_chars,
    index_to_name,
    name_to_index,
    placement_to_chars,
    possible_castling,
)


def test_index_mapping():
    assert index_to_name(0) == "a8"
    assert index_to_name(7) == "h8"
    assert index_to_name(56) == "a1"
    assert index_to_name(63) == "h1"
    assert index_to_name(60) == "e1"
    for name in ("a1", "e4", "h8", "d5"):
        assert index_to_name(name_to_index(name)) == name


def test_placement_roundtrip():
    chars = placement_to_chars(START_PLACEMENT)
    assert len(chars) == 64
    assert chars[0] == "r" and chars[60] == "K"
    assert chars_to_placement(chars) == START_PLACEMENT


def test_flip_is_involutive():
    chars = placement_to_chars(START_PLACEMENT)
    assert flip_chars(flip_chars(chars)) == chars


def test_castling_rights_from_start():
    assert possible_castling(placement_to_chars(START_PLACEMENT)) == {"K", "Q", "k", "q"}
    # roi blanc déplacé en e2 → plus aucun roque blanc
    moved = "rnbqkbnr/pppppppp/8/8/8/8/PPPPKPPP/RNBQ1BNR"
    assert possible_castling(placement_to_chars(moved)) == {"k", "q"}


def test_start_position_gets_full_castling_rights():
    """Le bug historique : les droits de roque étaient toujours « - »."""
    tracker = FenTracker()
    reading = tracker.build(placement_to_chars(START_PLACEMENT), "w")
    assert reading.fen.split()[2] == "KQkq"
    assert chess.Board(reading.fen) == chess.Board()


def test_castling_move_is_legal_after_reading():
    placement = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R"
    tracker = FenTracker()
    reading = tracker.build(placement_to_chars(placement), "w")
    board = chess.Board(reading.fen)
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert chess.Move.from_uci("e1g1") in board.legal_moves


def test_castling_right_lost_forever():
    tracker = FenTracker()
    tracker.build(placement_to_chars(START_PLACEMENT), "w", commit=True)
    # la tour h1 bouge en g1 …
    moved = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBRN"
    tracker.build(placement_to_chars(moved), "b", commit=True)
    # … puis revient : le droit ne doit pas ressusciter
    reading = tracker.build(placement_to_chars(START_PLACEMENT), "w")
    assert "K" not in reading.fen.split()[2]
    assert "Q" in reading.fen.split()[2]


def test_en_passant_detected():
    before = placement_to_chars(START_PLACEMENT)
    after = placement_to_chars("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    assert detect_en_passant(before, after) == "e3"


def test_en_passant_in_fen_when_capturable():
    """e2-e4 alors qu'un pion noir est en d4 → « e3 » doit figurer au FEN."""
    tracker = FenTracker()
    before = "rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR"
    after = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR"
    tracker.build(placement_to_chars(before), "w", commit=True)
    reading = tracker.build(placement_to_chars(after), "b")
    assert reading.fen.split()[3] == "e3"
    board = chess.Board(reading.fen)
    assert chess.Move.from_uci("d4e3") in board.legal_moves


def test_uncapturable_en_passant_is_dropped():
    """Sans pion adverse adjacent, python-chess refuse la case : on dégrade."""
    tracker = FenTracker()
    tracker.build(placement_to_chars(START_PLACEMENT), "w", commit=True)
    after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    reading = tracker.build(placement_to_chars(after), "b")
    assert reading.fen.split()[3] == "-"
    assert "en_passant_dropped" in reading.warnings
    assert chess.Board(reading.fen).is_valid()


def test_halfmove_and_fullmove_counters():
    tracker = FenTracker()
    tracker.build(placement_to_chars(START_PLACEMENT), "w", commit=True)
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    r = tracker.build(placement_to_chars(after_e4), "b", commit=True)
    assert r.fen.split()[4] == "0"          # un pion a bougé → compteur remis à 0
    after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    r = tracker.build(placement_to_chars(after_e5), "w", commit=True)
    assert r.fen.split()[5] == "2"          # les Noirs ont complété le coup 1


def test_missing_king_raises():
    tracker = FenTracker()
    headless = "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    try:
        tracker.build(placement_to_chars(headless), "w")
    except FenError as exc:
        assert "roi" in str(exc)
    else:
        raise AssertionError("FenError attendue pour une position sans roi noir")


def test_commit_is_opt_in():
    """build() sans commit ne doit pas polluer le suivi."""
    tracker = FenTracker()
    tracker.build(placement_to_chars(START_PLACEMENT), "w", commit=True)
    moved = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBRN"
    tracker.build(placement_to_chars(moved), "b")            # pas de commit
    reading = tracker.build(placement_to_chars(START_PLACEMENT), "w")
    assert reading.fen.split()[2] == "KQkq"                  # droit intact


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
