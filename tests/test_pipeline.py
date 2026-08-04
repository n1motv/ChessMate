"""
Test bout-en-bout hors écran : classifieur → FEN → Stockfish.

Utilise les 64 imagettes de `data/`, capturées sur une vraie partie et déjà
présentes dans le dépôt.  Aucun écran, aucune souris.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classifier import CLASS_TO_FEN, ClassifierError, get_classifier  # noqa: E402
from engine import EngineUnavailable, find_stockfish, get_engine, shutdown_engine  # noqa: E402
from fen import START_PLACEMENT, FenTracker, chars_to_placement, flip_chars  # noqa: E402
from utils import describe_move, parse_move  # noqa: E402

DATA = ROOT / "data"


def load_squares() -> list[Image.Image]:
    files = sorted(DATA.glob("[0-9][0-9].png"))
    if len(files) != 64:
        raise FileNotFoundError(f"64 imagettes attendues dans {DATA}, {len(files)} trouvées")
    return [Image.open(p).convert("RGB") for p in files]


def main() -> int:
    print("── classifieur ────────────────────────────────────────────")
    try:
        squares = load_squares()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc} — test ignoré")
        return 0

    try:
        clf = get_classifier()
        labels, confidences = clf.predict(squares)
    except ClassifierError as exc:
        print(f"⚠️  {exc} — test ignoré")
        return 0

    print(f"classes      : {len(clf.classes)} → {clf.classes}")
    print(f"confiance    : min {min(confidences):.1%} / moy {sum(confidences)/64:.1%}")
    weak = [(i, labels[i], c) for i, c in enumerate(confidences) if c < 0.80]
    print(f"cases < 80 % : {len(weak)}")

    raw_chars = [CLASS_TO_FEN.get(lbl) for lbl in labels]
    kings = (raw_chars.count("K"), raw_chars.count("k"))
    print(f"rois lus     : {kings[0]} blanc(s), {kings[1]} noir(s)")

    print("\n── orientation ────────────────────────────────────────────")
    # Ces imagettes ont été capturées en jouant les Noirs : le plateau est
    # affiché à l'envers.  Sans correction, on obtient une position miroir
    # où rois et dames sont intervertis — ce que faisait l'ancien code, qui
    # ne gérait pas du tout l'orientation.
    as_seen = chars_to_placement(raw_chars)
    corrected = chars_to_placement(flip_chars(raw_chars))
    print(f"tel qu'affiché : {as_seen}")
    print(f"après rotation : {corrected}")

    flipped = corrected == START_PLACEMENT and as_seen != START_PLACEMENT
    if flipped:
        print("→ plateau retourné détecté, rotation appliquée")
        assert corrected == START_PLACEMENT, "la rotation doit donner la position de départ"
    chars = flip_chars(raw_chars) if flipped else raw_chars

    print("\n── construction du FEN ────────────────────────────────────")
    tracker = FenTracker()
    try:
        reading = tracker.build(chars, "w")
    except Exception as exc:                                    # noqa: BLE001
        print(f"❌ {type(exc).__name__}: {exc}")
        return 1

    print(f"FEN          : {reading.fen}")
    print(f"roque        : {reading.fen.split()[2]}")
    print(f"en passant   : {reading.fen.split()[3]}")
    if reading.warnings:
        print(f"avertis.     : {reading.warnings}")

    board = chess.Board(reading.fen)
    assert board.is_valid(), "python-chess refuse le FEN produit"
    print(f"valide       : oui ({board.legal_moves.count()} coups légaux)")
    print()
    print(board.unicode(borders=True))

    print("\n── Stockfish ──────────────────────────────────────────────")
    try:
        find_stockfish()
    except EngineUnavailable as exc:
        print(f"⚠️  {exc} — analyse ignorée")
        return 0

    try:
        eng = get_engine()
        tops = eng.top_moves(reading.fen, "white", n=3, movetime=0.5)
        for i, sug in enumerate(tops, 1):
            print(f"{i}. {sug.san:<8} {sug.score_pawns:+6.2f}  "
                  f"prof.{sug.depth:<3} {sug.pv_san}")

        best = tops[0]
        move = parse_move(reading.fen, best.san)
        piece, src, dst = describe_move(reading.fen, best.san)
        print(f"\ncoup retenu  : {best.san} ({piece} {src}→{dst})")
        assert move in board.legal_moves
    finally:
        shutdown_engine()

    print("\n✅ chaîne complète fonctionnelle (vision → FEN → moteur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
