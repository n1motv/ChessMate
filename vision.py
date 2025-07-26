# vision.py  ───────────────────────────────────────────────────────────
import os, mss
from PIL import Image
from ChessToFEN import chessClassifier   # assure-toi que le path est correct

# Coordonnées adaptées à ton écran 
LEFT   = 441          # x de la case a8
TOP    = 237          # y de la case a8
SIZE   = 1088          # l’échiquier est carré 
SQUARE = SIZE // 8    

def screenshot_and_slice(output_dir="data", monitor=1):
    """Capture l'écran et découpe l'échiquier en 64 cases (8 × 8)."""
    os.makedirs(output_dir, exist_ok=True)

    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[monitor])
        img = Image.frombytes("RGB", screen.size, screen.rgb)

    # on isole l’échiquier
    img = img.crop((LEFT, TOP, LEFT + SIZE, TOP + SIZE))

    # on découpe en 64 cases
    for row in range(8):
        for col in range(8):
            x0, y0 = col * SQUARE, row * SQUARE
            x1, y1 = x0 + SQUARE, y0 + SQUARE
            img.crop((x0, y0, x1, y1)) \
               .save(os.path.join(output_dir, f"{row*8+col:02}.png"))

def screenshot_to_fen(side="w"):            # side = "w" ou "b"
    screenshot_and_slice("data")            # ← REMIS
    board = chessClassifier.predict_pieces("data")
    fen   = chessClassifier.convert_to_fen(board)

    cast  = "-"      # ou "KQkq" si tu veux
    ep    = "-"
    hm,fm = "0","1"
    return f"{fen} {side} {cast} {ep} {hm} {fm}"
