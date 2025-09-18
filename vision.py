import os, mss, cv2, numpy as np
from PIL import Image
from ChessToFEN import chessClassifier

# ────────────────── détection du plateau ────────────────────────────
def detect_board_region(monitor=1):
    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[monitor])
        img    = cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board   = max(cnts, key=cv2.contourArea)

    peri   = cv2.arcLength(board, True)
    approx = cv2.approxPolyDP(board, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx.reshape(-1, 2)) if len(approx) >= 4 \
                 else cv2.boundingRect(board)

    return img, x, y, w, h

# ────────────────── capture → FEN ───────────────────────────────────
def screenshot_and_slice(out_dir="data", monitor=1):
    os.makedirs(out_dir, exist_ok=True)
    img, x, y, w, h = detect_board_region(monitor)
    board = img[y:y+h, x:x+w]
    board = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))

    sq = w // 8
    for r in range(8):
        for c in range(8):
            x0, y0 = c*sq, r*sq
            board.crop((x0, y0, x0+sq, y0+sq)).save(f"{out_dir}/{r*8+c:02}.png")

def screenshot_to_fen(side="w", monitor=1):
    screenshot_and_slice("data", monitor)
    matrix = chessClassifier.predict_pieces("data")
    fen    = chessClassifier.convert_to_fen(matrix)
    return f"{fen} {side} - - 0 1"

# ────────────────── coordonnées pixels ──────────────────────────────
_img, LEFT, TOP, W, H = detect_board_region()
SQUARE = W // 8

def square_to_xy(square:str) -> tuple[int,int]:
    col = ord(square[0]) - ord('a')          # a→0 … h→7
    row = 8 - int(square[1])                 # rang 8 (haut) → 0
    x = LEFT + col*SQUARE + SQUARE//2
    y = TOP  + row*SQUARE + SQUARE//2
    return x, y
