# llm.py  ──────────────────────────────────────────────────────────────
import os, json, requests, textwrap, chess
from engine import BestMoveEngine
from utils  import ensure_san          # même fonction qu’avant

API_BASE = os.getenv("LM_ENDPOINT", "http://localhost:1234/v1")
API_KEY  = os.getenv("LM_API_KEY", "lm-studio")
MODEL    = os.getenv("LM_MODEL",    "dolphin-2.6-mistral-7b")

_sf = BestMoveEngine(depth=18)         # instance unique Stockfish

def choose_move_and_explain(fen: str, color: str,lang: str = "fr", top_n: int = 3):
    """
    Renvoie (move_SAN, explanation_fr, score_cp)    – 0.01 = un cent‑pion.
    LLM **doit** choisir l’un des `top_n` coups proposés par Stockfish.
    """

    # ── 1) on demande les N meilleurs coups à Stockfish ───────────────
    top = _sf.top_moves(fen, color, n=top_n)      # [(san, cp, pv5), …]
    menu_txt = "\n".join(f"{i+1}. {san}  ({cp:+.2f})"
                         for i,(san,cp,_) in enumerate(top))

    # premier coup = score de référence
    ref_move, ref_cp, ref_pv = top[0]

    side = "Blancs" if color == "white" else "Noirs"
    board_ascii = chess.Board(fen).unicode()
    # petits libellés dépendants de la langue
    L = {
        "fr": dict(side=("Blancs","Noirs"), explain="2–3 phrases FR", lang_hint="FR"),
        "en": dict(side=("White","Black"),  explain="2–3 sentences EN", lang_hint="EN")
    }[lang]
    side_str = L["side"][0] if color=="white" else L["side"][1]
    # ── 2) prompt ultra‑contraint  ────────────────────────────────────
    sys_msg = (
        "You are a chess grand‑master." if lang=="en"
        else "Tu es un grand‑maître d'échecs."
    )
    sys_msg += (
        " Choose **exactly one** move from the list below; never invent another."
        if lang=="en" else
        " Choisis **exactement un** coup dans la liste ci‑dessous ; n'invente jamais."
    )

    user_msg = textwrap.dedent(f"""
        Position FEN : {fen}
        Side to move : {side_str}
        Board :
        {board_ascii}

        Top {top_n} engine moves:
        {menu_txt}

        Reply STRICTLY with this JSON object (no other text):
        {{
          "move": "<one of the moves>",
          "explanation": "<{L["explain"]}>",
          "score": <numeric evaluation in pawns>
        }}
    """)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg}
        ],
        "temperature": 0.3,
        "max_tokens" : 160
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    txt = requests.post(f"{API_BASE}/chat/completions",
                        headers=headers, json=payload, timeout=60
                      ).json()["choices"][0]["message"]["content"].strip()

    # ── 3) parsing & validation ───────────────────────────────────────
    try:
        data  = json.loads(txt)
        move  = data["move"].strip()
        expl  = data["explanation"].strip()
        score = float(data.get("score", ref_cp))
    except Exception as e:
        raise RuntimeError(f"Réponse LLM mal formée : {txt!r}") from e

    # – vérifie qu’il a bien choisi dans la liste
    allowed = {m for m,_,_ in top}
    if move not in allowed:
        raise RuntimeError(f"Coup hors‑liste : {move}  (liste : {allowed})")

    # – vérifie une dernière fois la légalité
    move = ensure_san(fen, move)   # lève RuntimeError si problème
    return move, expl, score
