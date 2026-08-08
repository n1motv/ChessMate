"""
llm.py – explication du meilleur coup par un LLM local (API compatible OpenAI).

Corrections apportées :

* **la langue fonctionne enfin.**  L'ancien code portait un dictionnaire codé
  en dur ne contenant que « fr » et « en » : sélectionner l'espagnol, le russe,
  le chinois ou l'arabe levait un `KeyError`.  Tous les libellés viennent
  désormais de `lang/*.json` via `i18n.tr()`.
* **plus de second Stockfish.**  `_sf = BestMoveEngine(depth=18)` au niveau
  module créait un deuxième processus moteur, jamais fermé.  On utilise le
  singleton de `engine.get_engine()`.
* **les erreurs réseau ne sont plus avalées.**  `raise_for_status()`, timeout
  court et configurable, un retry, et des exceptions typées (`LLMError`) que
  l'interface peut afficher au lieu de faire disparaître le problème.
* **plus d'`IndexError`** quand la position n'offre aucun coup légal.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import textwrap
import time
from dataclasses import dataclass

import chess
import requests

import config
import i18n
from engine import MoveSuggestion, get_engine

log = logging.getLogger(__name__)

API_BASE = os.getenv("LM_ENDPOINT", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_API_KEY", "lm-studio")
MODEL = os.getenv("LM_MODEL", "dolphin-2.6-mistral-7b")


class LLMError(RuntimeError):
    """Le LLM est injoignable ou sa réponse est inexploitable."""


@dataclass
class MoveChoice:
    """Coup retenu, avec son explication éventuelle."""
    san: str
    explanation: str
    score_pawns: float
    mate_in: int | None
    pv_san: str
    from_llm: bool            # False = repli sur le choix brut du moteur
    error: str | None = None  # motif du repli, à afficher dans l'interface

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None


# ── extraction du JSON ──────────────────────────────────────────────
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """
    Les petits modèles encadrent volontiers leur JSON de ```json … ``` ou de
    commentaires.  On tente le parsing direct, puis le premier bloc {...}.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Réponse LLM non parsable : {text[:200]!r}")


# ── appel HTTP ──────────────────────────────────────────────────────
def _post_chat(messages: list[dict], *, timeout: float, retries: int) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 220,
    }
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    url = f"{API_BASE.rstrip('/')}/chat/completions"

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            last = LLMError(f"délai dépassé ({timeout:.0f} s) sur {url}")
            log.warning("LLM : timeout (essai %d/%d)", attempt + 1, retries + 1)
        except requests.ConnectionError:
            last = LLMError(f"serveur injoignable : {url}")
            log.warning("LLM : connexion impossible (essai %d/%d)", attempt + 1, retries + 1)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body = (exc.response.text[:200] if exc.response is not None else "")
            # 4xx : inutile de réessayer, la requête est mauvaise
            last = LLMError(f"HTTP {status} — {body}")
            if isinstance(status, int) and 400 <= status < 500:
                raise last from exc
        except ValueError as exc:               # JSON de réponse invalide
            raise LLMError(f"réponse non-JSON du serveur : {exc}") from exc
        else:
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(
                    f"réponse inattendue du serveur : {str(data)[:200]}"
                ) from exc

        if attempt < retries:
            time.sleep(0.4)

    raise last or LLMError("échec inconnu")


# ── style de jeu : un peu de variété entre coups quasi équivalents ───
def _humanize_choice(top: list[MoveSuggestion]) -> MoveSuggestion:
    """
    Choisit parmi `top` (déjà trié du meilleur au pire par le moteur) —
    la plupart du temps le meilleur coup, mais avec une chance de préférer
    une alternative *presque* aussi bonne, pour un style moins mécanique.

    On ne dévie jamais d'un mat forcé, et jamais vers un coup dont l'écart
    d'évaluation dépasse `humanize_margin` pawns : ça ne joue donc jamais un
    coup objectivement plus faible, juste parfois un autre coup tout aussi
    solide.
    """
    best = top[0]
    if not config.get("humanize_moves", True) or best.is_mate or len(top) <= 1:
        return best

    margin = float(config.get("humanize_margin", 0.15))
    close = [m for m in top if not m.is_mate
             and (best.score_pawns - m.score_pawns) <= margin]
    if len(close) <= 1:
        return best

    temperature = max(1e-3, float(config.get("humanize_temperature", 0.08)))
    weights = [math.exp(-(best.score_pawns - m.score_pawns) / temperature) for m in close]
    return random.choices(close, weights=weights, k=1)[0]


# ── prompt ──────────────────────────────────────────────────────────
def _build_messages(fen: str, chosen: MoveSuggestion,
                    top: list[MoveSuggestion], lang: str) -> list[dict]:
    board = chess.Board(fen)
    side_key = "llm_side_white" if board.turn == chess.WHITE else "llm_side_black"

    others = [s for s in top if s.san != chosen.san]
    alternatives = "\n".join(f"- {s.san} ({s.score_pawns:+.2f})" for s in others) or "(none)"
    mate_note = (f", {i18n.tr(lang, 'mate_in', n=abs(chosen.mate_in))}"
                if chosen.is_mate else "")

    system = f"{i18n.tr(lang, 'llm_system')} {i18n.tr(lang, 'llm_rule')}"
    user = textwrap.dedent(f"""\
        FEN: {fen}
        Side to move: {i18n.tr(lang, side_key)}
        Board:
        {board.unicode()}

        Move already decided, explain only this one: {chosen.san} ({chosen.score_pawns:+.2f}{mate_note}) [{chosen.pv_san}]
        Other moves the engine considered (context only, do not suggest them): {alternatives}

        Reply STRICTLY with this JSON object and nothing else:
        {{
          "explanation": "<{i18n.tr(lang, 'llm_explain_hint')}>"
        }}""")

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ── API publique ────────────────────────────────────────────────────
def choose_move_and_explain(fen: str,
                            color: str | None = None,
                            lang: str = "en",
                            top_n: int = 3) -> MoveChoice:
    """
    Demande à Stockfish ses `top_n` meilleurs coups, en retient un via
    `_humanize_choice` (le meilleur la plupart du temps, parfois une
    alternative quasi équivalente pour un style moins mécanique), puis
    demande au LLM d'expliquer *ce* coup dans la langue `lang`.

    Le choix du coup ne dépend jamais du LLM : c'est toujours le moteur (plus
    l'humanisation ci-dessus) qui décide, ce qui garantit un coup solide même
    si le LLM est désactivé, injoignable ou répond n'importe quoi. Dans ce
    cas on renvoie quand même le coup choisi, avec `from_llm=False` et le
    motif dans `error` : l'interface a toujours un coup jouable *et* de quoi
    expliquer à l'utilisateur pourquoi il n'y a pas de commentaire.

    Seules les erreurs *moteur* remontent : sans coup, il n'y a rien à faire.
    """
    top = get_engine().top_moves(fen, color, n=top_n)   # peut lever EngineError
    chosen = _humanize_choice(top)

    def _without_explanation(reason: str | None) -> MoveChoice:
        return MoveChoice(
            san=chosen.san,
            explanation="",
            score_pawns=chosen.score_pawns,
            mate_in=chosen.mate_in,
            pv_san=chosen.pv_san,
            from_llm=False,
            error=reason,
        )

    if not config.get("llm_enabled", True):
        return _without_explanation(None)

    timeout = float(config.get("llm_timeout", 12.0))
    retries = int(config.get("llm_retries", 1))

    try:
        raw = _post_chat(_build_messages(fen, chosen, top, lang),
                         timeout=timeout, retries=retries)
        data = _extract_json(raw)
        return MoveChoice(
            san=chosen.san,
            explanation=str(data.get("explanation", "")).strip(),
            score_pawns=chosen.score_pawns,
            mate_in=chosen.mate_in,
            pv_san=chosen.pv_san,
            from_llm=True,
        )

    except LLMError as exc:
        log.info("LLM ignoré (%s) — coup joué sans commentaire", exc)
        return _without_explanation(str(exc))
    except Exception as exc:           # noqa: BLE001 — filet de sécurité
        log.warning("LLM : erreur inattendue (%s)", exc, exc_info=True)
        return _without_explanation(str(exc))
