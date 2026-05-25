"""
api/reranker.py

Cross-encoder re-ranking: score (query, snippet) pairs jointly and
re-order snippets by relevance before passing to the Synthesizer.

Ranking is business-aware: snippets are grouped by business_id and
businesses are ranked by their best snippet score. This ensures the
Synthesizer sees diverse candidates rather than many snippets from
one high-scoring business — important for multi-turn flows where the
user may reject the first suggestion and ask for alternatives.

Public API:
    rerank(query, snippets) -> list[dict]
"""
import threading
from collections import defaultdict

from sentence_transformers import CrossEncoder

from config import settings

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(settings.rerank_model)
    return _model


def rerank(query: str, snippets: list[dict]) -> list[dict]:
    """
    Re-rank snippets by joint query–document relevance using a cross-encoder.

    Businesses are ranked by their best snippet score so the Synthesizer
    receives diverse candidates. Within each business, snippets are ordered
    by score descending.

    Returns snippets unchanged if rerank_enabled is False or list is empty.
    Each returned dict gains a "rerank_score" float key.
    """
    if not snippets or not settings.rerank_enabled:
        return snippets

    model = _get_model()
    pairs = [(query, s["text"]) for s in snippets]
    scores = model.predict(pairs)

    scored = [{**s, "rerank_score": float(score)} for s, score in zip(snippets, scores)]

    by_biz: dict[str, list] = defaultdict(list)
    for s in scored:
        by_biz[s["business_id"]].append(s)

    ranked_biz = sorted(
        by_biz.values(),
        key=lambda snips: max(s["rerank_score"] for s in snips),
        reverse=True,
    )

    result = []
    for snips in ranked_biz:
        result.extend(sorted(snips, key=lambda s: s["rerank_score"], reverse=True))
    return result
