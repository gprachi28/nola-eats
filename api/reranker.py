"""
api/reranker.py

Cross-encoder re-ranking: score (query, snippet) pairs jointly and
re-order snippets by relevance before passing to the Synthesizer.

Public API:
    rerank(query, snippets) -> list[dict]
"""
import threading

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

    Returns snippets unchanged if rerank_enabled is False or list is empty.
    Each returned dict gains a "rerank_score" float key.
    """
    if not snippets or not settings.rerank_enabled:
        return snippets

    model = _get_model()
    pairs = [(query, s["text"]) for s in snippets]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, snippets),
        key=lambda x: x[0],
        reverse=True,
    )
    return [{**s, "rerank_score": float(score)} for score, s in ranked]
