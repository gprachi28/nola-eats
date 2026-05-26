"""
Tests for api/reranker.py

Mocks CrossEncoder — does NOT download or run the real model.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.reranker import rerank


def _snippets():
    return [
        {"business_id": "biz_a", "text": "Decent place, nothing special.", "stars": 3.5, "distance": 0.1},
        {"business_id": "biz_b", "text": "Amazing jazz brunch, loud and lively!", "stars": 5.0, "distance": 0.2},
        {"business_id": "biz_c", "text": "Good for groups, great atmosphere.", "stars": 4.0, "distance": 0.3},
    ]


def _mock_model(scores):
    m = MagicMock()
    m.predict.return_value = scores
    return m


def test_rerank_sorts_by_score_descending():
    snippets = _snippets()
    mock_ce = _mock_model([0.2, 0.9, 0.6])

    with patch("api.reranker._get_model", return_value=mock_ce), \
         patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = True
        result = rerank("jazz brunch spot", snippets)

    assert result[0]["business_id"] == "biz_b"
    assert result[1]["business_id"] == "biz_c"
    assert result[2]["business_id"] == "biz_a"


def test_rerank_adds_rerank_score_key():
    snippets = _snippets()
    mock_ce = _mock_model([0.2, 0.9, 0.6])

    with patch("api.reranker._get_model", return_value=mock_ce), \
         patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = True
        result = rerank("jazz brunch spot", snippets)

    assert all("rerank_score" in s for s in result)
    assert result[0]["rerank_score"] == pytest.approx(0.9)


def test_rerank_passthrough_when_disabled():
    snippets = _snippets()

    with patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = False
        result = rerank("jazz brunch spot", snippets)

    assert result is snippets


def test_rerank_passthrough_on_empty_snippets():
    with patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = True
        result = rerank("jazz brunch spot", [])

    assert result == []


def test_rerank_groups_by_business_for_diversity():
    # biz_a has 2 snippets scored 0.9 and 0.7 (best business, max=0.9)
    # biz_b has 1 snippet scored 0.8 (second business, max=0.8)
    # biz_a should rank first despite biz_b having a higher score than biz_a's second snippet
    snippets = [
        {"business_id": "biz_a", "text": "snippet a1", "stars": 4.5},
        {"business_id": "biz_b", "text": "snippet b1", "stars": 4.0},
        {"business_id": "biz_a", "text": "snippet a2", "stars": 4.5},
    ]
    mock_ce = _mock_model([0.9, 0.8, 0.7])

    with patch("api.reranker._get_model", return_value=mock_ce), \
         patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = True
        result = rerank("jazz brunch spot", snippets)

    assert result[0]["business_id"] == "biz_a"
    assert result[0]["rerank_score"] == pytest.approx(0.9)
    assert result[1]["business_id"] == "biz_a"
    assert result[1]["rerank_score"] == pytest.approx(0.7)
    assert result[2]["business_id"] == "biz_b"
    assert result[2]["rerank_score"] == pytest.approx(0.8)


def test_rerank_passes_correct_pairs_to_model():
    snippets = _snippets()
    mock_ce = _mock_model([0.5, 0.7, 0.6])

    with patch("api.reranker._get_model", return_value=mock_ce), \
         patch("api.reranker.settings") as mock_cfg:
        mock_cfg.rerank_enabled = True
        rerank("jazz brunch spot", snippets)

    call_args = mock_ce.predict.call_args[0][0]
    assert call_args == [
        ("jazz brunch spot", "Decent place, nothing special."),
        ("jazz brunch spot", "Amazing jazz brunch, loud and lively!"),
        ("jazz brunch spot", "Good for groups, great atmosphere."),
    ]