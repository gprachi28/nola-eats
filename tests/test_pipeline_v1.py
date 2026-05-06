"""
Tests for api/pipeline_v1.py

All external calls (query planner, SQL filter, retriever, synthesizer, SQLite)
are mocked — tests verify orchestration logic only.
"""
from unittest.mock import MagicMock, patch

import pytest

import json

from api.pipeline_v1 import run, stream, _fetch_business_meta
from api.schemas import BusinessResult, QueryPlan, QueryResponse


# ── shared mocks ───────────────────────────────────────────────────────────────


def _make_plan(**kwargs):
    return QueryPlan(
        intent=kwargs.get("intent", "find_businesses"),
        sql_filters=kwargs.get("sql_filters", {"noise_level": "loud"}),
        semantic_query=kwargs.get("semantic_query", "loud bachelor party"),
    )


def _make_snippet(business_id="biz_a", text="Great spot.", distance=0.1):
    return {"business_id": business_id, "text": text, "stars": 4.5, "distance": distance}


def _make_business_result(business_id="biz_a"):
    return BusinessResult(
        business_id=business_id,
        name="Bayou Jazz",
        stars=4.5,
        price_range=2,
        evidence=["Great spot."],
    )


# ── happy path ─────────────────────────────────────────────────────────────────


def test_run_returns_query_response():
    plan = _make_plan()
    snippets = [_make_snippet()]
    biz_result = _make_business_result()

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]),
        patch("api.pipeline_v1.retrieve", return_value=snippets),
        patch("api.pipeline_v1._fetch_business_meta", return_value={"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}),
        patch("api.pipeline_v1.synthesize", return_value=("Great choice!", [biz_result])),
    ):
        result = run("bachelor party spot")

    assert isinstance(result, QueryResponse)
    assert result.answer == "Great choice!"
    assert len(result.businesses) == 1
    assert result.query_plan == plan
    assert result.latency_ms >= 0


def test_run_passes_sql_filters_to_filter_businesses():
    plan = _make_plan(sql_filters={"noise_level": "loud", "good_for_groups": True})

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]) as mock_filter,
        patch("api.pipeline_v1.retrieve", return_value=[]),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("No results.", [])),
    ):
        run("question")

    mock_filter.assert_called_once_with({"noise_level": "loud", "good_for_groups": True})


def test_run_passes_candidate_ids_to_retrieve():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a", "biz_b"]),
        patch("api.pipeline_v1.retrieve", return_value=[]) as mock_retrieve,
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    mock_retrieve.assert_called_once_with("loud bachelor party", ["biz_a", "biz_b"])


def test_run_passes_none_to_retrieve_on_semantic_only_fallback():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=None),
        patch("api.pipeline_v1.retrieve", return_value=[]) as mock_retrieve,
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    mock_retrieve.assert_called_once_with("loud bachelor party", None)


def test_run_fetches_meta_only_for_retrieved_businesses():
    plan = _make_plan()
    snippets = [
        _make_snippet("biz_a"),
        _make_snippet("biz_b"),
        _make_snippet("biz_a"),  # duplicate — should only fetch once
    ]

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a", "biz_b", "biz_c"]),
        patch("api.pipeline_v1.retrieve", return_value=snippets),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}) as mock_meta,
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    fetched_ids = set(mock_meta.call_args[0][0])
    assert fetched_ids == {"biz_a", "biz_b"}  # biz_c had no snippets


def test_run_cache_hit_defaults_false():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=None),
        patch("api.pipeline_v1.retrieve", return_value=[]),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        result = run("question")

    assert result.cache_hit is False


# ── _fetch_business_meta ───────────────────────────────────────────────────────


def test_fetch_business_meta_empty_list_returns_empty():
    assert _fetch_business_meta([]) == {}


def test_fetch_business_meta_queries_sqlite(tmp_path):
    import sqlite3
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE businesses (
        business_id TEXT, name TEXT, stars REAL, price_range INTEGER,
        review_count INTEGER, noise_level TEXT, alcohol TEXT, attire TEXT,
        wifi TEXT, smoking TEXT, good_for_groups INTEGER, takes_reservations INTEGER,
        outdoor_seating INTEGER, good_for_kids INTEGER, good_for_dancing INTEGER,
        happy_hour INTEGER, has_tv INTEGER, caters INTEGER, wheelchair_accessible INTEGER,
        dogs_allowed INTEGER, byob INTEGER, corkage INTEGER,
        ambience TEXT, good_for_meal TEXT, music TEXT, parking TEXT,
        categories TEXT, latitude REAL, longitude REAL
    )""")
    conn.execute("INSERT INTO businesses (business_id, name, stars, price_range, categories) "
                 "VALUES ('biz_a', 'Bayou Jazz', 4.5, 2, 'Restaurants')")
    conn.commit()
    conn.close()

    with patch("api.pipeline_v1.settings") as mock_settings:
        mock_settings.sqlite_path = db
        result = _fetch_business_meta(["biz_a"])

    assert result == {"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}


# ── stream() ───────────────────────────────────────────────────────────────────


def _parse_sse_events(events: list[str]) -> list[tuple[str, dict]]:
    """Parse raw SSE strings into (event_name, data) pairs."""
    parsed = []
    for e in events:
        lines = e.strip().split("\n")
        name = lines[0].split(": ", 1)[1]
        data = json.loads(lines[1].split(": ", 1)[1])
        parsed.append((name, data))
    return parsed


def test_stream_emits_correct_event_sequence():
    plan = _make_plan()
    snippet = _make_snippet()
    biz_result = _make_business_result()
    meta = {"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]),
        patch("api.pipeline_v1.retrieve", return_value=[snippet]),
        patch("api.pipeline_v1._fetch_business_meta", return_value=meta),
        patch("api.pipeline_v1.synthesize_stream", return_value=(iter(["Hello", " world"]), [biz_result])),
    ):
        events = _parse_sse_events(list(stream("bachelor party spot")))

    names = [e[0] for e in events]
    assert names == ["planning", "candidates", "token", "token", "done"]


def test_stream_planning_event_carries_query_plan():
    plan = _make_plan(intent="find_businesses", sql_filters={"noise_level": "loud"}, semantic_query="lively")
    meta = {"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]),
        patch("api.pipeline_v1.retrieve", return_value=[_make_snippet()]),
        patch("api.pipeline_v1._fetch_business_meta", return_value=meta),
        patch("api.pipeline_v1.synthesize_stream", return_value=(iter([]), [_make_business_result()])),
    ):
        events = _parse_sse_events(list(stream("test")))

    planning_data = events[0][1]
    assert planning_data["intent"] == "find_businesses"
    assert planning_data["sql_filters"] == {"noise_level": "loud"}


def test_stream_candidates_event_carries_business_names():
    meta = {
        "biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2},
        "biz_b": {"name": "Café Du Monde", "stars": 4.2, "price_range": 1},
    }
    snippets = [_make_snippet("biz_a"), _make_snippet("biz_b")]

    with (
        patch("api.pipeline_v1.plan_query", return_value=_make_plan()),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a", "biz_b"]),
        patch("api.pipeline_v1.retrieve", return_value=snippets),
        patch("api.pipeline_v1._fetch_business_meta", return_value=meta),
        patch("api.pipeline_v1.synthesize_stream", return_value=(iter([]), [])),
    ):
        events = _parse_sse_events(list(stream("test")))

    candidates_data = events[1][1]
    assert candidates_data["count"] == 2
    assert set(candidates_data["businesses"]) == {"Bayou Jazz", "Café Du Monde"}


def test_stream_yields_generic_error_event_on_exception():
    with patch("api.pipeline_v1.plan_query", side_effect=ValueError("internal details here")):
        events = _parse_sse_events(list(stream("test question")))

    assert len(events) == 1
    assert events[0][0] == "error"
    assert events[0][1]["message"] == "An error occurred. Please try again."
    # Internal details must not leak to the client
    assert "internal details here" not in events[0][1]["message"]
