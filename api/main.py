"""
api/main.py

FastAPI application. Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from api.pipeline_v1 import run, stream
from api.retriever import _get_collection, _get_model, retrieve
from api.schemas import QueryRequest, QueryResponse

app = FastAPI(title="Yelp NOLA Conversational Assistant")

_index_html: str = ""


@app.on_event("startup")
def warmup() -> None:
    """Eager-load the embedding model and ChromaDB index at server startup.

    Without this, the first user query bears the full cold-start cost (~12s):
    embedding model initialisation + HNSW index loading from disk. After warmup,
    every query hits warm state.
    """
    global _index_html
    _index_html = (Path(__file__).parent.parent / "static" / "index.html").read_text()
    _get_model()
    _get_collection()
    retrieve("warmup")  # forces HNSW index into memory


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(_index_html)


@app.post("/api/v1/query", response_model=QueryResponse)
def query_v1(request: QueryRequest) -> QueryResponse:
    try:
        return run(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/v1/query/stream")
def query_v1_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        stream(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
