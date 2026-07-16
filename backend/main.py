"""BizLens Backend - FastAPI Application"""

import asyncio
import csv
import io
import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from backend.config import (
    ALLOWED_EXTENSIONS,
    CORS_ORIGINS,
    HOST,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    PORT,
    ROOT_DIR,
    STORAGE_DIR,
)
from backend.llm_client import get_llm_client
from agents.graph import analysis_graph, AnalysisState
from mcp_client.knowledge_client import startup_knowledge_client, shutdown_knowledge_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure storage dirs exist on startup
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(
    title="BizLens API",
    description="Data Analysis Agent Backend",
    version="0.1.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lifecycle — persistent MCP Knowledge Server session
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Pre-warm the MCP Knowledge Server connection (synchronous, ~12s)."""
    logger.info("[Startup] Warming up MCP Knowledge Server...")
    await startup_knowledge_client()


@app.on_event("shutdown")
async def on_shutdown():
    """Clean up the persistent MCP session."""
    await shutdown_knowledge_client()


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "BizLens"}


# ---------------------------------------------------------------------------
# File Upload + Metadata Extraction
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV/Excel file, validate, and return metadata preview.

    Flow:
    1. Validate file extension (.xlsx / .csv)
    2. Validate file size (≤ 10 MB)
    3. Generate session_id and save file to storage/{session_id}/
    4. Parse metadata: column names, row count, file size (nrows=0 for safety)
    5. Return preview info to frontend
    """

    # --- 1. Extension validation ---
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'。仅支持 {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # --- 2. Size validation ---
    content = await file.read()
    file_size = len(content)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大 ({file_size / 1024 / 1024:.1f}MB)。最大允许 {MAX_FILE_SIZE_MB}MB",
        )

    # --- 3. Save file ---
    session_id = uuid.uuid4().hex[:12]
    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / filename
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"File saved: {file_path} (session={session_id}, size={file_size})")

    # --- 4. Parse metadata (nrows=0 — only headers, no data — safe outside sandbox) ---
    try:
        if ext == ".csv":
            # Use csv module first for row count (memory efficient)
            text_content = content.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text_content))
            rows = list(reader)
            row_count = max(len(rows) - 1, 0)  # subtract header
            columns = rows[0] if rows else []
            # Also use pandas for dtypes preview
            df = pd.read_csv(io.StringIO(text_content), nrows=5)
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            sample_rows = min(5, row_count)
        else:  # .xlsx
            df_header = pd.read_excel(io.BytesIO(content), nrows=0)
            columns = list(df_header.columns)
            # Full read for row count (xlsx can't easily count rows without reading)
            df_full = pd.read_excel(io.BytesIO(content))
            row_count = len(df_full)
            dtypes = {col: str(dtype) for col, dtype in df_full.dtypes.items()}
            # Sample data preview
            df_sample = df_full.head(5)
            sample_rows = min(5, row_count)

        # Convert to Python dicts, then aggressively clean NaN → None
        # (Pandas numeric columns cannot hold None — numpy converts it back to NaN)
        sample_data = df.head(5).to_dict(orient="records") if ext != ".xlsx" else df_sample.to_dict(orient="records")

        preview = {
            "session_id": session_id,
            "filename": filename,
            "columns": [str(c) if c is not None else "" for c in columns],
            "column_count": len(columns),
            "row_count": int(row_count),
            "file_size_bytes": file_size,
            "file_size_display": _format_file_size(file_size),
            "dtypes": {str(k): str(v) for k, v in dtypes.items()},
            "sample_data": _clean_nan(sample_data),
            "sample_rows": min(5, int(row_count)),
        }

        logger.info(f"Metadata extracted: {len(columns)} cols × {row_count} rows")
        json_str = json.dumps(_clean_nan(preview), ensure_ascii=False, allow_nan=False)
        return Response(content=json_str, media_type="application/json")

    except Exception as e:
        # Clean up saved file on parse failure
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Failed to parse file: {e}")
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")


# ---------------------------------------------------------------------------
# Chat Test Endpoint (verifies DeepSeek API works)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512


@app.post("/api/chat")
async def chat_test(request: ChatRequest):
    """Test endpoint to verify DeepSeek API connectivity."""
    if not request.message:
        raise HTTPException(status_code=400, detail="message is required")

    client = get_llm_client()
    try:
        response = await client.chat(
            messages=[{"role": "user", "content": request.message}],
            max_tokens=request.max_tokens,
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat test failed: {e}")
        raise HTTPException(status_code=500, detail=f"API 调用失败: {str(e)}")


# ---------------------------------------------------------------------------
# Analysis Endpoint — LangGraph workflow with SSE progress streaming
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str
    session_id: str


# Human-readable labels for each graph node
NODE_LABELS = {
    "planner": "🔍 正在分析问题结构...",
    "code_interpreter": "🐍 正在执行数据分析代码...",
    "visualization": "📊 正在生成图表...",
    "insight": "📝 正在生成分析报告...",
}

# Node order for determining completion
NODE_ORDER = ["planner", "code_interpreter", "visualization", "insight"]


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_analysis_stream(
    query: str, session_id: str
) -> AsyncGenerator[str, None]:
    """Execute the LangGraph workflow and yield SSE events per node.

    Yields:
        SSE formatted strings for each node transition (running → done).
    """
    initial_state: AnalysisState = {
        "query": query,
        "session_id": session_id,
        "steps": [],
        "expected_output": "",
        "code": "",
        "exec_result": {},
        "chart_json": None,
        "report": "",
        "error": None,
        "current_step": "",
    }

    completed_nodes: set[str] = set()

    try:
        async for chunk in analysis_graph.astream(
            initial_state,
            stream_mode="updates",
        ):
            # chunk is dict: {node_name: state_update_dict}
            for node_name, node_state in chunk.items():
                if node_name in completed_nodes:
                    continue

                label = NODE_LABELS.get(node_name, node_name)

                # Emit "running" event
                yield _sse_event("step_update", {
                    "step": node_name,
                    "status": "running",
                    "label": label,
                })

                # Check if this node set an error
                if node_state.get("error"):
                    yield _sse_event("step_update", {
                        "step": node_name,
                        "status": "error",
                        "label": label,
                        "error": node_state["error"],
                    })
                    yield _sse_event("done", {"status": "error", "error": node_state["error"]})
                    return

                # Emit "done" event with partial output
                done_data: dict[str, Any] = {
                    "step": node_name,
                    "status": "done",
                    "label": label.replace("正在", "").replace("...", "").strip(),
                }

                if node_name == "planner":
                    done_data["steps"] = node_state.get("steps", [])
                    done_data["expected_output"] = node_state.get("expected_output", "")
                elif node_name == "visualization":
                    done_data["chart_json"] = node_state.get("chart_json")
                elif node_name == "insight":
                    done_data["report"] = node_state.get("report", "")

                yield _sse_event("step_update", done_data)
                completed_nodes.add(node_name)

        # All nodes completed successfully
        yield _sse_event("done", {"status": "completed"})

    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        yield _sse_event("done", {"status": "error", "error": str(e)})


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Run the full 4-Agent analysis workflow with SSE progress streaming.

    The frontend receives step_update events for each node transition,
    and a final done event when the entire workflow completes.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    logger.info(f"[Analyze] Starting workflow: query={request.query[:80]}... session={request.session_id}")

    return StreamingResponse(
        _run_analysis_stream(request.query, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Knowledge Base Management — proxy MCP tool calls via REST API
# ---------------------------------------------------------------------------

KB_UPLOAD_DIR = ROOT_DIR / "storage" / "_knowledge_uploads"
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/knowledge/status")
async def knowledge_status():
    """Return MCP Knowledge Server connection status.

    Frontend uses this to show whether the knowledge base is available,
    and if it's still initializing (loading the embedding model).
    """
    from mcp_client.knowledge_client import _get_persistent

    ps = _get_persistent()
    # Try a quick connect check — _ensure_connected is lazy and cached
    connected = await ps._ensure_connected()
    return {
        "connected": connected,
        "status": "ready" if connected else ("connecting" if ps._available is None else "unavailable"),
        "detail": (
            "MCP Knowledge Server is ready"
            if connected
            else (
                "MCP server is starting up (model loading, may take 30-60s)..."
                if ps._available is None
                else "MCP Knowledge Server is not available"
            )
        ),
    }


@app.get("/api/knowledge/documents")
async def list_knowledge_documents(category: str | None = None):
    """List all documents in the knowledge base, optionally filtered by category."""
    from mcp_client.knowledge_client import list_knowledge_documents as list_docs

    try:
        docs = await list_docs(category)
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Knowledge list failed: {e}")
        raise HTTPException(status_code=500, detail=f"知识库查询失败: {str(e)}")


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics (file count, chunk count, storage size)."""
    from mcp_client.knowledge_client import get_knowledge_stats as get_stats

    try:
        stats = await get_stats()
        return stats
    except Exception as e:
        logger.error(f"Knowledge stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"知识库统计失败: {str(e)}")


@app.post("/api/knowledge/upload")
async def upload_knowledge_document(file: UploadFile = File(...), category: str = "general"):
    """Upload a document (PDF/TXT/MD) to the knowledge base.

    The file is saved temporarily, then forwarded to MCP Knowledge Server
    for chunking, embedding, and storage in ChromaDB.
    """
    from mcp_client.knowledge_client import add_knowledge_document

    # --- Validate ---
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".markdown"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'。仅支持 {', '.join(allowed)}",
        )

    # Max 20MB for knowledge docs
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大，最大允许 20MB")

    # --- Save to temp ---
    tmp_path = KB_UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    logger.info(f"Knowledge upload saved: {tmp_path} ({len(content)} bytes)")

    # --- Forward to MCP Server ---
    try:
        result = await add_knowledge_document(str(tmp_path), category)
        logger.info(f"Knowledge document added: {result}")
        return result
    except Exception as e:
        logger.error(f"Knowledge add_document failed: {e}")
        raise HTTPException(status_code=500, detail=f"知识库导入失败: {str(e)}")
    finally:
        # Clean up temp file (MCP Server has already read it)
        if tmp_path.exists():
            tmp_path.unlink()


@app.delete("/api/knowledge/{doc_id}")
async def delete_knowledge_document(doc_id: str):
    """Remove a document from the knowledge base by its ID."""
    from mcp_client.knowledge_client import remove_knowledge_document

    try:
        result = await remove_knowledge_document(doc_id)
        return result
    except Exception as e:
        logger.error(f"Knowledge remove failed: {e}")
        raise HTTPException(status_code=500, detail=f"知识库删除失败: {str(e)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/NaT with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    # pandas NaT (Not a Time) also isn't JSON-safe
    if obj is pd.NaT:
        return None
    return obj


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f}MB"


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
