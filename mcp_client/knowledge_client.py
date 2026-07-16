"""MCP Knowledge Client — connect to Knowledge Server via persistent subprocess.

Architecture:
    - Spawns python -m mcp_knowledge_agent as a persistent subprocess
    - Talks JSON-RPC over stdin/stdout directly (no mcp.client.stdio dependency)
    - This avoids anyio task-scoping bugs on Python 3.13 + Windows + ProactorEventLoop
    - The subprocess is kept alive so the 2.5GB embedding model stays warm

面试说法:
    "MCP Knowledge Server 是独立的标准化知识检索服务（项目二），
     BizLens 通过 MCP 协议消费它——Insight Agent 生成报告时注入知识库上下文。
     这是三项目能力体系的关键交叉点：ARC（知识系统）→ MCP Server（协议抽象）
     → BizLens（数据系统+协议消费）。"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server configuration
# ---------------------------------------------------------------------------

MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", sys.executable)
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", "-m,mcp_knowledge_agent").split(",")

# Env for the subprocess: offline mode for HuggingFace + API key passthrough
MCP_SERVER_ENV = {
    **os.environ,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
# The MCP server uses openai.AsyncOpenAI which checks OPENAI_API_KEY env var.
# Ensure DeepSeek API key is available under both names.
if os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    MCP_SERVER_ENV["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]


# ---------------------------------------------------------------------------
# Low-level JSON-RPC over subprocess pipes
# ---------------------------------------------------------------------------

class _JsonRpcError(RuntimeError):
    """Error returned by the MCP server."""


async def _read_exactly(stream: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes from an asyncio stream."""
    data = b""
    while len(data) < n:
        chunk = await stream.read(n - len(data))
        if not chunk:
            raise EOFError("MCP server closed stdout unexpectedly")
        data += chunk
    return data


async def _read_line(stream: asyncio.StreamReader) -> str:
    """Read a line from the stream (JSON-RPC messages are newline-delimited)."""
    line = await stream.readline()
    if not line:
        raise EOFError("MCP server stdout closed")
    return line.decode("utf-8").strip()


async def _send_request(writer: asyncio.StreamWriter, request: dict) -> None:
    """Send a JSON-RPC request to the MCP server."""
    payload = json.dumps(request, ensure_ascii=False) + "\n"
    writer.write(payload.encode("utf-8"))
    await writer.drain()


async def _recv_response(reader: asyncio.StreamReader) -> dict:
    """Read and parse a JSON-RPC response."""
    line = await _read_line(reader)
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.warning("[MCP] Unparseable response: %s", line[:200])
        raise


# ---------------------------------------------------------------------------
# Persistent session singleton
# ---------------------------------------------------------------------------

class _PersistentMCPSession:
    """Holds a long-lived subprocess connection to the MCP Knowledge Server.

    The subprocess is spawned once on first use and stays alive so the
    embedding model (loaded at server startup, ~10s) remains warm.
    """

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._available: bool | None = None
        self._lock = asyncio.Lock()       # protects _startup / shutdown
        self._call_lock = asyncio.Lock()  # serializes all call_tool requests
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _startup(self) -> bool:
        """Launch the MCP server subprocess and perform JSON-RPC initialization."""
        if self._available is not None:
            return self._available

        async with self._lock:
            if self._available is not None:  # double-check
                return self._available

            try:
                logger.info("[MCP] Spawning: %s %s", MCP_SERVER_COMMAND, MCP_SERVER_ARGS)
                # Spawn the MCP server subprocess
                self._proc = await asyncio.create_subprocess_exec(
                    MCP_SERVER_COMMAND,
                    *MCP_SERVER_ARGS,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=MCP_SERVER_ENV,
                )
                logger.info("[MCP] Subprocess PID=%s stdin=%s", self._proc.pid, type(self._proc.stdin).__name__)
                self._reader = self._proc.stdout
                self._writer = self._proc.stdin

                # JSON-RPC initialization
                rid = self._next_id()
                await _send_request(self._writer, {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": rid,
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "BizLens", "version": "0.1"},
                    },
                })

                # Read init response (model pre-load takes ~10s, so expect a wait)
                try:
                    resp = await asyncio.wait_for(_recv_response(self._reader), timeout=120)
                except asyncio.TimeoutError:
                    logger.error("[MCP] Initialize timed out after 120s")
                    self._available = False
                    return False

                if "error" in resp:
                    logger.error("[MCP] Initialize error: %s", resp["error"])
                    self._available = False
                    return False

                # Send initialized notification
                await _send_request(self._writer, {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                })

                logger.info("[MCP] Persistent session established (subprocess PID=%s)", self._proc.pid)
                self._available = True
                return True

            except Exception as e:
                logger.warning("[MCP] Startup failed: %s", str(e)[:200])
                self._available = False
                self._reader = None
                self._writer = None
                if self._proc:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                    self._proc = None
                return False

    async def _ensure_connected(self) -> bool:
        if self._available is True and self._proc is not None and self._proc.returncode is None:
            return True
        if self._available is False:
            return False

        # Process died or never started — reset and retry
        self._available = None
        self._reader = None
        self._writer = None
        self._proc = None
        return await self._startup()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool on the persistent session.

        Serialized by _call_lock — only one JSON-RPC request in flight at a time,
        preventing response mixing under concurrent HTTP requests.
        """
        if not await self._ensure_connected():
            raise RuntimeError("MCP Knowledge Server is not available")

        async with self._call_lock:
            rid = self._next_id()
            try:
                await _send_request(self._writer, {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": rid,
                    "params": {"name": tool_name, "arguments": arguments},
                })

                resp = await asyncio.wait_for(_recv_response(self._reader), timeout=300)

                # ── Validate response ID matches request ID ──
                resp_id = resp.get("id")
                if resp_id is not None and resp_id != rid:
                    logger.error(
                        "[MCP] Response ID mismatch: expected %s, got %s. "
                        "This indicates a protocol desync — resetting connection.",
                        rid, resp_id,
                    )
                    # Drain any stale data and reset
                    self._available = None
                    self._proc = None
                    self._reader = None
                    self._writer = None
                    raise RuntimeError(
                        f"MCP response ID mismatch (expected {rid}, got {resp_id})"
                    )

            except (EOFError, ConnectionError, OSError) as e:
                # Subprocess died — reset for next attempt
                logger.warning("[MCP] Connection lost during '%s': %s", tool_name, e)
                self._available = None
                self._proc = None
                self._reader = None
                self._writer = None
                raise RuntimeError(f"MCP connection lost: {e}") from e

            except asyncio.TimeoutError:
                logger.error("[MCP] Tool '%s' timed out after 300s", tool_name)
                raise RuntimeError(f"MCP tool '{tool_name}' timed out")

            if "error" in resp:
                err = resp["error"]
                raise _JsonRpcError(f"{err.get('code', '?')}: {err.get('message', str(err))}")

            # Extract text content from MCP ContentBlock format
            result = resp.get("result", {})
            content_blocks = result.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            raw_text = "\n".join(text_parts)
            try:
                return json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                # FastMCP splits list returns into multiple text blocks (NDJSON).
                # Each block is an individual JSON object — parse separately.
                if len(text_parts) > 1:
                    items = []
                    for part in text_parts:
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            items.append(json.loads(part))
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if items:
                        return items
                return raw_text if raw_text else result

    async def shutdown(self) -> None:
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                try:
                    self._proc.terminate()
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except Exception:
                    self._proc.kill()
            self._proc = None
            self._reader = None
            self._writer = None
            self._available = None


# Global singleton
_persistent: _PersistentMCPSession | None = None


def _get_persistent() -> _PersistentMCPSession:
    global _persistent
    if _persistent is None:
        _persistent = _PersistentMCPSession()
    return _persistent


# ---------------------------------------------------------------------------
# Client class (public API)
# ---------------------------------------------------------------------------

class MCPKnowledgeClient:
    """Thin wrapper that delegates to the persistent MCP session.

    Usage:
        async with MCPKnowledgeClient() as client:
            result = await client.answer_with_citation("Q3营收增长驱动", top_k=3)
    """

    def __init__(self, **kwargs):
        self._ps = _get_persistent()

    async def _connect(self) -> bool:
        return await self._ps._ensure_connected()

    async def _disconnect(self) -> None:
        pass  # persistent

    async def __aenter__(self) -> "MCPKnowledgeClient":
        await self._connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def answer_with_citation(self, query: str, top_k: int = 3) -> dict[str, Any]:
        try:
            result = await self._ps.call_tool("answer_with_citation", {"question": query, "top_k": top_k})
            if isinstance(result, dict):
                return {"available": True, "answer": result.get("answer", str(result)), "citations": result.get("sources", [])}
            return {"available": True, "answer": str(result), "citations": []}
        except Exception as e:
            logger.warning("[MCP] answer_with_citation failed: %s", str(e)[:150])
            return {"available": False, "answer": None, "citations": []}

    async def list_documents(self, category: str | None = None) -> list[dict[str, Any]]:
        try:
            args = {}
            if category is not None:
                args["category"] = category
            result = await self._ps.call_tool("list_documents", args)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "documents" in result:
                return result["documents"]
            return [result] if result else []
        except Exception as e:
            logger.warning("[MCP] list_documents failed: %s", str(e)[:150])
            return []

    async def get_index_stats(self) -> dict[str, Any]:
        try:
            result = await self._ps.call_tool("get_index_stats", {})
            if isinstance(result, dict):
                return result
            return {"available": True, "raw": str(result)}
        except Exception as e:
            logger.warning("[MCP] get_index_stats failed: %s", str(e)[:150])
            return {"available": False, "error": str(e)}

    async def add_document(self, file_path: str, category: str = "general") -> dict[str, Any]:
        try:
            result = await self._ps.call_tool("add_document", {"file_path": file_path, "category": category})
            if isinstance(result, dict):
                return {"success": True, **result}
            return {"success": True, "raw": str(result)}
        except Exception as e:
            msg = str(e)[:200]
            logger.warning("[MCP] add_document failed: %s", msg)
            return {"success": False, "error": msg}

    async def remove_document(self, doc_id: str) -> dict[str, Any]:
        try:
            result = await self._ps.call_tool("remove_document", {"doc_id": doc_id})
            if isinstance(result, dict):
                return {"success": True, **result}
            return {"success": True, "raw": str(result)}
        except Exception as e:
            msg = str(e)[:200]
            logger.warning("[MCP] remove_document failed: %s", msg)
            return {"success": False, "error": msg}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def query_knowledge(query: str, top_k: int = 3, **kwargs) -> dict[str, Any]:
    async with MCPKnowledgeClient() as client:
        return await client.answer_with_citation(query=query, top_k=top_k)


async def add_knowledge_document(file_path: str, category: str = "general") -> dict[str, Any]:
    async with MCPKnowledgeClient() as client:
        return await client.add_document(file_path, category)


async def remove_knowledge_document(doc_id: str) -> dict[str, Any]:
    async with MCPKnowledgeClient() as client:
        return await client.remove_document(doc_id)


async def list_knowledge_documents(category: str | None = None) -> list[dict[str, Any]]:
    async with MCPKnowledgeClient() as client:
        return await client.list_documents(category)


async def get_knowledge_stats() -> dict[str, Any]:
    async with MCPKnowledgeClient() as client:
        return await client.get_index_stats()


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

async def startup_knowledge_client():
    """Pre-warm the MCP connection (launch subprocess, load model)."""
    ps = _get_persistent()
    logger.info("[MCP] Starting persistent Knowledge Server (model pre-load ~10s)...")
    ok = await ps._ensure_connected()
    if ok:
        logger.info("[MCP] Knowledge Server ready.")
    else:
        logger.warning("[MCP] Knowledge Server not available.")


async def shutdown_knowledge_client():
    global _persistent
    if _persistent is not None:
        await _persistent.shutdown()
        _persistent = None
