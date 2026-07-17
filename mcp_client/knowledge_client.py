"""MCP Knowledge Client — connect to Knowledge Server via HTTP (Streamable HTTP).

Architecture:
    - Connects to an independently-deployed MCP Knowledge Server via HTTP
    - Uses MCP SDK's streamable_http_client transport + ClientSession
    - Persistent HTTP session: one initialize handshake, reused across requests
    - No subprocess management needed — the MCP Server is deployed separately
      (Docker container, systemd, or manual `python -m mcp_knowledge_agent`)

Transient errors (connection drop, server restart) trigger automatic reconnect
+ single retry on the next call_tool invocation.

Design note:
    MCP Knowledge Server provides standardized knowledge retrieval via MCP protocol.
    BizLens consumes it — Insight Agent injects knowledge context into reports.
    Cross-project architecture: ARC (knowledge) → MCP Server (protocol) → BizLens (data).
    The MCP Server is independently deployable so both ARC and BizLens can share one
    instance — one embedding model, one ChromaDB, zero duplication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server configuration
# ---------------------------------------------------------------------------

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")


# ---------------------------------------------------------------------------
# Persistent HTTP session
# ---------------------------------------------------------------------------

class _PersistentMCPSession:
    """Holds a long-lived HTTP session to the MCP Knowledge Server.

    Uses MCP SDK's streamable_http_client + ClientSession.  The SDK handles
    connection pooling, message framing, and the initialize handshake —
    no hand-rolled pipe I/O, no subprocess lifecycle.
    """

    def __init__(self):
        self._url = MCP_SERVER_URL
        self._http_ctx = None       # async ctx-mgr returned by streamable_http_client
        self._session: ClientSession | None = None
        self._available: bool | None = None
        self._lock = asyncio.Lock()       # protects connect / disconnect
        self._call_lock = asyncio.Lock()  # serializes call_tool requests

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> bool:
        """Establish HTTP connection + MCP initialize handshake."""
        try:
            logger.info("[MCP] Connecting to %s ...", self._url)
            self._http_ctx = streamable_http_client(self._url)
            read, write, _ = await self._http_ctx.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()  # start background task group (required for send_request)
            await self._session.initialize()
            logger.info("[MCP] HTTP session established — Knowledge Server ready")
            return True
        except Exception as e:
            logger.warning("[MCP] Connect failed: %s", str(e)[:200])
            self._http_ctx = None
            self._session = None
            return False

    async def _ensure_connected(self) -> bool:
        """Return True if the HTTP session is alive; reconnect if needed."""
        if self._available is True and self._session is not None:
            return True

        async with self._lock:
            if self._available is True and self._session is not None:
                return True
            ok = await self._connect()
            self._available = ok
            return ok

    # ------------------------------------------------------------------
    # Tool call
    # ------------------------------------------------------------------

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool over the persistent HTTP session.

        Serialized by _call_lock — one request in flight at a time.
        On transient failure (connection drop, server restart), reconnects
        and retries once before raising.
        """
        if not await self._ensure_connected():
            raise RuntimeError("MCP Knowledge Server is not available")

        async with self._call_lock:
            for attempt in (1, 2):
                try:
                    result = await self._session.call_tool(tool_name, arguments)
                    break  # success
                except Exception as e:
                    logger.warning(
                        "[MCP] Tool '%s' attempt %d failed: %s",
                        tool_name, attempt, e,
                    )
                    if attempt == 2:
                        raise RuntimeError(
                            f"MCP tool '{tool_name}' failed after 2 attempts: {e}"
                        ) from e
                    # Reconnect and retry — close old session first
                    try:
                        if self._session is not None:
                            await self._session.__aexit__(None, None, None)
                    except Exception:
                        pass
                    try:
                        if self._http_ctx is not None:
                            await self._http_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._available = None
                    self._http_ctx = None
                    self._session = None
                    if not await self._ensure_connected():
                        raise RuntimeError(
                            f"MCP Knowledge Server unavailable after reconnect: {e}"
                        ) from e

            # ── Parse MCP ContentBlock format ──
            content_blocks = getattr(result, "content", [])
            text_parts: list[str] = []
            for block in content_blocks:
                if getattr(block, "type", "") == "text":
                    text_parts.append(getattr(block, "text", ""))

            raw_text = "\n".join(text_parts)
            try:
                return json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                # FastMCP splits list returns into multiple text blocks (NDJSON).
                if len(text_parts) > 1:
                    items: list[Any] = []
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

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """Close the HTTP session."""
        async with self._lock:
            if self._session is not None:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception:
                    pass
                self._session = None
            if self._http_ctx is not None:
                try:
                    await self._http_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                self._http_ctx = None
            self._available = None


# Global singleton
_persistent: _PersistentMCPSession | None = None


def _get_persistent() -> _PersistentMCPSession:
    global _persistent
    if _persistent is None:
        _persistent = _PersistentMCPSession()
    return _persistent


# ---------------------------------------------------------------------------
# Client class (public API — same interface as before)
# ---------------------------------------------------------------------------

class MCPKnowledgeClient:
    """Thin wrapper that delegates to the persistent MCP HTTP session.

    Usage:
        async with MCPKnowledgeClient() as client:
            result = await client.answer_with_citation("Q3营收增长驱动", top_k=3)
    """

    def __init__(self, **kwargs):
        self._ps = _get_persistent()

    async def _connect(self) -> bool:
        return await self._ps._ensure_connected()

    async def _disconnect(self) -> None:
        pass  # persistent — not closed per-client

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
            result = await self._ps.call_tool(
                "answer_with_citation", {"question": query, "top_k": top_k}
            )
            if isinstance(result, dict):
                return {
                    "available": True,
                    "answer": result.get("answer", str(result)),
                    "citations": result.get("sources", []),
                }
            return {"available": True, "answer": str(result), "citations": []}
        except Exception as e:
            logger.warning("[MCP] answer_with_citation failed: %s", str(e)[:150])
            return {"available": False, "answer": None, "citations": []}

    async def list_documents(self, category: str | None = None) -> list[dict[str, Any]]:
        """List knowledge base documents.

        Raises on MCP failure — callers must distinguish "empty" from "unavailable".
        """
        args: dict[str, Any] = {}
        if category is not None:
            args["category"] = category
        result = await self._ps.call_tool("list_documents", args)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "documents" in result:
            return result["documents"]
        return [result] if result else []

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
            result = await self._ps.call_tool(
                "add_document", {"file_path": file_path, "category": category}
            )
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
# Convenience functions (public API preserved)
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
    """Pre-warm the MCP HTTP connection (verify server is reachable)."""
    ps = _get_persistent()
    logger.info("[MCP] Connecting to Knowledge Server at %s ...", MCP_SERVER_URL)
    ok = await ps._ensure_connected()
    if ok:
        logger.info("[MCP] Knowledge Server ready.")
    else:
        logger.warning(
            "[MCP] Knowledge Server at %s is not available — "
            "knowledge features will be disabled until it comes online.",
            MCP_SERVER_URL,
        )


async def shutdown_knowledge_client():
    """Close the persistent HTTP session."""
    global _persistent
    if _persistent is not None:
        await _persistent.disconnect()
        _persistent = None
