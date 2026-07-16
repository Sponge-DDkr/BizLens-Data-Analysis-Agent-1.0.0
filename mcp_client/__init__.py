"""BizLens MCP Integration Package — Knowledge Server Client"""

from mcp_client.knowledge_client import (
    MCPKnowledgeClient,
    query_knowledge,
    add_knowledge_document,
    remove_knowledge_document,
    list_knowledge_documents,
    get_knowledge_stats,
    startup_knowledge_client,
    shutdown_knowledge_client,
)

__all__ = [
    "MCPKnowledgeClient",
    "query_knowledge",
    "add_knowledge_document",
    "remove_knowledge_document",
    "list_knowledge_documents",
    "get_knowledge_stats",
    "startup_knowledge_client",
    "shutdown_knowledge_client",
]
