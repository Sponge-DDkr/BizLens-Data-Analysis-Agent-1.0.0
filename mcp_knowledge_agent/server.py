"""MCP Knowledge Agent Server — FastMCP 实例 + create_server() 工厂

8 个工具分两层：
- 智能层 1 个：answer_with_citation — 检索 + LLM 生成 + 证据标注（Day 3 完成）
- 原子层 7 个：search_knowledge / search_similar / evaluate_retrieval
              add_document / remove_document / list_documents / get_index_stats
"""

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.config import Config
from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline

# 原子层工具注册函数
from mcp_knowledge_agent.tools.search import register_search_tool, register_similar_tool
from mcp_knowledge_agent.tools.evaluate import register_evaluate_tool
from mcp_knowledge_agent.tools.document import register_document_tools
from mcp_knowledge_agent.tools.manage import register_manage_tools
# 智能层工具
from mcp_knowledge_agent.tools.answer import register_answer_tool


def create_server(
    config: Config | None = None,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> FastMCP:
    """创建 MCP Knowledge Agent Server 实例

    注册全部 8 个工具（原子层 7 + 智能层 1），
    底层检索管线直接复用 AI Research Copilot 的 vector/ 模块。

    Args:
        config: 配置对象，None 时从环境变量加载
        host: HTTP 模式监听地址（stdio 模式忽略）
        port: HTTP 模式监听端口（stdio 模式忽略）

    Returns:
        配置好的 FastMCP 实例
    """
    if config is None:
        config = Config.from_env()

    pipeline = HybridSearchPipeline(config)

    mcp = FastMCP(
        name="mcp-knowledge-agent",
        json_response=True,
        host=host,
        port=port,
    )

    # ── 原子层（7 个工具，来自独立模块）──
    register_search_tool(mcp, pipeline)
    register_similar_tool(mcp, pipeline)
    register_evaluate_tool(mcp, pipeline)
    register_document_tools(mcp, pipeline)
    register_manage_tools(mcp, pipeline)

    # ── 智能层（1 个工具）──
    register_answer_tool(mcp, pipeline, config)

    # ── 预加载模型 — 避免首次 tool call 时阻塞 event loop ──
    import sys as _sys
    import time as _time
    _sys.stderr.write("[MCP] Pre-loading embedding model...\n")
    _sys.stderr.flush()
    _t0 = _time.time()
    try:
        _ = pipeline.collection  # triggers get_embedder() → load bge-large-zh-v1.5
        _sys.stderr.write(f"[MCP] Model ready in {_time.time()-_t0:.1f}s (chunks={_.count()})\n")
    except Exception as _e:
        _sys.stderr.write(f"[MCP] Model pre-load FAILED: {_e}\n")
    _sys.stderr.flush()

    return mcp
