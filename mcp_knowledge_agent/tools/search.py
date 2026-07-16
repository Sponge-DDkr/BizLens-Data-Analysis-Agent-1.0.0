"""search_knowledge + search_similar — 核心检索工具

search_knowledge: 三阶段混合检索（Dense+BM25→RRF→Reranker）
search_similar: 基于文档向量的相似文档发现
"""

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline
from mcp_knowledge_agent.tools.schemas import SearchInput


def register_search_tool(mcp: FastMCP, pipeline: HybridSearchPipeline):
    """注册 search_knowledge — 三阶段混合检索"""

    @mcp.tool(description=(
        "在知识库中搜索文档。\n"
        "三阶段混合检索：Dense(BGE-large-zh-v1.5) + BM25(jieba) → RRF 融合 → Cross-Encoder Reranker。\n"
        "支持 snippet_mode 截断省 token、filters 元数据过滤。"
    ))
    async def search_knowledge(params: SearchInput) -> dict:
        results = await pipeline.search(
            query=params.query,
            top_k=params.top_k,
            snippet_mode=params.snippet_mode,
            filters=params.filters or {},
        )
        return {
            "query": params.query,
            "total_hits": len(results),
            "results": results,
        }


def register_similar_tool(mcp: FastMCP, pipeline: HybridSearchPipeline):
    """注册 search_similar — 相似文档发现"""

    @mcp.tool(description=(
        "查找与指定文档内容相似的其他文档。"
        "用于发现关联资料、去重检测。"
    ))
    async def search_similar(doc_id: str, top_k: int = 5) -> list[dict]:
        return await pipeline.search_similar(doc_id, top_k)
