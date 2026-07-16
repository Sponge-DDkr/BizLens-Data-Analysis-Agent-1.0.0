"""list_documents + get_index_stats — 知识库管理工具

list_documents: 列出已上传文件，可按分类过滤
get_index_stats: 知识库统计（文件数、chunk 数、向量维度、存储大小）
"""

import sys
import time

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline


def register_manage_tools(mcp: FastMCP, pipeline: HybridSearchPipeline):
    """注册 list_documents + get_index_stats"""

    @mcp.tool(description=(
        "列出知识库中所有已上传的文档。"
        "可按 category 分类过滤。"
        "返回每个文件的名称、切片数量和分类。"
    ))
    async def list_documents(category: str | None = None) -> list[dict]:
        sys.stderr.write(f"[manage] list_documents(category={category}) start\n")
        sys.stderr.flush()
        t0 = time.time()
        try:
            result = await pipeline.list_documents(category)
            sys.stderr.write(f"[manage] list_documents done in {time.time()-t0:.1f}s: {len(result)} docs\n")
            sys.stderr.flush()
            return result
        except Exception as e:
            sys.stderr.write(f"[manage] list_documents ERROR: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            raise

    @mcp.tool(description=(
        "获取知识库统计信息。"
        "返回：文件数、chunk 总数、向量维度（1024）、存储路径、存储大小。"
    ))
    async def get_index_stats() -> dict:
        sys.stderr.write("[manage] get_index_stats start\n")
        sys.stderr.flush()
        t0 = time.time()
        try:
            result = await pipeline.get_stats()
            sys.stderr.write(f"[manage] get_index_stats done in {time.time()-t0:.1f}s\n")
            sys.stderr.flush()
            return result
        except Exception as e:
            sys.stderr.write(f"[manage] get_index_stats ERROR: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            raise
