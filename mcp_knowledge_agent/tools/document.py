"""add_document + remove_document — 文档管理工具

add_document: 上传文档（PDF/TXT/MD），自动分块、向量化存入 ChromaDB
remove_document: 删除指定文档及其所有向量索引
"""

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline


def register_document_tools(mcp: FastMCP, pipeline: HybridSearchPipeline):
    """注册 add_document + remove_document"""

    @mcp.tool(description=(
        "上传文档到知识库，支持 PDF/TXT/MD。"
        "自动分块、向量化存入 ChromaDB。"
        "返回文档 ID 供后续引用或删除。"
    ))
    async def add_document(file_path: str, category: str = "general") -> dict:
        return await pipeline.add_document(file_path, category)

    @mcp.tool(description=(
        "删除知识库中的指定文档及其所有向量索引。"
        "支持传入 add_document 返回的 doc_id 进行精确删除。"
    ))
    async def remove_document(doc_id: str) -> dict:
        return await pipeline.remove_document(doc_id)
