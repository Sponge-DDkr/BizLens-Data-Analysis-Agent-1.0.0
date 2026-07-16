"""Pydantic 数据模型 — 全部 MCP 工具的输入/输出 Schema"""

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════
# search_knowledge
# ═══════════════════════════════════════════════════

class SearchInput(BaseModel):
    query: str = Field(description="检索查询，支持中文自然语言")
    top_k: int = Field(default=5, ge=1, le=20)
    snippet_mode: bool = Field(
        default=False,
        description="True 时内容截断到 ~500 字，减少 token 消耗",
    )
    filters: dict | None = Field(
        default=None,
        description="元数据过滤。预留企业权限场景: {'department': 'finance'}",
    )


class SearchResult(BaseModel):
    doc_id: str
    content: str  # snippet_mode=True 时截断
    source_file: str
    score: float
    retrieval_path: str  # "Dense+BM25→RRF→Reranker"
    metadata: dict


# ═══════════════════════════════════════════════════
# answer_with_citation
# ═══════════════════════════════════════════════════

class CitationAnswer(BaseModel):
    answer: str = Field(description="基于知识库检索结果生成的回答")
    sources: list[dict] = Field(
        default_factory=list,
        description="引用来源列表: [{file, excerpt, score}, ...]",
    )
    confidence: str = Field(
        description="置信度: high / medium / low / uncertain"
    )


# ═══════════════════════════════════════════════════
# evaluate_retrieval
# ═══════════════════════════════════════════════════

class EvalResult(BaseModel):
    total_queries: int = Field(description="测试用例总数")
    recall_at_5: float = Field(description="Recall@5 指标")
    mrr: float = Field(description="Mean Reciprocal Rank")
    by_category: dict[str, dict] = Field(
        default_factory=dict,
        description="按类别分项指标",
    )
    failed_cases: list[dict] = Field(
        default_factory=list,
        description="未命中的测试用例",
    )


# ═══════════════════════════════════════════════════
# add_document / remove_document
# ═══════════════════════════════════════════════════

class DocumentResult(BaseModel):
    doc_id: str = Field(description="文档唯一标识")
    status: str = Field(description="操作状态: indexed / removed")


class RemoveResult(BaseModel):
    doc_id: str
    status: str
    chunks_deleted: int = Field(default=0)


# ═══════════════════════════════════════════════════
# list_documents / get_index_stats
# ═══════════════════════════════════════════════════

class DocumentInfo(BaseModel):
    filename: str = Field(description="文件名")
    chunks: int = Field(description="切片数量")
    category: str = Field(default="general", description="文档分类")


class IndexStats(BaseModel):
    total_files: int = Field(description="已索引的文档数")
    total_chunks: int = Field(description="总切片数")
    embedding_dim: int = Field(description="向量维度")
    storage_path: str = Field(description="ChromaDB 持久化路径")
    storage_size_mb: float = Field(description="存储大小 (MB)")
