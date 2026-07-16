"""配置管理 — 环境变量 + .env 文件，与 AI Research Copilot 配置解耦"""

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 加载 mcp-knowledge-agent 项目根目录的 .env
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass
class Config:
    """MCP Knowledge Agent Server 全局配置"""

    # ── LLM ──
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # ── ChromaDB ──
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
    )

    # ── Embedding & Reranker 模型 ──
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
        )
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )

    # ── 检索参数 ──
    retrieval_coarse_n: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_COARSE_N", "30"))
    )
    retrieval_fine_n: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_FINE_N", "5"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )

    # ── 文档处理 ──
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )
    max_file_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE", str(20 * 1024 * 1024)))
    )

    # ── AI Research Copilot 路径（用于复用 vector/ 模块） ──
    copilot_backend_dir: str = field(
        default_factory=lambda: os.getenv(
            "COPILOT_BACKEND_DIR",
            str(Path(__file__).resolve().parent.parent.parent.parent / "AI Research Copilot" / "backend"),
        )
    )

    @property
    def llm(self) -> dict:
        """返回 LLM 配置字典，供 vector_pipeline 使用"""
        return {
            "api_key": self.deepseek_api_key,
            "base_url": self.deepseek_base_url,
            "model": self.deepseek_model,
        }

    def validate(self) -> list[str]:
        """校验必要配置项，返回缺失项列表"""
        missing = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY（answer_with_citation 需要）")
        return missing

    @classmethod
    def from_env(cls) -> "Config":
        """工厂方法：从环境变量创建配置"""
        return cls()


_config: Config | None = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
