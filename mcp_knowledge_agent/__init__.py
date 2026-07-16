"""MCP Knowledge Agent Server — 把 AI Research Copilot 的 RAG 检索管线封装为 MCP 协议 Server

8 个工具分两层：
- 智能层 1 个：answer_with_citation — 检索 + LLM 生成 + 证据标注，一步闭环
- 原子层 7 个：search_knowledge / search_similar / evaluate_retrieval
              add_document / remove_document / list_documents / get_index_stats

底层直接复用 AI Research Copilot 的 vector/ 模块，零改动。
"""

__version__ = "0.1.0"
