"""RAG 检索管线封装 — 直接复用 AI Research Copilot 的 vector/ 模块

提供 HybridSearchPipeline 类，封装三阶段检索（Dense+BM25→RRF→Reranker）
及文档管理、评估等能力。所有 MCP Tool 通过本类调用底层向量库。

核心原则：只加封装壳，不改 Copilot vector/ 模块一行代码。
"""

import sys
import uuid
from pathlib import Path
from typing import Any

from mcp_knowledge_agent.core.config import Config, get_config
from mcp_knowledge_agent.core.loader import extract_text, chunk_text


class HybridSearchPipeline:
    """混合检索管线 — 封装 AI Research Copilot 的 vector/ 模块

    提供统一接口供 MCP Tools 调用：
    - search() — 三阶段混合检索
    - search_similar() — 相似文档发现
    - add_document() — 文档上传+索引
    - remove_document() — 文档删除
    - list_documents() — 文档列表
    - get_stats() — 索引统计
    - evaluate() — 检索质量评估
    - generate_answer_with_citations() — LLM 生成+证据锚定
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

        # 将 Copilot backend 目录加入 sys.path，以便 import vector 模块
        copilot_backend = Path(self.config.copilot_backend_dir).resolve()
        if str(copilot_backend) not in sys.path:
            sys.path.insert(0, str(copilot_backend))

        # 延迟导入 — 确保 ChromaDB 数据目录正确设置后再初始化
        self._vector = None
        self._collection = None
        self._collection_name = "knowledge_base"

    @property
    def vector(self):
        """懒加载 vector 模块"""
        if self._vector is None:
            import vector as _v
            self._vector = _v
        return self._vector

    @property
    def collection(self):
        """懒加载 knowledge_base collection"""
        if self._collection is None:
            self._collection = self.vector.get_knowledge_collection()
        return self._collection

    # ═══════════════════════════════════════════════════
    # 核心检索
    # ═══════════════════════════════════════════════════

    async def search(
        self,
        query: str,
        top_k: int = 5,
        snippet_mode: bool = False,
        filters: dict | None = None,
    ) -> list[dict]:
        """三阶段混合检索：Dense+BM25→RRF→Reranker

        Args:
            query: 检索查询
            top_k: 返回结果数（1-20）
            snippet_mode: True 时内容截断到 ~500 字
            filters: 元数据过滤（预留）

        Returns:
            [{"doc_id", "content", "source_file", "score", "retrieval_path", "metadata"}, ...]
        """
        top_k = max(1, min(top_k, 20))

        if not query.strip():
            return []

        col = self.collection
        if col.count() == 0:
            return []

        # 第一阶段：混合检索（Dense + BM25 → RRF）
        fused_ids, rrf_scores, id_to_data = self.vector.hybrid_retrieve(
            collection=col,
            collection_name=self._collection_name,
            query=query.strip(),
            coarse_n=self.config.retrieval_coarse_n,
            rrf_k=self.config.rrf_k,
            rrf_top_n=self.config.retrieval_coarse_n,
        )

        if not fused_ids:
            return []

        # 构建粗排列表
        coarse_items = []
        coarse_docs = []
        for doc_id in fused_ids:
            data = id_to_data.get(doc_id, {})
            doc = data.get("document", "")
            meta = data.get("metadata", {})
            coarse_items.append({"id": doc_id, "doc": doc, "meta": meta})
            coarse_docs.append(doc)

        # 第二阶段：Reranker 精排
        reranked = self.vector.rerank(query.strip(), coarse_docs, top_n=top_k)

        if reranked:
            result_indices = [r[0] for r in reranked]
            result_scores = [r[1] for r in reranked]
            retrieval_path = "Dense+BM25→RRF→Reranker"
        else:
            # Fallback：按 RRF 分数
            n = min(top_k, len(coarse_items))
            result_indices = list(range(n))
            result_scores = [rrf_scores[i] if i < len(rrf_scores) else 0.0 for i in range(n)]
            retrieval_path = "Dense+BM25→RRF（Reranker 未启用）"

        # 构建结果
        results = []
        for rank, idx in enumerate(result_indices):
            if idx >= len(coarse_items):
                continue
            item = coarse_items[idx]
            content = item["doc"]
            if snippet_mode:
                content = content[:500]

            results.append({
                "doc_id": item["id"],
                "content": content,
                "source_file": item["meta"].get("source_file", "未知来源"),
                "score": result_scores[rank],
                "retrieval_path": retrieval_path,
                "metadata": item["meta"],
            })

        return results

    # ═══════════════════════════════════════════════════
    # 相似文档发现
    # ═══════════════════════════════════════════════════

    async def search_similar(self, doc_id: str, top_k: int = 5) -> list[dict]:
        """查找与指定文档内容相似的文档

        通过 doc_id 对应的 chunk 向量做相似度查询。

        Args:
            doc_id: 文档 chunk ID
            top_k: 返回结果数

        Returns:
            相似文档列表（格式同 search()）
        """
        top_k = max(1, min(top_k, 10))

        col = self.collection
        if col.count() == 0:
            return []

        try:
            # 获取目标 chunk
            target = col.get(ids=[doc_id])
            if not target or not target.get("ids"):
                return []

            target_embedding = target.get("embeddings")
            if not target_embedding:
                return []

            # 向量相似度查询（排除自身）
            results = col.query(
                query_embeddings=[target_embedding[0]],
                n_results=top_k + 1,  # +1 因为结果可能包含自身
            )

            similar = []
            if results.get("ids") and results["ids"][0]:
                for i, rid in enumerate(results["ids"][0]):
                    if rid == doc_id:
                        continue  # 排除自身
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    doc = results["documents"][0][i] if results.get("documents") else ""
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = max(0.0, 1.0 - distance / 2.0)

                    similar.append({
                        "doc_id": rid,
                        "content": doc[:800],
                        "source_file": meta.get("source_file", "未知来源"),
                        "score": round(score, 4),
                        "retrieval_path": "VectorSimilarity",
                        "metadata": meta,
                    })

                    if len(similar) >= top_k:
                        break

            return similar
        except Exception:
            return []

    # ═══════════════════════════════════════════════════
    # 文档管理
    # ═══════════════════════════════════════════════════

    async def add_document(self, file_path: str, category: str = "general") -> dict:
        """上传文档到知识库，自动分块、向量化存入 ChromaDB

        Args:
            file_path: 文件路径（PDF/TXT/MD）
            category: 文档分类标签

        Returns:
            {"doc_id": doc_id_base, "file": file_path, "chunks": n, "status": "indexed"}
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        # 校验文件大小
        file_size = path.stat().st_size
        if file_size > self.config.max_file_size:
            raise ValueError(
                f"文件过大（{file_size / 1024 / 1024:.1f} MB），"
                f"限制 {self.config.max_file_size / 1024 / 1024:.0f} MB"
            )

        # 提取文本
        text = extract_text(file_path)
        if not text.strip():
            raise ValueError("文件中未提取到文本内容")

        # 切片
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            raise ValueError("文本切片失败")

        # 向量化 + 存入 ChromaDB
        col = self.collection
        doc_id_base = uuid.uuid4().hex[:12]
        ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_file": path.name,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                "char_count": str(len(chunk)),
                "category": category,
            }
            for i, chunk in enumerate(chunks)
        ]

        col.upsert(ids=ids, documents=chunks, metadatas=metadatas)

        # BM25 缓存失效
        self.vector._invalidate_bm25_cache(self._collection_name)

        return {
            "doc_id": doc_id_base,
            "file": str(path),
            "chunks": len(chunks),
            "status": "indexed",
        }

    async def remove_document(self, doc_id: str) -> dict:
        """删除知识库中的指定文档及其所有向量索引

        Args:
            doc_id: 文档 ID（add_document 返回的 doc_id 或完整 chunk ID）

        Returns:
            {"doc_id": doc_id, "status": "removed", "chunks_deleted": n}
        """
        col = self.collection
        if col.count() == 0:
            raise ValueError("知识库为空")

        # 查找匹配的所有 chunk ID
        all_data = col.get()
        ids_to_delete = []
        for i, cid in enumerate(all_data.get("ids", [])):
            if cid.startswith(doc_id):  # 支持 doc_id_base 前缀匹配
                ids_to_delete.append(cid)

        if not ids_to_delete:
            raise ValueError(f"未找到文档「{doc_id}」")

        col.delete(ids=ids_to_delete)
        self.vector._invalidate_bm25_cache(self._collection_name)

        return {
            "doc_id": doc_id,
            "status": "removed",
            "chunks_deleted": len(ids_to_delete),
        }

    async def list_documents(self, category: str | None = None) -> list[dict]:
        """列出知识库文档，支持按分类过滤

        Returns:
            [{"doc_id": str, "filename": str, "chunks": int, "category": str}, ...]
        """
        col = self.collection
        if col.count() == 0:
            return []

        try:
            all_data = col.get()
            file_info: dict[str, dict] = {}
            ids = all_data.get("ids", [])
            metadatas = all_data.get("metadatas", [])
            for i, meta in enumerate(metadatas):
                if not meta:
                    continue
                fname = meta.get("source_file", "未知文件")
                cat = meta.get("category", "general")

                if category and cat != category:
                    continue

                if fname not in file_info:
                    # Extract doc_id base from chunk ID ("abc123_chunk_0" → "abc123")
                    chunk_id = ids[i] if i < len(ids) else ""
                    doc_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
                    file_info[fname] = {
                        "doc_id": doc_id,
                        "filename": fname,
                        "chunks": 0,
                        "category": cat,
                    }
                file_info[fname]["chunks"] += 1

            return sorted(file_info.values(), key=lambda x: x["filename"])
        except Exception:
            return []

    async def get_stats(self) -> dict:
        """知识库统计

        Returns:
            {"total_files": n, "total_chunks": n, "embedding_dim": 1024, "storage_path": str}
        """
        col = self.collection
        chunk_count = col.count()

        # 统计唯一文件数
        file_count = 0
        if chunk_count > 0:
            try:
                all_data = col.get()
                sources = set()
                if all_data.get("metadatas"):
                    for meta in all_data["metadatas"]:
                        if meta and "source_file" in meta:
                            sources.add(meta["source_file"])
                file_count = len(sources) if sources else chunk_count
            except Exception:
                file_count = chunk_count

        # ChromaDB 存储大小 — 取 vector 模块的真实持久化目录，
        # 而非 CHROMA_PERSIST_DIR 环境变量（后者可能与实际不符）
        import os
        real_dir = getattr(self.vector, "CHROMA_DIR", None)
        chroma_dir = Path(real_dir) if real_dir else Path(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        )
        storage_size = 0
        if chroma_dir.exists():
            storage_size = sum(
                f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file()
            )

        return {
            "total_files": file_count,
            "total_chunks": chunk_count,
            "embedding_dim": 1024,
            "storage_path": str(chroma_dir.resolve()),
            "storage_size_bytes": storage_size,
            "storage_size_mb": round(storage_size / 1024 / 1024, 2),
        }

    # ═══════════════════════════════════════════════════
    # 检索质量评估
    # ═══════════════════════════════════════════════════

    async def evaluate(self) -> dict:
        """基于内置测试用例评估检索质量

        评估方式（按源文件而非 chunk ID 匹配，适应索引重建场景）：
        - Recall@5：预期文件中有几个出现在 top-5 返回结果里
        - MRR：第一个命中期盼文件的排名倒数
        - 关键词检查：返回内容中是否包含预期关键词

        Returns:
            {"total_queries", "recall_at_5", "mrr", "keyword_pass_rate",
             "by_category": {...}, "failed_cases": [...]}
        """
        # 加载测试用例
        test_cases = self._load_test_cases()
        if not test_cases:
            return {
                "total_queries": 0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "keyword_pass_rate": 0.0,
                "by_category": {},
                "failed_cases": [],
            }

        # 按类别分组统计
        by_category: dict[str, dict] = {}
        total_expected_files = 0
        total_files_hit = 0
        total_rr = 0.0
        total_keyword_checks = 0
        total_keyword_pass = 0
        failed_cases: list[dict] = []

        for case in test_cases:
            query = case.get("query", "")
            expected_files = case.get("expected_files", [])
            expected_keywords = case.get("expected_keywords", [])
            category = case.get("category", "未分类")

            if category not in by_category:
                by_category[category] = {
                    "total": 0, "expected_files": 0, "files_hit": 0,
                    "rr_sum": 0.0, "keyword_pass": 0, "keyword_total": 0,
                }

            by_category[category]["total"] += 1
            by_category[category]["expected_files"] += len(expected_files)
            total_expected_files += len(expected_files)

            if not query:
                continue

            # 执行检索
            results = await self.search(query, top_k=5)
            result_files = [r["source_file"] for r in results]
            result_content = " ".join([r["content"] for r in results])

            # ── Recall@5：按源文件匹配 ──
            expected_set = set(expected_files)
            result_set = set(result_files)
            files_hit = len(result_set & expected_set)
            total_files_hit += files_hit
            by_category[category]["files_hit"] += files_hit

            # ── MRR：第一个命中期盼文件的排名 ──
            rr = 0.0
            for rank, fname in enumerate(result_files, start=1):
                if fname in expected_set:
                    rr = 1.0 / rank
                    break
            total_rr += rr
            by_category[category]["rr_sum"] += rr

            # ── 关键词验证 ──
            if expected_keywords:
                by_category[category]["keyword_total"] += len(expected_keywords)
                total_keyword_checks += len(expected_keywords)
                for kw in expected_keywords:
                    if kw.lower() in result_content.lower():
                        total_keyword_pass += 1
                        by_category[category]["keyword_pass"] += 1

            # ── 未命中记录 ──
            if files_hit == 0:
                failed_cases.append({
                    "query": query,
                    "category": category,
                    "expected_files": expected_files,
                    "returned_files": result_files[:5],
                    "expected_keywords": expected_keywords,
                })

        # ── 计算分项指标 ──
        for cat, stats in by_category.items():
            n = stats["total"]
            ef = stats["expected_files"]
            stats["recall_at_5"] = round(stats["files_hit"] / ef, 4) if ef > 0 else 0.0
            stats["mrr"] = round(stats["rr_sum"] / n, 4) if n > 0 else 0.0
            kt = stats["keyword_total"]
            stats["keyword_pass_rate"] = round(stats["keyword_pass"] / kt, 4) if kt > 0 else 1.0
            del stats["files_hit"], stats["expected_files"], stats["rr_sum"], stats["keyword_pass"], stats["keyword_total"]

        return {
            "total_queries": len(test_cases),
            "recall_at_5": round(total_files_hit / total_expected_files, 4) if total_expected_files > 0 else 0.0,
            "mrr": round(total_rr / len(test_cases), 4) if test_cases else 0.0,
            "keyword_pass_rate": round(total_keyword_pass / total_keyword_checks, 4) if total_keyword_checks > 0 else 1.0,
            "by_category": by_category,
            "failed_cases": failed_cases,
        }

    def _load_test_cases(self) -> list[dict]:
        """加载内置测试用例"""
        import json
        cases_path = Path(__file__).resolve().parent.parent.parent / "tests" / "test_cases" / "baseline.json"
        if cases_path.exists():
            try:
                return json.loads(cases_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    # ═══════════════════════════════════════════════════
    # 智能问答
    # ═══════════════════════════════════════════════════

    async def generate_answer_with_citations(
        self,
        question: str,
        contexts: list[dict],
        llm_config: dict,
    ) -> tuple[str, list[dict]]:
        """LLM 生成带证据标注的回答

        Args:
            question: 用户问题
            contexts: 检索到的相关文档列表
            llm_config: {"api_key", "base_url", "model"}

        Returns:
            (answer_text, used_sources) — 回答文本 + 使用的来源列表
        """
        if not contexts:
            return "知识库中未找到与该问题足够相关的信息。", []

        # 构建 LLM 消息
        context_text = "\n\n---\n\n".join([
            f"[来源 {i+1}] 文件: {c.get('source_file', '未知')}\n{c.get('content', '')[:1000]}"
            for i, c in enumerate(contexts)
        ])

        system_prompt = (
            "你是一个知识库问答助手。根据提供的文档片段回答问题。\n"
            "要求：\n"
            "1. 只基于提供的文档内容回答，不要编造信息\n"
            "2. 回答中引用来源时使用 [来源 N] 标注\n"
            "3. 如果文档信息不足以回答问题，请明确说明\n"
            "4. 回答结构清晰，使用 Markdown 格式\n"
            "5. 回答末尾列出所有引用的来源"
        )

        user_prompt = f"## 问题\n{question}\n\n## 参考文档\n{context_text}\n\n请基于以上文档回答问题。"

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
            )

            response = await client.chat.completions.create(
                model=llm_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            answer_text = response.choices[0].message.content or ""

            # 构建来源列表
            used_sources = [
                {
                    "source": c.get("source_file", "未知"),
                    "content": c.get("content", ""),
                    "score": c.get("score", 0.0),
                }
                for c in contexts
            ]

            return answer_text, used_sources

        except Exception as e:
            return f"LLM 生成回答失败：{e}", []
