"""answer_with_citation — 智能问答工具 ★

检索 + LLM 生成 + 证据链标注，一个工具完成知识问答闭环。
与 search_knowledge 的区别：本工具直接返回「答案 + 引用来源」，不需要客户端再整合。
检索质量不足时 confidence 返回 'uncertain'，不会强行编造。
"""

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.config import Config
from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline


def register_answer_tool(mcp: FastMCP, pipeline: HybridSearchPipeline, config: Config):
    """注册 answer_with_citation — 智能问答（检索 + LLM 生成 + 证据标注）"""

    @mcp.tool(description=(
        "基于知识库检索结果生成带证据链的回答。\n"
        "流程：检索 → 筛选高相关片段 → LLM 生成回答 → 标注来源。\n"
        "与 search_knowledge 的区别：本工具直接返回「答案 + 引用来源」，不需要客户端再整合。\n"
        "检索质量不足时 confidence 返回 'uncertain'，不会强行编造。"
    ))
    async def answer_with_citation(
        question: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> dict:
        # 1. 检索
        raw = await pipeline.search(query=question, top_k=top_k)
        relevant = [r for r in raw if r.get("score", 0) >= min_score]

        if not relevant:
            return {
                "answer": "知识库中未找到与该问题足够相关的信息。",
                "sources": [],
                "confidence": "uncertain",
            }

        # 2. LLM 生成 + 证据锚定
        answer_text, used = await pipeline.generate_answer_with_citations(
            question=question,
            contexts=relevant,
            llm_config=config.llm,
        )

        # 3. LLM 不可用时的降级：generate_answer_with_citations 内部 catch 异常
        #    并返回 "LLM 生成回答失败：..." 字符串，不会抛出 → 需检查返回值
        if "LLM 生成回答失败" in answer_text:
            answer_text = (
                f"（LLM 暂不可用，以下为原始检索结果）\n\n"
                f"**问题**: {question}\n\n"
                f"**检索到 {len(relevant)} 条相关片段**:\n\n"
                + "\n\n---\n\n".join(
                    f"**[{r.get('source_file', '未知')}]** (相关度: {r.get('score', 0):.2f})\n{r.get('content', '')[:500]}"
                    for r in relevant[:5]
                )
            )
            used = relevant

        # 3. 置信度评估：基于引用来源的平均分
        if used:
            avg = sum(s.get("score", 0) for s in used) / len(used)
            if avg >= 0.7:
                confidence = "high"
            elif avg >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "uncertain"

        return {
            "answer": answer_text,
            "sources": [
                {
                    "file": s.get("source", "未知"),
                    "excerpt": s.get("content", "")[:200],
                    "score": s.get("score", 0),
                }
                for s in used
            ],
            "confidence": confidence,
        }
