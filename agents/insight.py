"""Insight Agent — synthesize analysis results into a structured report.

Architecture:
    1. Receive exec_result + chart_json + query from upstream agents
    2. Query MCP Knowledge Server for business context (answer_with_citation)
    3. LLM (deepseek-chat / V3) generates 5-section report:
       数据事实 / 图表说明 / 业务解读 / 🎯 总结与建议 / 数据来源
    4. Input truncation to avoid API rejection on large analysis outputs
    5. Fallback retry with shorter prompt on LLM call failure
    6. Return markdown report for frontend rendering + PDF export

Day 5 optimization (2026-07-16):
    - 4-section → 5-section (added 🎯 总结与建议 with accuracy constraints)
    - Reduced truncation thresholds (2500/800) to prevent API rejection
    - Added fallback retry with aggressive truncation (1500/400)
    - Uses V3 model (not R1) for better Chinese writing quality

Design note:
    Insight Agent synthesizes three data sources: (1) Code Interpreter's real results,
    (2) Visualization's chart description, (3) MCP Knowledge Server's business context.
    5-section report separates data facts from business interpretation for traceability.
    Summary section directly answers the user's core question with key data points.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.llm_client import get_llm_client
from mcp_client.knowledge_client import query_knowledge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Insight System Prompt
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = """你是一个资深数据分析报告撰写专家。你需要根据数据分析结果、图表信息和业务背景知识，生成一份专业的分析报告。

## 报告格式要求

报告必须包含以下五个部分，使用 Markdown 格式：

### 📊 数据事实
- 基于分析执行结果，提取关键数据指标
- 使用表格展示核心数据对比
- 列出 Top 3、增长率、总额等关键数值
- ⚠️ 这部分只能用分析执行结果中的真实数字，不能编造

### 📈 图表说明
- 描述生成的图表展示了什么信息
- 说明图表类型选择的原因（为什么用柱状图/折线图/饼图）
- 指出图表中的关键趋势或异常点

### 💡 业务解读
{knowledge_section}

### 🎯 总结与建议
- **直接回答**：用 1-2 句话直接回应用户问题中的核心关切（首句必须复述用户问题的关键词）
- **关键数据支撑**：从「数据事实」中提取最关键的 1-2 个数字作为结论支撑（⚠️ 这些数字必须在「数据事实」段落中能找到原文对应，禁止凭空出现新数字）
- **可行建议**：如有明确结论，精炼为 1-2 条行动建议（每条不超过 20 字）；如数据不足以给出建议，说明「需要补充哪些信息」
- **外部归因**：如有知识库来源且与结论相关，用一句话归因（标注来源文档名）
- ⚠️ 总字数不超过 150 字，精炼简洁

### 📁 数据来源
- 列出分析所使用的数据文件
- 如有知识库来源，列出引用文档和摘录

## 写作规范

1. **数据准确**：所有数字必须来自分析结果，不要编造
2. **语言专业**：使用商业分析术语，但保持可读性
3. **结论明确**：每个部分要有明确的结论句
4. **格式整洁**：合理使用表格、列表、加粗强调
5. **区分事实与观点**：数据事实 vs 业务解读 vs 总结要有明确边界
6. 中文输出，数字使用千分位格式
7. **边界情况处理（重要）**：
   - 如数据分析结果中出现 `inf%`、`inf` → 意味着增长率的基数为 0（新品上线/上月无数据），报告中使用"🆕 新增（上月未上线）"替代
   - NaN 或空值 → 解读为"数据缺失"或"该月份未上线"，不要解释为"营收为 0"
   - 遇到 N/A → 说明该指标因数据不足无法计算，如实呈现即可
8. **总结锚定规则（严格遵守）**：
   - 总结中出现的任何数字，必须在「数据事实」部分能找到原文对应——禁止凭空出现新数字
   - 总结段首句必须包含用户问题的核心关键词，确保不答非所问
   - 如果没有足够数据得出结论，如实说明「当前数据不足以给出明确结论」，不要强行编造
"""

INSIGHT_WITH_KNOWLEDGE_SECTION = """- 结合知识库提供的业务背景，分析数据背后的原因
- 每一条业务归因必须标注来源（文档名 + 引用）
- 如果知识库无相关信息，基于数据本身给出合理推测，并明确标注「推测」
- 区分「有来源支撑的结论」和「基于数据的推测」"""

INSIGHT_WITHOUT_KNOWLEDGE_SECTION = """- 基于数据本身给出合理的业务归因推测
- ⚠️ 知识库暂不可用，所有业务解读均为推测，需明确标注「推测」
- 建议业务人员结合实际情况验证"""


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

async def generate_insight_report(
    exec_result: dict[str, Any],
    chart_json: dict[str, Any] | None,
    query: str,
    steps: list[dict[str, Any]],
    session_id: str = "",
) -> str:
    """Generate a comprehensive 4-section analysis report.

    Args:
        exec_result: Output from Code Interpreter (output, results, success).
        chart_json: Plotly figure JSON from Visualization Agent (or None).
        query: User's original analysis question.
        steps: Planner's analysis steps for context.
        session_id: Upload session ID.

    Returns:
        Markdown-formatted report string.
    """
    client = get_llm_client()

    # ------------------------------------------------------------------
    # Step 1: Gather analysis summary
    # ------------------------------------------------------------------
    analysis_summary = _build_analysis_summary(exec_result, steps)
    chart_summary = _build_chart_summary(chart_json, steps)

    # ------------------------------------------------------------------
    # Step 2: Query MCP Knowledge Server
    # ------------------------------------------------------------------
    logger.info("[Insight] Querying MCP Knowledge Server...")
    knowledge = await query_knowledge(query=query, top_k=3)
    knowledge_text = _format_knowledge_context(knowledge)

    # ------------------------------------------------------------------
    # Step 3: Build prompt
    # ------------------------------------------------------------------
    if knowledge.get("available") and knowledge.get("answer"):
        knowledge_section = INSIGHT_WITH_KNOWLEDGE_SECTION
        logger.info("[Insight] Knowledge context available — enriching report")
    else:
        knowledge_section = INSIGHT_WITHOUT_KNOWLEDGE_SECTION
        logger.info("[Insight] No knowledge context — generating report with data only")

    system_prompt = INSIGHT_SYSTEM_PROMPT.format(knowledge_section=knowledge_section)

    user_prompt = f"""## 用户问题

{query}

## 数据分析结果

{analysis_summary}

## 图表信息

{chart_summary}

## 知识库背景

{knowledge_text}

---

请根据以上信息，生成四段式分析报告。"""

    # ------------------------------------------------------------------
    # Step 4: Generate report via LLM
    # ------------------------------------------------------------------
    logger.info("[Insight] Generating report via LLM...")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        report = await client.chat(
            messages=messages,
            temperature=0.4,  # Slightly higher for natural language quality
            max_tokens=3072,
        )
    except Exception as e:
        # Fallback: retry with aggressively truncated prompt
        logger.warning(f"[Insight] First attempt failed ({e}), retrying with shorter prompt...")
        short_summary = _build_analysis_summary(
            exec_result, steps,
            max_output_len=1500, max_step_len=400,
        )
        short_user_prompt = f"""## 用户问题

{query}

## 数据分析结果

{short_summary}

## 图表信息

{chart_summary}

## 知识库背景

{knowledge_text}

---

请根据以上信息，生成四段式分析报告。"""
        messages[1] = {"role": "user", "content": short_user_prompt}
        report = await client.chat(
            messages=messages,
            temperature=0.4,
            max_tokens=3072,
        )

    logger.info(f"[Insight] Report generated ({len(report)} chars)")
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analysis_summary(
    exec_result: dict[str, Any],
    steps: list[dict[str, Any]],
    max_output_len: int = 2500,
    max_step_len: int = 800,
) -> str:
    """Extract a structured summary from code execution results."""
    parts: list[str] = []

    # Overall output
    output = exec_result.get("output", "")
    if output:
        # Truncate very long outputs (insight prompt must stay concise for API reliability)
        if len(output) > max_output_len:
            output = output[:max_output_len] + "\n... (输出已截断)"
        parts.append(output)

    # Per-step results
    results = exec_result.get("results", [])
    if results:
        for r in results:
            step_num = r.get("step", "?")
            desc = r.get("description", "")
            step_out = r.get("output", "")
            success = r.get("success", True)

            status = "✅" if success else "❌"
            parts.append(f"\n### Step {step_num}: {desc} {status}")
            if step_out:
                if len(step_out) > max_step_len:
                    step_out = step_out[:max_step_len] + "\n... (输出已截断)"
                parts.append(step_out)
            if not success:
                err = r.get("error", "")
                if err:
                    parts.append(f"错误: {err}")

    if not parts:
        return "（暂无分析结果）"

    return "\n".join(parts)


def _build_chart_summary(
    chart_json: dict[str, Any] | None,
    steps: list[dict[str, Any]],
) -> str:
    """Describe the generated chart."""
    if not chart_json:
        chart_steps = [s for s in steps if s.get("type") == "chart"]
        if chart_steps:
            return f"计划生成图表：{'；'.join(s.get('description', '') for s in chart_steps)}（实际未生成）"
        return "本次分析未生成图表"

    parts: list[str] = []

    # Chart type from data traces
    data = chart_json.get("data", [])
    if isinstance(data, list):
        trace_types = []
        for trace in data:
            if isinstance(trace, dict):
                ttype = trace.get("type", "unknown")
                name = trace.get("name", "")
                label = f"{ttype}" + (f" ({name})" if name else "")
                trace_types.append(label)
        if trace_types:
            parts.append(f"图表类型: {', '.join(trace_types)}")

    # Layout info
    layout = chart_json.get("layout", {})
    if isinstance(layout, dict):
        title = layout.get("title", {})
        if isinstance(title, dict):
            title_text = title.get("text", "")
        else:
            title_text = str(title) if title else ""
        if title_text:
            parts.append(f"图表标题: {title_text}")

        xaxis = layout.get("xaxis", {})
        yaxis = layout.get("yaxis", {})
        if isinstance(xaxis, dict) and xaxis.get("title"):
            x_title = xaxis["title"]
            if isinstance(x_title, dict):
                x_title = x_title.get("text", "")
            parts.append(f"X轴: {x_title}")
        if isinstance(yaxis, dict) and yaxis.get("title"):
            y_title = yaxis["title"]
            if isinstance(y_title, dict):
                y_title = y_title.get("text", "")
            parts.append(f"Y轴: {y_title}")

    return "\n".join(parts) if parts else "图表已生成（交互式 Plotly 图表）"


def _format_knowledge_context(knowledge: dict[str, Any]) -> str:
    """Format MCP knowledge result for inclusion in the LLM prompt."""
    if not knowledge.get("available") or not knowledge.get("answer"):
        return "（知识库暂不可用，请基于数据本身进行分析和推测）"

    parts: list[str] = []
    answer = knowledge.get("answer", "")
    if answer:
        parts.append(f"知识库回答:\n{answer}")

    citations = knowledge.get("citations", [])
    if citations:
        parts.append("\n信息来源:")
        for i, cite in enumerate(citations, 1):
            source = cite.get("source", "未知来源")
            excerpt = cite.get("excerpt", "")
            parts.append(f"  [{i}] {source}")
            if excerpt:
                # Truncate long excerpts
                if len(excerpt) > 200:
                    excerpt = excerpt[:200] + "..."
                parts.append(f"      引用: {excerpt}")

    return "\n".join(parts) if parts else "（知识库返回为空）"
