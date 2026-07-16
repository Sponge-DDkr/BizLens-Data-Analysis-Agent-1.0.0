"""Planner Agent — decompose natural-language queries into structured analysis steps.

Uses DeepSeek API (via backend.llm_client) with JSON structured output mode.
Pydantic models validate the LLM response before it enters the LangGraph state.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from backend.config import MODEL_REASONING
from backend.llm_client import get_llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class AnalysisStep(BaseModel):
    """A single analysis step in the plan."""

    step: int = Field(description="1-based step number")
    description: str = Field(description="What this step does, in plain Chinese")
    type: str = Field(
        default="code",
        description="Step type: 'code' (data processing), 'chart' (visualization), or 'insight' (report generation)",
    )


class PlannerOutput(BaseModel):
    """Validated output from the Planner Agent."""

    steps: list[AnalysisStep] = Field(description="Ordered list of analysis steps")
    expected_output: str = Field(description="Summary of what the final output should contain")


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """你是一个数据分析规划专家。用户上传了一个 CSV/Excel 文件，并提出了分析问题。你需要将问题拆解为可执行的分析步骤序列。

## 步骤类型

- **code**: 需要用 Python (Pandas) 执行的数据处理步骤（读取数据、分组统计、计算指标等）
- **chart**: 需要生成图表的步骤（折线图、柱状图、饼图等）
- **insight**: 需要生成文字结论/报告的步骤

## 拆解规则

1. **第一步永远是 code 类型**：了解数据概况（dtypes / describe / null 检查）
2. **中间步骤**：根据用户问题拆解为具体的分析操作（分组、聚合、排序、筛选等），每个步骤对应一个 code 类型
3. **数据可视化**：如果需要图表，单独列一个 chart 步骤
4. **最后一步**：insight 类型，生成分析报告和结论
5. 步骤描述用中文，简洁明确，让代码生成 Agent 能直接理解

## 输出格式

必须输出严格 JSON 格式，不要包含 markdown 代码块标记：

{
  "steps": [
    {"step": 1, "description": "读取数据概况：dtypes、describe、null检查", "type": "code"},
    {"step": 2, "description": "按XX分组统计YY", "type": "code"},
    ...
  ],
  "expected_output": "趋势图 + 排名表 + 分析结论"
}
"""

PLANNER_USER_TEMPLATE = """用户上传了数据文件，并提出以下分析问题：

{query}

请拆解为分析步骤序列（JSON 格式）。"""


# ---------------------------------------------------------------------------
# Planner function
# ---------------------------------------------------------------------------

async def plan_analysis(query: str, session_id: str = "") -> dict[str, Any]:
    """Decompose a user query into structured analysis steps.

    Args:
        query: Natural-language analysis question from the user.
        session_id: Current upload session ID (for context, not used in Day 2).

    Returns:
        Dict with keys:
            steps: list of {"step": int, "description": str, "type": str}
            expected_output: str

    Raises:
        ValueError: If the LLM response fails Pydantic validation.
    """
    client = get_llm_client()

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_TEMPLATE.format(query=query)},
    ]

    logger.info(f"[Planner] Sending decomposition request for: {query[:100]}")

    # Use structured JSON mode for reliable parsing
    raw = await client.chat_structured(messages=messages, temperature=0.2, max_tokens=2048, model=MODEL_REASONING)

    # Pydantic validation
    try:
        validated = PlannerOutput.model_validate(raw)
    except Exception as validation_error:
        logger.error(f"[Planner] Pydantic validation failed: {validation_error}")
        logger.error(f"[Planner] Raw response: {raw}")
        raise ValueError(f"Planner 输出格式校验失败: {validation_error}") from validation_error

    result = {
        "steps": [s.model_dump() for s in validated.steps],
        "expected_output": validated.expected_output,
    }

    logger.info(
        f"[Planner] Decomposition complete: {len(result['steps'])} steps → {result['expected_output']}"
    )
    return result
