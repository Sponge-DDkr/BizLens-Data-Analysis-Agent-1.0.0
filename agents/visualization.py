"""Visualization Agent — auto-select chart type, generate Plotly code, execute in sandbox.

Architecture:
    1. Receive analysis results + chart-type steps from Planner
    2. LLM (deepseek-reasoner / R1) selects chart type + generates Plotly Python code
    3. Sandbox executes code → fig.to_json()
    4. On failure: fix prompt with column names + data preview → retry (max 2)
    5. Temperature ramp (0.1 → 0.2) for diverse fix attempts
    6. Return chart_json for frontend rendering

Day 5 optimization (2026-07-16):
    - Fix prompt now includes actual data columns + preview (aligned with Code Interpreter)
    - Fix temperature increases with retry_count to avoid generating identical broken code
    - Uses deepseek-reasoner (R1) for more reliable Plotly code generation
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.code_interpreter import _get_data_context
from backend.config import MODEL_REASONING
from backend.llm_client import get_llm_client
from sandbox.executor import execute_in_sandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VIS_SYSTEM_PROMPT = """你是一个数据可视化专家。根据分析任务描述，选择合适的图表类型并生成 Plotly Python 代码。

## 图表选型规则

| 数据特征 | 图表类型 | Plotly 实现 |
|---------|---------|------------|
| 时序趋势（月份/季度/年份/日期） | 折线图 | go.Scatter(mode='lines+markers') |
| 类别对比（产品线/地区/部门/人员） | 柱状图 | go.Bar() |
| 占比分析（百分比/份额/构成） | 饼图 | go.Pie() |
| 数值分布（年龄/金额分布） | 直方图 | go.Histogram() |
| 双变量关系（相关性） | 散点图 | go.Scatter(mode='markers') |
| 多指标/多系列对比 | 分组/堆叠柱状图 | go.Bar() 多个 trace |
| 排名对比 | 水平柱状图 | go.Bar(orientation='h') |

优先选择最合适的单一图表类型。如果数据适合组合图（如柱状+折线），可以使用多个 trace。
如果使用 `make_subplots` 创建多子图，**必须纵向排列**（`rows=N, cols=1`），禁止横向并排（`rows=1, cols=N`），每个子图高度至少 350px。

## 代码示例（严格参照以下模式编写）

### 分组柱状图（多系列垂直柱）
```python
categories = ['产品A', '产品B', '产品C']
fig = go.Figure()
fig.add_trace(go.Bar(name='Q1', x=categories, y=[120, 200, 150]))
fig.add_trace(go.Bar(name='Q2', x=categories, y=[180, 220, 170]))
fig.update_layout(barmode='group')
```

### 水平分组柱状图（多系列水平柱 — 注意 x/y 互换！）
```python
categories = ['渠道A', '渠道B', '渠道C']
fig = go.Figure()
fig.add_trace(go.Bar(name='点击率', y=categories, x=[5.2, 8.1, 6.8], orientation='h'))
fig.add_trace(go.Bar(name='转化率', y=categories, x=[2.1, 1.5, 3.2], orientation='h'))
fig.add_trace(go.Bar(name='成交率', y=categories, x=[0.8, 0.5, 1.1], orientation='h'))
fig.update_layout(barmode='group')
```
⚠️ 水平柱状图的关键：**y=类别标签, x=数值, orientation='h'**——和垂直柱相反！

### 双轴组合图（柱状+折线）
```python
from plotly.subplots import make_subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(name='营收', x=['7月','8月','9月'], y=[420,510,680]), secondary_y=False)
fig.add_trace(go.Scatter(name='增长率', x=['7月','8月','9月'], y=[0,21,33],
    mode='lines+markers', marker=dict(size=8)), secondary_y=True)
```

## 代码规范

1. 使用 `import plotly.graph_objects as go`，不要用 plotly.express
2. **数据文件定位**（必须按此模板）：
   ```python
   data_dir = os.environ['DATA_DIR']
   files = [f for f in os.listdir(data_dir) if f.endswith(('.csv','.xlsx'))]
   filepath = os.path.join(data_dir, files[0])
   if filepath.endswith('.csv'):
       df = pd.read_csv(filepath, encoding='utf-8-sig')
   else:
       df = pd.read_excel(filepath)
   ```
   不要硬编码 `DATA_DIR = 'data'`，必须用 `os.environ['DATA_DIR']`！
3. 执行必要的数据聚合/分组/排序
4. 设置合适的中文标题、轴标签
5. 多系列用不同颜色区分
6. **最后一行必须是 `print(fig.to_json())`**，前面不要有任何 print 语句
7. `fig.update_layout(template='plotly_white', font=dict(family='Microsoft YaHei, SimHei, sans-serif'))`
8. 图表尺寸合适：`width=700, height=450`；水平柱状图根据类别数量动态调整：`height=max(350, 60 * len(categories) + 100)`
9. **边界情况处理（重要）**：
   - 图表数据中如有 inf/-inf/NaN → 用 `fillna(0)` 或 `replace([np.inf, -np.inf], np.nan)` 清理后再绘图
   - 柱状图/折线图中值为 NaN 的数据点自动跳过，不要显示为 0
   - 如数据本身代表"该月未上线"（NaN），在 tooltip 中显示"未上线"而非数值

## 安全约束（严格遵守）

- 禁止 import os.system / subprocess / socket / requests / urllib / shutil
- 禁止 eval / exec / compile / __import__
- 禁止 open() 写入文件
- 禁止访问网络

## 输出格式

**只输出 Python 代码**，不要加 markdown 代码块标记、不要解释注释、不要前后文字。
"""

VIS_USER_TEMPLATE = """用户问题：{query}

需要生成的图表：{chart_description}

数据分析结果（供参考数据列名和值）：
{analysis_summary}

## 重要提示
- 注意数据的实际类型：月份可能是 "7月"/"8月" 这样的字符串，不是纯数字
- 产品线/分类列的值可能是中文，直接使用即可
- 按原样使用数据中的值，不要做类型转换（不要用 astype(int) 处理含中文的列）
- **水平柱状图**：y=类别标签列表, x=数值列表, orientation='h'——注意和垂直柱 x/y 相反！
- **多 trace 图表**：每个指标单独 add_trace()，数据需要提前从分析结果中提取好
- 如果分析结果中的指标是计算出来的（如点击率=点击量/曝光量），直接在代码中从 df 计算即可

请生成 Plotly 可视化代码，输出 `fig.to_json()`。"""

VIS_FIX_TEMPLATE = """之前生成的可视化代码执行失败了。

**可用数据列**：{columns}

**数据预览（前3行）**：
{data_preview}

**失败代码**：
```python
{failed_code}
```

**报错信息**：
{error_message}

请修正代码并重新输出。常见问题：
- 列名拼写错误（仔细核对上面的「可用数据列」）
- 数据分组/聚合方式不对
- Plotly API 参数错误
- 数据类型不匹配（确保数值列是 int/float）
- 数据值包含中文（如月份写成"7月"），不要用 astype(int) 转换

**只输出修正后的 Python 代码，不要解释。**"""


# ---------------------------------------------------------------------------
# Clean code helper
# ---------------------------------------------------------------------------

def _clean_code(text: str) -> str:
    """Strip markdown fences and surrounding whitespace from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _extract_json(output: str) -> dict[str, Any] | None:
    """Extract Plotly figure JSON from sandbox output.

    The sandbox output should be just `fig.to_json()`, but we handle
    leading/trailing whitespace or log noise gracefully.
    """
    text = output.strip()
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the output
    # Look for the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

async def generate_chart(
    exec_result: dict[str, Any],
    query: str,
    chart_steps: list[dict[str, Any]],
    session_id: str,
    max_retries: int = 2,
) -> dict[str, Any] | None:
    """Generate a Plotly chart based on analysis results.

    Args:
        exec_result: Output from code_interpreter (contains 'output' and 'results').
        query: User's original analysis question.
        chart_steps: Chart-type steps from the Planner (description fields).
        session_id: Upload session ID for locating data files.

    Returns:
        Plotly figure JSON dict, or None if generation failed.
    """
    from backend.config import STORAGE_DIR

    session_dir = str(STORAGE_DIR / session_id)

    # Build chart description from planner steps
    chart_descriptions = [s.get("description", "") for s in chart_steps if s.get("type") == "chart"]
    chart_description = "；".join(chart_descriptions) if chart_descriptions else query

    # Build analysis summary from code_interpreter results
    analysis_parts: list[str] = []
    if exec_result.get("output"):
        # Truncate to avoid token overflow
        analysis_parts.append(exec_result["output"][:3000])
    if exec_result.get("results"):
        for r in exec_result["results"]:
            step_desc = r.get("description", "")
            step_out = r.get("output", "")[:500]
            if step_out:
                analysis_parts.append(f"[{step_desc}]\n{step_out}")
    analysis_summary = "\n\n".join(analysis_parts) if analysis_parts else "（暂无分析结果）"

    client = get_llm_client()

    # --- Step 1: Generate Plotly code ---
    messages = [
        {"role": "system", "content": VIS_SYSTEM_PROMPT},
        {"role": "user", "content": VIS_USER_TEMPLATE.format(
            query=query,
            chart_description=chart_description,
            analysis_summary=analysis_summary[:4000],
        )},
    ]

    code = await client.chat(messages=messages, temperature=0.2, max_tokens=2048, model=MODEL_REASONING)
    code = _clean_code(code)
    logger.info(f"[Visualization] Generated Plotly code ({len(code)} chars)")

    # --- Step 2: Execute + retry loop ---
    from sandbox.docker_executor import SandboxResult

    result = execute_in_sandbox(code, session_dir)
    retry_count = 0

    while not result.success and retry_count < max_retries:
        logger.warning(
            f"[Visualization] Execution failed (retry {retry_count + 1}/{max_retries}): "
            f"{result.error[:120]}"
        )

        # Get actual data columns + preview for the fix prompt
        # (aligns with Code Interpreter's self-healing approach)
        columns, data_preview = _get_data_context(session_dir)

        fix_messages = [
            {"role": "system", "content": VIS_SYSTEM_PROMPT},
            {"role": "user", "content": VIS_FIX_TEMPLATE.format(
                failed_code=code,
                error_message=result.error,
                columns=columns,
                data_preview=data_preview,
            )},
        ]

        # Gradually increase temperature for more diverse fix attempts
        fix_temp = 0.1 + retry_count * 0.1
        code = await client.chat(messages=fix_messages, temperature=fix_temp, max_tokens=2048, model=MODEL_REASONING)
        code = _clean_code(code)
        logger.info(f"[Visualization] Regenerated fix code ({len(code)} chars)")

        result = execute_in_sandbox(code, session_dir)
        retry_count += 1

    if not result.success:
        logger.error(f"[Visualization] Chart generation failed after {retry_count} retries")
        return None

    # --- Step 3: Parse JSON ---
    chart_json = _extract_json(result.output)
    if chart_json is None:
        logger.error(f"[Visualization] Failed to parse JSON from output: {result.output[:300]}")
        return None

    logger.info(
        f"[Visualization] Chart generated successfully "
        f"(retries: {retry_count}, duration: {result.duration_ms:.0f}ms, "
        f"json_keys: {list(chart_json.keys())})"
    )
    return chart_json
