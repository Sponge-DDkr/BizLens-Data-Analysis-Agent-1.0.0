"""Code Interpreter Agent — LLM generates Python, Sandbox executes, errors self-heal.

Architecture:
    1. Receive analysis steps from Planner (only "code" type steps)
    2. Step 0 (always): Data Understanding → dtypes, describe, null counts
    3. For each subsequent step: LLM generates Python code → sandbox execution
    4. On failure: error + context fed back to LLM → code fix → retry (max 3)
    5. Temperature ramp (0.1 → 0.25 → 0.4) for diverse fix attempts
    6. On last retry: degradation prompt forces fallback to simple describe/value_counts
    7. Collect all results into exec_result for downstream agents
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.config import MODEL_REASONING, STORAGE_DIR
from backend.llm_client import get_llm_client
from sandbox.docker_executor import SANDBOX_MAX_RETRIES, SandboxResult
from sandbox.executor import execute_in_sandbox  # unified入口 — dev用subprocess, 生产用Docker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CODE_SYSTEM_PROMPT = """你是一个 Python 数据分析专家。你需要根据分析步骤描述生成可执行的 Python 代码。

## 运行环境

- Python 3.12 + Pandas + DuckDB + OpenPyXL + NumPy
- 用户上传的数据文件在 `DATA_DIR` 环境变量指定的目录下
- 使用 `os.environ['DATA_DIR']` 获取数据目录
- 数据目录下的第一个 CSV/Excel 文件就是用户上传的数据

## 代码规范

1. **只输出 Python 代码**，不要加 markdown 代码块标记、不要解释、不要注释
2. 使用 `print()` 输出结果，结果会返回给后续分析节点
3. 自动发现数据文件：用 `os.listdir(DATA_DIR)` 找到 CSV/Excel 文件
4. 根据扩展名选择 pd.read_csv 或 pd.read_excel
5. 处理可能的编码问题（CSV 用 utf-8-sig）
6. 输出格式整洁，使用 `print(df.to_string())` 打印 DataFrame
7. 如果文件不存在，`print("ERROR: 数据文件未找到")`
8. **边界情况处理（重要）**：
   - 计算增长率/百分比时，分母为 0 或 NaN → 输出 "N/A" 或 "新增（上月未上线）"，不要输出 inf
   - `pd.read_csv()` 空值会被解析为 NaN，计算前用 `df['数值列'] = pd.to_numeric(df['数值列'], errors='coerce')` 确认类型
   - 增长率公式示例：`df['增长率'] = ((df['本月'] - df['上月']) / df['上月'].replace(0, pd.NA) * 100).round(1)`，然后用 `fillna('N/A')` 处理
   - 打印前统一替换 inf/-inf 为 "N/A"：`df.replace([np.inf, -np.inf], np.nan, inplace=True)`

## 安全约束（严格遵守，违反会执行失败）

**禁止行为**：
- 禁止 import os.system / subprocess / shutil / socket / requests / urllib
- 禁止使用 eval() / exec() / compile() / __import__()
- 禁止用 open() 写入任何文件（只读模式可以）
- 禁止访问网络、发送请求
- 禁止修改 sys.path 或 os.environ
- 禁止执行系统命令

## 示例

用户数据文件是 sales_q3.csv，分析步骤：按产品线分组统计营收

```python
import os, pandas as pd

data_dir = os.environ['DATA_DIR']
files = [f for f in os.listdir(data_dir) if f.endswith(('.csv','.xlsx'))]
if not files:
    print("ERROR: 数据文件未找到")
else:
    filepath = os.path.join(data_dir, files[0])
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    else:
        df = pd.read_excel(filepath)

    result = df.groupby('产品线')['营收(¥)'].sum().reset_index()
    print(result.to_string())
```

输出这段代码即可，不要加任何前后缀文字。
"""


DATA_UNDERSTANDING_PROMPT = """生成数据概况代码。

需要输出以下三部分：
1. `df.dtypes` — 每列的数据类型
2. `df.describe()` — 数值列的统计摘要
3. `df.isnull().sum()` — 每列的缺失值统计

这是所有分析任务的标准第一步，让后续步骤知道数据长什么样。"""


STEP_PROMPT_TEMPLATE = """用户上传了数据文件，经过数据概况分析后，现在需要执行以下分析步骤：

{step_description}

{data_context}

请生成完成此分析步骤的 Python 代码。"""


FIX_PROMPT_TEMPLATE = """你之前生成的分析代码执行失败了。

**原始分析步骤**：{step_description}

**你生成的代码**：
```python
{failed_code}
```

**执行报错**：
```
{error_message}
```

**可用数据列**：{columns}

**数据预览（前3行）**：
{data_preview}

请修正代码并重新输出。常见错误：
- 列名拼写错误（仔细核对可用数据列）
- 文件路径错误（使用 os.environ['DATA_DIR']）
- 方法名/参数错误
- 数据类型不匹配
- 分组后操作方式不对

**只输出修正后的 Python 代码，不要解释。**"""


DEGRADE_FIX_PROMPT_TEMPLATE = """你之前生成的分析代码经过多次修正仍然失败了。现在请放弃复杂分析，生成一个**简化版本**的代码。

**原始分析步骤**：{step_description}

**上一次报错**：
```
{error_message}
```

**可用数据列**：{columns}

**数据预览（前3行）**：
{data_preview}

**简化策略**：
- 放弃复杂的分组聚合、多维度交叉分析
- 只做 `df.describe()` + 数值列的基本统计（mean/sum/min/max）
- 或对单一维度做 `value_counts()` 取 Top 5
- 只用最简单的 Pandas 操作，确保代码能跑通
- 边界处理：除零→'N/A'，inf→NaN→'N/A'

**只输出简化后的 Python 代码，不要解释。**"""


# ---------------------------------------------------------------------------
# Code builder
# ---------------------------------------------------------------------------

def _build_data_understanding_code() -> str:
    """Generate the mandatory data overview step code."""
    return """import os, pandas as pd

data_dir = os.environ.get('DATA_DIR', '.')
files = [f for f in os.listdir(data_dir) if f.endswith(('.csv','.xlsx'))]

if not files:
    print("ERROR: 数据文件未找到")
else:
    filepath = os.path.join(data_dir, files[0])
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    else:
        df = pd.read_excel(filepath)

    print("=== DTYPES ===")
    print(df.dtypes.to_string())
    print()
    print("=== DESCRIBE ===")
    print(df.describe(include='all').to_string())
    print()
    print("=== NULL COUNTS ===")
    print(df.isnull().sum().to_string())
    print()
    print("=== COLUMNS ===")
    print(list(df.columns))
    print()
    print("=== HEAD (3 rows) ===")
    print(df.head(3).to_string())"""


def _get_data_context(session_dir: str) -> tuple[str, str]:
    """Extract column names and data preview from the session directory.

    Returns:
        (columns_str, data_preview_str)
    """
    session_path = Path(session_dir)
    try:
        files = [f for f in os.listdir(session_dir) if f.endswith(('.csv', '.xlsx'))]
        if not files:
            return "未知", "文件不存在"

        import pandas as pd
        filepath = session_path / files[0]
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8-sig', nrows=3)
        else:
            df = pd.read_excel(filepath, nrows=3)

        columns = list(df.columns)
        preview = df.head(3).to_string()
        return str(columns), preview
    except Exception as e:
        return "解析失败", str(e)


# ---------------------------------------------------------------------------
# Execution with self-healing retry loop
# ---------------------------------------------------------------------------

async def execute_single_step(
    step_description: str,
    session_dir: str,
    data_context: str = "",
    max_retries: int = SANDBOX_MAX_RETRIES,
) -> SandboxResult:
    """Execute a single analysis step with self-healing retry.

    Flow:
        1. LLM generates Python code for the step
        2. Sandbox executes the code
        3. If failed → error + context → LLM fixes → retry (up to max_retries)
        4. On last retry → degradation prompt (fall back to simple describe/value_counts)
        5. Temperature ramp (0.1 → 0.25 → 0.4) for diverse fix attempts

    Args:
        step_description: What this analysis step should do.
        session_dir: Path to the session's uploaded files.
        data_context: Optional context about available columns/data.
        max_retries: Max number of fix-and-retry attempts.

    Returns:
        SandboxResult with success/error/output.
    """
    client = get_llm_client()

    # --- Step 1: Generate initial code ---
    messages = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": STEP_PROMPT_TEMPLATE.format(
            step_description=step_description,
            data_context=data_context or "（数据概况尚未生成）",
        )},
    ]

    code = await client.chat(messages=messages, temperature=0.2, max_tokens=2048, model=MODEL_REASONING)
    code = _clean_code(code)
    logger.info(f"[CodeInterpreter] Generated code ({len(code)} chars)")

    # --- Step 2: Execute + retry loop ---
    result = execute_in_sandbox(code, session_dir)
    result.retry_count = 0

    while not result.success and result.retry_count < max_retries:
        logger.warning(
            f"[CodeInterpreter] Execution failed (retry {result.retry_count + 1}/{max_retries}): "
            f"{result.error[:120]}"
        )

        # Get fresh column context for the fix prompt
        columns, preview = _get_data_context(session_dir)

        # On last retry → use degradation fix prompt (fall back to simpler analysis)
        is_last = (result.retry_count == max_retries - 1)
        template = DEGRADE_FIX_PROMPT_TEMPLATE if is_last else FIX_PROMPT_TEMPLATE

        fix_messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": template.format(
                step_description=step_description,
                failed_code=code,
                error_message=result.error,
                columns=columns,
                data_preview=preview,
            )},
        ]

        # Temperature ramp: 0.1 → 0.25 → 0.4 for diverse fix attempts
        fix_temp = 0.1 + result.retry_count * 0.15
        code = await client.chat(messages=fix_messages, temperature=fix_temp, max_tokens=2048, model=MODEL_REASONING)
        code = _clean_code(code)
        logger.info(f"[CodeInterpreter] Regenerated fix code ({len(code)} chars, temp={fix_temp})")

        result = execute_in_sandbox(code, session_dir)
        result.retry_count += 1

    if result.success:
        logger.info(f"[CodeInterpreter] Step succeeded (retries: {result.retry_count})")
    else:
        logger.error(f"[CodeInterpreter] Step failed after {result.retry_count} retries: {result.error[:200]}")

    return result


async def execute_analysis(
    steps: list[dict[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    """Execute all 'code' type analysis steps through the Code Interpreter.

    This is the main entry point called by the LangGraph code_interpreter_node.

    Args:
        steps: Planner output steps list (step, description, type).
        session_id: Upload session ID for locating files.

    Returns:
        dict with keys:
            success: bool
            output: str (concatenated output from all steps)
            results: list of individual step results
            error: str | None
    """
    session_dir = str(STORAGE_DIR / session_id)

    if not Path(session_dir).exists():
        return {
            "success": False,
            "output": "",
            "results": [],
            "error": f"Session directory not found: {session_dir}",
        }

    # Filter only 'code' type steps
    code_steps = [s for s in steps if s.get("type") == "code"]
    if not code_steps:
        logger.info("[CodeInterpreter] No 'code' type steps — skipping")
        return {"success": True, "output": "", "results": [], "error": None}

    all_outputs: list[str] = []
    all_results: list[dict[str, Any]] = []

    # --- Step 0: Data Understanding (always first, not from planner) ---
    logger.info("[CodeInterpreter] Step 0: Data Understanding")
    du_code = _build_data_understanding_code()
    du_result = execute_in_sandbox(du_code, session_dir)
    du_result.retry_count = 0

    if du_result.success:
        all_outputs.append(f"=== Data Understanding ===\n{du_result.output}")
    else:
        # Data Understanding failure is fatal — we can't proceed without knowing the schema
        logger.error(f"[CodeInterpreter] Data Understanding failed: {du_result.error}")
        return {
            "success": False,
            "output": "",
            "results": [],
            "error": f"数据概况分析失败: {du_result.error}",
        }

    all_results.append({
        "step": 0,
        "description": "数据概况（自动）",
        "success": du_result.success,
        "output": du_result.output[:2000],
        "retries": du_result.retry_count,
    })

    # Build data context from understanding output for subsequent steps
    data_context = f"数据概况结果:\n{du_result.output[:2000]}"

    # --- Execute each planned code step ---
    for step_info in code_steps:
        step_num = step_info.get("step", "?")
        description = step_info.get("description", str(step_info))
        logger.info(f"[CodeInterpreter] Step {step_num}: {description[:80]}")

        result = await execute_single_step(
            step_description=description,
            session_dir=session_dir,
            data_context=data_context,
            max_retries=SANDBOX_MAX_RETRIES,
        )

        step_output = f"=== Step {step_num}: {description} ===\n{result.output}"
        all_outputs.append(step_output)

        all_results.append({
            "step": step_num,
            "description": description,
            "success": result.success,
            "output": result.output[:2000],
            "error": result.error,
            "retries": result.retry_count,
            "duration_ms": result.duration_ms,
        })

        if not result.success:
            logger.warning(f"[CodeInterpreter] Step {step_num} failed: {result.error[:120]}")

    # Compose final output
    combined_output = "\n\n".join(all_outputs)

    # Check if all steps passed
    all_success = all(r["success"] for r in all_results)

    return {
        "success": all_success,
        "output": combined_output,
        "results": all_results,
        "error": None if all_success else "部分分析步骤执行失败，请查看详细结果",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_code(text: str) -> str:
    """Extract Python code from LLM response.

    Removes markdown code fences, leading/trailing whitespace, and
    common LLM verbosity patterns.
    """
    text = text.strip()

    # Remove markdown code fences: ```python ... ```
    if text.startswith("```"):
        # Find first newline after opening fence
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return text
