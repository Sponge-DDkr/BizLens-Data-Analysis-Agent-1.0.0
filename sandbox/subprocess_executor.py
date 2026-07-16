"""Subprocess Executor — lightweight code execution for dev/demo.

No Docker required. Runs LLM-generated Python in a subprocess with basic isolation
(temp directory + explicit DATA_DIR env var).

面试说法：
    "Demo 阶段用 subprocess + tempfile 隔离快速迭代，
     生产环境切换到 Docker Sandbox——只读挂载数据卷、CPU/内存限制、
     禁网络、30 秒超时。接口层统一，切换只需改一个环境变量。"
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time

from sandbox.docker_executor import SandboxResult

logger = logging.getLogger(__name__)

SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))


class SubprocessExecutor:
    """Execute Python code in a local subprocess.

    Security note:
        This is a DEVELOPMENT executor. No real OS-level isolation — LLM-generated
        code runs in the host Python process. For production, set BIZLENS_USE_DOCKER=true
        to activate the Docker sandbox with 5-layer security constraints.
    """

    def execute(
        self,
        code: str,
        session_dir: str,
        timeout: int = SANDBOX_TIMEOUT,
    ) -> SandboxResult:
        """Execute code with access to data files in session_dir.

        The script is written to a temp directory for basic filesystem isolation.
        DATA_DIR env var points to session_dir so the script can locate CSV/Excel files.
        """
        session_path = os.path.abspath(session_dir)

        if not os.path.exists(session_path):
            return SandboxResult(
                success=False,
                error=f"Session dir not found: {session_dir}",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            logger.info(
                "[SubprocessExecutor] Executing code (%d chars) in temp dir...",
                len(code),
            )
            t0 = time.perf_counter()

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=tmpdir,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONIOENCODING": "utf-8",
                        "DATA_DIR": session_path,
                    },
                )

                duration_ms = (time.perf_counter() - t0) * 1000

                if proc.returncode != 0:
                    return SandboxResult(
                        success=False,
                        output=proc.stdout[:2000],
                        error=proc.stderr[:1000] or f"Exit code {proc.returncode}",
                        duration_ms=duration_ms,
                        mode="local",
                    )

                return SandboxResult(
                    success=True,
                    output=proc.stdout[:50000],  # 50KB — Plotly JSON can be large
                    duration_ms=duration_ms,
                    mode="local",
                )

            except subprocess.TimeoutExpired:
                duration_ms = (time.perf_counter() - t0) * 1000
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {timeout}s",
                    duration_ms=duration_ms,
                    mode="local",
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                return SandboxResult(
                    success=False,
                    error=f"Execution error: {str(e)}",
                    duration_ms=duration_ms,
                    mode="local",
                )
