"""Subprocess Executor — lightweight code execution for development.

No Docker required. Runs LLM-generated Python in a subprocess with basic isolation
(temp directory + explicit DATA_DIR env var).

For production, set BIZLENS_USE_DOCKER=true to switch to Docker Sandbox with
read-only mounts, CPU/memory limits, network isolation, and 30s timeout.
Unified interface via get_executor() factory — switch with one env var.
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
