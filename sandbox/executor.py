"""Unified Sandbox Executor — switches between Docker and Subprocess modes.

开发阶段: SubprocessExecutor (默认, BIZLENS_USE_DOCKER 未设置)
生产环境: DockerExecutor  (设置 BIZLENS_USE_DOCKER=true)

切换只需改一个环境变量，接口层完全一致。
"""

from __future__ import annotations

import logging
import os

from sandbox.docker_executor import SandboxResult, is_docker_available

logger = logging.getLogger(__name__)

SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Executor factory
# ---------------------------------------------------------------------------

def get_executor():
    """Return the appropriate executor based on BIZLENS_USE_DOCKER env var.

    BIZLENS_USE_DOCKER=true  → DockerExecutor (5-layer security)
    otherwise                → SubprocessExecutor (dev/demo)
    """
    if os.getenv("BIZLENS_USE_DOCKER"):
        from sandbox.docker_executor import execute_in_sandbox as _docker_exec

        class _DockerExecutor:
            """Thin wrapper that matches SubprocessExecutor interface."""

            def execute(self, code: str, session_dir: str, timeout: int = SANDBOX_TIMEOUT) -> SandboxResult:
                return _docker_exec(code, session_dir, timeout, prefer_docker=True)

        logger.info("[Executor] Using Docker executor (BIZLENS_USE_DOCKER=true)")
        return _DockerExecutor()
    else:
        from sandbox.subprocess_executor import SubprocessExecutor

        logger.info("[Executor] Using Subprocess executor (dev/demo mode)")
        return SubprocessExecutor()


# ---------------------------------------------------------------------------
# Convenience function — mirrors the old docker_executor.execute_in_sandbox signature
# ---------------------------------------------------------------------------

def execute_in_sandbox(
    code: str,
    session_dir: str,
    timeout: int = SANDBOX_TIMEOUT,
) -> SandboxResult:
    """Execute Python code in a sandboxed environment.

    Backend selection:
        - BIZLENS_USE_DOCKER=true → Docker with 5-layer security
        - default → subprocess + tempfile (dev/demo)

    Args:
        code: Python source code to execute.
        session_dir: Path to the session's uploaded files.
        timeout: Max execution time in seconds.

    Returns:
        SandboxResult with success/error/output/duration/mode.
    """
    return get_executor().execute(code, session_dir, timeout)
