"""Sandbox Execution Layer — dual-mode: subprocess (dev) + Docker (production)

Usage:
    from sandbox import execute_in_sandbox, get_executor

    # Default: subprocess executor (dev/demo)
    result = execute_in_sandbox(code, session_dir)

    # Explicit executor selection
    executor = get_executor()  # SubprocessExecutor or DockerExecutor
    result = executor.execute(code, session_dir)

Switch to Docker:
    set BIZLENS_USE_DOCKER=true
"""

from sandbox.executor import execute_in_sandbox, get_executor
from sandbox.docker_executor import SandboxResult, is_docker_available

__all__ = [
    "execute_in_sandbox",
    "get_executor",
    "SandboxResult",
    "is_docker_available",
]
