"""Docker Sandbox Executor — 5-layer security-constrained code execution.

Primary mode: Docker container with ro mount, CPU/memory limits, network=none, timeout.
Fallback mode: local subprocess (for dev environments without Docker).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "bizlens-sandbox:latest")
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))
SANDBOX_CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "100000"))  # 1 core
SANDBOX_MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "512m")
SANDBOX_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Result from sandbox code execution."""

    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    retry_count: int = 0
    mode: str = "local"  # "docker" | "local"


# ---------------------------------------------------------------------------
# Docker availability check
# ---------------------------------------------------------------------------

_docker_available: Optional[bool] = None


def is_docker_available() -> bool:
    """Check whether Docker is installed and the daemon is reachable."""
    global _docker_available
    if _docker_available is not None:
        return _docker_available

    if not shutil.which("docker"):
        _docker_available = False
        return False

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        _docker_available = result.returncode == 0
    except Exception:
        _docker_available = False

    return _docker_available


# ---------------------------------------------------------------------------
# Docker executor — 5-layer security
# ---------------------------------------------------------------------------

def _execute_docker(code: str, session_dir: str, timeout: int = SANDBOX_TIMEOUT) -> SandboxResult:
    """Execute Python code inside a Docker container with 5-layer constraints.

    Layer 1: Read-only data mount     → /data/input : ro
    Layer 2: CPU quota 100000 (1 core) → cpu_quota
    Layer 3: Memory limit 512MB        → mem_limit
    Layer 4: Network disabled          → network_mode="none"
    Layer 5: Timeout 30s auto-kill     → container.wait(timeout=N)
    """
    session_path = Path(session_dir).resolve()

    if not session_path.exists():
        return SandboxResult(success=False, error=f"Session dir not found: {session_dir}")

    docker_cmd = [
        "docker", "run",
        "--rm",                          # auto-remove after execution
        "--network", "none",             # Layer 4: no network
        "--cpu-quota", str(SANDBOX_CPU_QUOTA),  # Layer 2: 1 CPU core
        "--memory", SANDBOX_MEM_LIMIT,   # Layer 3: 512 MB
        "--memory-swap", SANDBOX_MEM_LIMIT,  # no swap
        "--read-only",                   # filesystem read-only by default
        "--tmpfs", "/tmp:rw,noexec,size=64m",  # writable /tmp only
        "--tmpfs", "/data/output:rw,noexec,size=128m",
        "-v", f"{session_path}:/data/input:ro",  # Layer 1: read-only data
        "-w", "/data",
        SANDBOX_IMAGE,
        code,
    ]

    logger.info(f"[DockerSandbox] Executing code ({len(code)} chars) in container...")
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 5,  # Layer 5: +5s grace for docker itself
        )

        duration_ms = (time.perf_counter() - t0) * 1000
        output = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            # Parse meaningful error from stderr
            error_lines = [line for line in stderr.split("\n") if line.strip()]
            # Filter Docker daemon noise
            meaningful = [l for l in error_lines if not l.startswith("WARNING")]
            error_msg = "\n".join(meaningful[:20]) if meaningful else f"Exit code {proc.returncode}"

            return SandboxResult(
                success=False,
                output=output[:2000],
                error=error_msg[:1000],
                duration_ms=duration_ms,
                mode="docker",
            )

        return SandboxResult(
            success=True,
            output=output[:5000],
            error="",
            duration_ms=duration_ms,
            mode="docker",
        )

    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - t0) * 1000
        return SandboxResult(
            success=False,
            error=f"Sandbox execution timed out after {timeout}s (Layer 5)",
            duration_ms=duration_ms,
            mode="docker",
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        return SandboxResult(
            success=False,
            error=f"Docker execution failed: {str(e)}",
            duration_ms=duration_ms,
            mode="docker",
        )


# ---------------------------------------------------------------------------
# Local subprocess executor — dev fallback
# ---------------------------------------------------------------------------

def _execute_local(code: str, session_dir: str, timeout: int = SANDBOX_TIMEOUT) -> SandboxResult:
    """Execute Python code in a local subprocess.

    Security note: This is a DEVELOPMENT fallback. No real isolation — LLM-generated
    code runs in the host Python process. Use Docker mode for production.
    """
    session_path = Path(session_dir).resolve()

    if not session_path.exists():
        return SandboxResult(success=False, error=f"Session dir not found: {session_dir}")

    # Build a wrapper script that:
    # 1. Changes to the session dir so relative paths work
    # 2. Injects DATA_DIR env var so code can find the CSV
    # 3. Executes the user's code
    wrapper = f"""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['DATA_DIR'] = {str(session_path)!r}
os.chdir({str(session_path)!r})

# ---- USER CODE ----
{code}
"""

    logger.info(f"[LocalSandbox] Executing code ({len(code)} chars) via subprocess...")
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(session_path),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "DATA_DIR": str(session_path),
            },
        )

        duration_ms = (time.perf_counter() - t0) * 1000
        output = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            return SandboxResult(
                success=False,
                output=output[:2000],
                error=stderr[:1000] or f"Exit code {proc.returncode}",
                duration_ms=duration_ms,
                mode="local",
            )

        return SandboxResult(
            success=True,
            output=output[:5000],
            error="",
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_in_sandbox(
    code: str,
    session_dir: str,
    timeout: int = SANDBOX_TIMEOUT,
    prefer_docker: bool = True,
) -> SandboxResult:
    """Execute Python code in a sandboxed environment.

    Uses Docker with 5-layer security if available, otherwise falls back to
    local subprocess execution for development.

    Args:
        code: Python source code to execute.
        session_dir: Path to the session's uploaded files (mounted at /data/input in Docker).
        timeout: Max execution time in seconds.
        prefer_docker: If True, try Docker first; fall back to local.

    Returns:
        SandboxResult with success/error/output/duration.
    """
    use_docker = prefer_docker and is_docker_available()

    if use_docker:
        logger.info("[Sandbox] Using Docker executor (5-layer security)")
        return _execute_docker(code, session_dir, timeout)
    else:
        logger.info("[Sandbox] Docker unavailable — using local subprocess (dev mode)")
        return _execute_local(code, session_dir, timeout)
