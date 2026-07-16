"""MCP Knowledge Agent Server — python -m 入口"""

import os
import sys as _sys

# Force offline mode — model must be pre-cached at ~/.cache/huggingface/hub/
# Prevents network timeout hang when huggingface.co is unreachable.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from mcp_knowledge_agent.server import create_server


def main():
    """stdio 模式启动 MCP Knowledge Agent Server"""
    _sys.stderr.write("[MCP] Starting Knowledge Server (pre-loading models)...\n")
    _sys.stderr.flush()
    server = create_server()
    _sys.stderr.write("[MCP] Server created, entering stdio loop.\n")
    _sys.stderr.flush()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
