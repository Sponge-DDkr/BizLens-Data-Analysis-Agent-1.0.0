"""MCP Knowledge Agent Server — python -m 入口

两种传输模式（通过 --transport 参数切换）：

  stdio              → python -m mcp_knowledge_agent
  streamable-http    → python -m mcp_knowledge_agent --transport streamable-http --port 8001

环境变量 MCP_TRANSPORT / FASTMCP_HOST / FASTMCP_PORT 也可用，
但命令行参数优先级更高（推荐直接用参数，避免透传问题）。
"""

import argparse
import os
import sys as _sys

# Force offline mode — model must be pre-cached at ~/.cache/huggingface/hub/
# Prevents network timeout hang when huggingface.co is unreachable.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    """启动 MCP Knowledge Agent Server"""
    parser = argparse.ArgumentParser(
        description="MCP Knowledge Agent Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m mcp_knowledge_agent                                        # stdio\n"
            "  python -m mcp_knowledge_agent --transport streamable-http --port 8001 # HTTP\n"
        ),
    )
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=("stdio", "streamable-http"),
        help="Transport protocol (default: stdio, or $MCP_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FASTMCP_HOST", "0.0.0.0"),
        help="Host for HTTP mode (default: 0.0.0.0, or $FASTMCP_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FASTMCP_PORT", "8001")),
        help="Port for HTTP mode (default: 8001, or $FASTMCP_PORT)",
    )
    args = parser.parse_args()

    # Push to env so FastMCP picks them up via its Settings (FASTMCP_ prefix)
    os.environ["FASTMCP_HOST"] = args.host
    os.environ["FASTMCP_PORT"] = str(args.port)

    _sys.stderr.write(
        f"[MCP] Starting Knowledge Server (transport={args.transport}, "
        f"host={args.host}, port={args.port})...\n"
    )
    _sys.stderr.flush()

    from mcp_knowledge_agent.server import create_server

    server = create_server(host=args.host, port=args.port)

    _sys.stderr.write(f"[MCP] Server created, entering {args.transport} loop.\n")
    _sys.stderr.flush()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
