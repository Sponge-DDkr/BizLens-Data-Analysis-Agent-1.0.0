"""BizLens Backend Configuration"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Storage
STORAGE_DIR = ROOT_DIR / "storage"
REPORTS_DIR = ROOT_DIR / "reports"
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx", ".csv"}

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Per-Agent model selection (Day 5 optimization — 2026-07-16)
# Code/Logic agents (Planner, Code Interpreter, Visualization) → deepseek-reasoner (R1)
#   - R1 excels at precise code generation, structured JSON output, and logic reasoning
#   - Fixes the "V3 generates identical broken code on retry" problem
# Report agent (Insight) → default deepseek-chat (V3)
#   - V3 produces more fluent, natural Chinese business writing
# Override via env: BIZLENS_REASONING_MODEL
MODEL_REASONING = os.getenv("BIZLENS_REASONING_MODEL", "deepseek-reasoner")

# Sandbox
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "bizlens-sandbox:latest")
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))
SANDBOX_CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "100000"))  # 1 core
SANDBOX_MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "512m")
SANDBOX_MAX_RETRIES = 3

# MCP Knowledge Server
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", "").split(",") if os.getenv("MCP_SERVER_ARGS") else []

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
