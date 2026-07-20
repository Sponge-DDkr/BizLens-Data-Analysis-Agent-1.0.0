# BizLens — 智能数据分析助手

> 上传 Excel/CSV，打字提问，拿到带图表的分析报告。让不会写代码的业务人员自己做数据分析。

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed)](https://docker.com)

---

## 📺 项目介绍视频

[![BizLens 项目介绍](https://img.shields.io/badge/Bilibili-项目介绍视频-00A1D6)](https://space.bilibili.com/494206751)

---

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url>
cd DataAgent

# 2. 配置 DeepSeek API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-your-key

# 3. 一键启动
docker-compose up
```

浏览器打开 `http://localhost`，上传数据文件，输入分析问题。

---

## 这是什么

BizLens 是一个**多 Agent 协作的自动化数据分析系统**。上传 Excel/CSV 文件，用自然语言描述分析需求，系统自动完成：

- **拆解分析步骤**（Planner Agent — DeepSeek-R1）
- **生成并执行 Python 代码**（Code Interpreter Agent — DeepSeek-R1 + Sandbox）
- **自动选择图表类型并生成可视化**（Visualization Agent — DeepSeek-R1 + Plotly）
- **生成五段式分析报告**（Insight Agent — DeepSeek-V3 + MCP 知识注入）

4 个 Agent 串行协作，全程 SSE 流式推送进度，最终输出可交互的 Plotly 图表 + Markdown 分析报告。

### 五段式报告

| 章节 | 内容 |
|------|------|
| 数据事实 | 真实计算结果的数字、表格、Top N |
| 图表说明 | 图表类型选择原因、关键趋势 |
| 业务解读 | MCP 知识库归因（有来源标注）或标注「推测」 |
| 总结与建议 | 直接回答用户问题 + 关键数字支撑 + 行动建议（≤150字） |
| 数据来源 | 数据文件 + 知识库引用文档 |

---

## 技术架构

```
React 19 + TypeScript + Vite 8 (Nginx)
    ↓  HTTP + SSE
FastAPI (Uvicorn)
    ↓  LangGraph StateGraph
Planner → Code Interpreter → Visualization → Insight
  R1           R1                 R1             V3
   ↓            ↓                  ↓              ↓
DeepSeek    Sandbox             Plotly         MCP Client
 API      (subprocess / Docker)  (JSON→前端)  (streamable HTTP)
                                                  ↓
                                          MCP Knowledge Server
                                         (独立 HTTP 服务 :8001)
                                         answer_with_citation
```

**模型分层策略**：代码/逻辑 Agent 使用 DeepSeek-R1（推理强、代码准确率高），报告 Agent 使用 DeepSeek-V3（中文流畅、商业术语地道）。各取所长。

---

## 项目结构

```
DataAgent
├── frontend/                     # React 19 + TypeScript + Vite 8
│   └── src/components/           # FileUpload / AnalysisInput / ReportView / ProgressBar
├── backend/                      # FastAPI
│   ├── main.py                   # API 路由：/upload、/analyze(SSE)、/knowledge/*
│   ├── config.py                  # 环境变量 + 模型分层配置
│   └── llm_client.py             # DeepSeek API 客户端（支持 per-call model override）
├── agents/
│   ├── graph.py                  # LangGraph StateGraph 4 节点 DAG
│   ├── planner.py                # 任务拆解（Pydantic 结构化输出）
│   ├── code_interpreter.py       # 代码生成 + Step 0 数据理解 + 3 次错误自愈（温度递增 + 降级兜底）
│   ├── visualization.py          # Plotly 图表自动选择 + 3 次修正循环 + 降级兜底
│   └── insight.py                # 五段式报告生成 + MCP 知识注入 + 双重降级
├── sandbox/
│   ├── executor.py               # 双模式切换（subprocess / Docker）
│   ├── subprocess_executor.py    # 开发用 Subprocess 执行器
│   ├── docker_executor.py        # Docker 5 层安全约束
│   └── Dockerfile                # bizlens-sandbox 镜像
├── mcp_client/
│   └── knowledge_client.py       # MCP streamable HTTP 持久化客户端 + query_knowledge()
├── mcp_knowledge_agent/          # MCP Knowledge Server 源码（8 工具 / HTTP 传输）
├── knowledge_base/               # 检索知识库 .md 文件（9 份）
├── storage/                      # 用户上传文件（session_id 隔离）
├── tests/
├── docker-compose.yml            # 三服务编排（MCP :8001 + Backend :8000 + Frontend :80）
├── Dockerfile.mcp                # MCP Knowledge Server 独立镜像
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
├── pyproject.toml                # mcp-knowledge-agent 包定义
├── .env.example
└── README.md
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 19 · TypeScript · Vite 8 · Plotly.js · react-markdown |
| 后端 | FastAPI · Uvicorn · SSE (sse-starlette) |
| Agent 框架 | LangGraph StateGraph (4 节点 DAG) |
| LLM | DeepSeek-R1（推理） + DeepSeek-V3（报告） |
| 数据处理 | Pandas · openpyxl |
| 可视化 | Plotly (go.Figure → JSON → 前端渲染) |
| 沙箱 | Subprocess (dev) / Docker 5 层约束 (prod) |
| 知识库 | MCP Knowledge Server · ChromaDB · sentence-transformers · BGE-Reranker · streamable HTTP |
| 部署 | Docker Compose (mcp-knowledge + backend + frontend 三服务) |

---

## 核心亮点

- **多 Agent 协作架构**：LangGraph StateGraph 构建 4 Agent 串行 DAG，每个 Agent 有独立 State 和工具集，比单 LLM 调用更精准可控
- **Per-Agent 模型分层**：代码 Agent 用 R1、报告 Agent 用 V3——不同任务选不同模型，体现选型判断力
- **错误自愈机制**：LLM 生成的代码出错时，自动注入列名+数据上下文修复（两类代码 Agent 统一 3 次重试，温度递增，最后一次降级到最简可行方案保底产出）
- **MCP 知识注入**：通过 MCP 协议消费独立 Knowledge Server，Insight 报告带来源标注的业务归因
- **五段式报告**：数据事实/图表说明/业务解读/总结与建议/数据来源——区分事实与解读，防幻觉关键设计
- **三项目能力矩阵**：ARC（知识系统）→ MCP Server（协议抽象）→ BizLens（数据系统+协议消费），互相配合覆盖企业 AI 全场景

---

## 示例数据

### 分析测试数据

`Project_Review/数据分析测试用例/` 下提供 4 份测试数据（971-1000 行不等）：

| 文件 | 内容 | 推荐测试问题 |
|------|------|-------------|
| `sales_q3_2024.xlsx` | SaaS 销售明细 | "Q3 各产品线营收对比" |
| `channel_roi_2024h1.xlsx` | 渠道投放 ROI | "各渠道转化漏斗对比" |
| `product_usage_analytics.xlsx` | 用户行为数据 | "付费和免费用户有什么不同" |
| `startup_finance_2024.xlsx` | 创业公司财务 | "这份财报健康吗" |

详细测试问题见 `Project_Review/数据分析测试用例/测试问题.md`（含 20 道题、图表预期、Agent 链路）。

### 检索知识库

`Project_Review/检索知识库/` 下提供 9 份行业知识文档，覆盖 SaaS 运营、数字广告、创业融资、云计算、产品运营等领域，与测试数据主题一一对应。Insight Agent 通过 MCP Knowledge Server 检索这些文档注入业务归因。

详见 [Project_Review/检索知识库/](Project_Review/检索知识库/)

---

## License

MIT
