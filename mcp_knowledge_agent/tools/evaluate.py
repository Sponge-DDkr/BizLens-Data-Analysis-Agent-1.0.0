"""evaluate_retrieval — 基于内置测试用例评估检索质量

输出 Recall@5、MRR、按 8 个类别分项指标、未命中列表。
每次调参后跑一次，量化效果变化。
"""

from mcp.server.fastmcp import FastMCP

from mcp_knowledge_agent.core.vector_pipeline import HybridSearchPipeline


def register_evaluate_tool(mcp: FastMCP, pipeline: HybridSearchPipeline):
    """注册 evaluate_retrieval — 检索质量评估"""

    @mcp.tool(description=(
        "基于内置测试用例评估检索质量。"
        "返回 MRR、Recall@5，按类别分项。"
        "每次调参后跑一次，量化效果变化。"
    ))
    async def evaluate_retrieval() -> dict:
        result = await pipeline.evaluate()
        return result
