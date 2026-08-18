"""RAG 专项测试套件：检索质量、分块、忠实性、相关性、引用、端到端。

通过导入各子模块触发 @test 装饰器注册用例（用例发现依赖此导入）。
"""
from llmqa.suites.rag import (  # noqa: F401
    chunking_quality,
    citation_quality,
    faithfulness,
    pipeline_e2e,
    relevance,
    retrieval_quality,
)

__all__ = [
    "retrieval_quality",
    "chunking_quality",
    "faithfulness",
    "relevance",
    "citation_quality",
    "pipeline_e2e",
]
