"""llmqa —— 企业级 LLM/Agent 质量保障测试框架。

六大子系统：Prompt 管理 / LLM 质量 / RAG 专项 / Agent / 安全红队 / 性能。

包级扁平化导出：把各子系统常用符号集中到 llmqa 顶层，用户 `from llmqa import ...` 即可。
"""
from llmqa.assertors import (
    AssertionFailed,
    Judge,
    assert_contains,
    assert_in_language,
    assert_json_schema,
    assert_not_contains,
    assert_refusal,
    assert_similarity,
)
from llmqa.clients import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    MockClient,
    MockRule,
    TokenUsage,
    ToolCall,
)
from llmqa.config import Settings
from llmqa.core import (
    Severity,
    SkipTest,
    TestContext,
    TestOutcome,
    Verdict,
    discover,
    test,
)

# 单一版本号来源，发布时递增
__version__ = "0.1.0"

# 显式声明公共 API，避免 `from llmqa import *` 引入内部符号
__all__ = [
    "AssertionFailed",
    "Judge",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "MockClient",
    "MockRule",
    "Settings",
    "Severity",
    "SkipTest",
    "TestContext",
    "TestOutcome",
    "TokenUsage",
    "ToolCall",
    "Verdict",
    "__version__",
    "assert_contains",
    "assert_in_language",
    "assert_json_schema",
    "assert_not_contains",
    "assert_refusal",
    "assert_similarity",
    "discover",
    "test",
]
