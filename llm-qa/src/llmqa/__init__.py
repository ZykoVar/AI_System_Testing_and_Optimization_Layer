"""llmqa —— 企业级 LLM/Agent 质量保障测试框架。

六大子系统：Prompt 管理 / LLM 质量 / RAG 专项 / Agent / 安全红队 / 性能。
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

__version__ = "0.1.0"

__all__ = [
    "__version__", "Settings", "Severity", "Verdict", "SkipTest",
    "TestContext", "TestOutcome", "test", "discover",
    "LLMClient", "LLMError", "LLMResponse", "Message", "ToolCall", "TokenUsage",
    "MockClient", "MockRule",
    "AssertionFailed", "Judge", "assert_contains", "assert_not_contains",
    "assert_similarity", "assert_refusal", "assert_json_schema", "assert_in_language",
]
