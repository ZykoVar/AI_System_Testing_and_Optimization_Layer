"""统一 LLM 客户端抽象层：mock / OpenAI 兼容 / Anthropic。"""
# 聚合导出：测试用例只需 “from llmqa.clients import ...” 即可取用全部客户端相关类型。
from llmqa.clients.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from llmqa.clients.factory import ClientPool, build_client, scripted_or_real
from llmqa.clients.mock import DEFAULT_REFUSAL, MockClient, MockRule

__all__ = [
    "LLMClient", "LLMError", "LLMResponse", "Message", "StreamChunk",
    "TokenUsage", "ToolCall", "ClientPool", "build_client", "scripted_or_real",
    "MockClient", "MockRule", "DEFAULT_REFUSAL",
]
