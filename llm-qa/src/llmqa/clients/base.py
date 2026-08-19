"""LLM 客户端统一抽象。

所有 Provider 适配器与 Mock 均实现 LLMClient 接口；
测试用例只依赖该接口，从而可在 mock / 云端 API / 自托管模型之间无缝切换。
"""
from __future__ import annotations

import abc
import time
import uuid
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """模型发起的一次工具调用。"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])  # 截断 UUID 以缩短请求体，碰撞风险可接受
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """OpenAI 风格的多轮对话消息。"""
    role: str                                    # system | user | assistant | tool
    content: str = ""
    name: str | None = None                      # 工具名，仅 role=tool 时有意义
    tool_calls: list[ToolCall] | None = None     # 仅 role=assistant 且模型选择调用工具时非空
    tool_call_id: str | None = None              # 关联上一条 assistant 工具调用，仅 role=tool 时使用

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


class TokenUsage(BaseModel):
    """一次调用的 token 用量；兼容 OpenAI 风格字段命名。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    """一次生成调用的完整结果。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None      # stop | tool_calls | length 等，由各 Provider 定义
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0               # 端到端耗时，性能测试据此统计 P95/P99
    cost_usd: float | None = None         # None 表示无法估算（如未配置 pricing）
    model: str | None = None              # Provider 实际返回的模型名，可能因别名重写
    raw: dict[str, Any] = Field(default_factory=dict)  # Provider 原始响应，供调试与审计


class StreamChunk(BaseModel):
    """流式输出的单个增量。"""
    text_delta: str = ""
    finish_reason: str | None = None


class LLMError(RuntimeError):
    """Provider 层错误（网络、鉴权、限流等）。status 为 HTTP 状态码。"""

    def __init__(self, provider: str, message: str, status: int | None = None):
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] {message}")


class LLMClient(abc.ABC):
    """所有 Provider 的统一接口。"""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.model!r}>"

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """同步（非流式）生成。传入 tools 时结果中的 tool_calls 为模型选择的工具调用。"""
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式生成。默认实现按词切分 generate 结果，适配器可覆盖为真流式。"""
        resp = await self.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
        words = resp.text.split(" ")
        # 默认实现非真流式：先完整生成，再按每 8 词切块模拟增量，避免依赖具体分词器。
        for i in range(0, len(words), 8):
            chunk = " ".join(words[i:i + 8])
            yield StreamChunk(text_delta=chunk + (" " if i + 8 < len(words) else ""))
        yield StreamChunk(finish_reason=resp.finish_reason)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """无 tokenizer 时的粗略估算：约 4 字符/token。"""
        return max(1, (len(text) + 3) // 4)
