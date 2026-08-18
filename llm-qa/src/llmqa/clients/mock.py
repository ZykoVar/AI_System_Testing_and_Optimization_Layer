"""确定性 Mock 客户端。

MockClient 依据规则脚本返回确定性回复，用于：
1. CI 冒烟测试 —— 无需 API Key，离线可跑，结果可复现；
2. 单元/集成测试 —— 为被测的 RAG/Agent/安全逻辑构造"理想模型"；
3. 故障注入 —— 通过 error 规则模拟 429 限流、500 故障等。

规则优先级：按注册顺序第一个命中者生效；times 用尽后跳过。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from llmqa.clients.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

DEFAULT_REFUSAL = "抱歉，我无法协助这个请求。"


class MockRule(BaseModel):
    """一条命中规则。reply 可为：
    - 字符串：直接作为回复文本；
    - 字典：{"content": str, "tool_calls": [{name, arguments, id?}]}；
    - 字典：{"refusal": true} 输出标准拒绝话术。
    error 非空时模拟 Provider 异常（如 {"status": 429, "message": "rate limited"}）。
    """
    match: str = ".*"                       # 正则；默认匹配"最后一条用户消息"
    reply: Any = ""
    error: dict[str, Any] | None = None
    times: int | None = None                # 最多命中次数，None 为无限
    latency_ms: float = 2.0
    match_transcript: bool = False          # True 时对整个会话转录匹配

    _used: int = 0


class MockClient(LLMClient):
    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        *,
        default_reply: str = "Mock 默认回复。",
        rules: list[MockRule] | None = None,
        default_latency_ms: float = 2.0,
    ):
        super().__init__(name, model)
        self.default_reply = default_reply
        self.default_latency_ms = default_latency_ms
        self.rules: list[MockRule] = list(rules or [])

    def add_rule(self, rule: MockRule | dict) -> MockRule:
        if isinstance(rule, dict):
            rule = MockRule(**rule)
        self.rules.append(rule)
        return rule

    @staticmethod
    def format_transcript(messages: list[Message]) -> str:
        lines: list[str] = []
        for m in messages:
            head = f"[{m.role}] {m.content}".strip()
            if m.tool_calls:
                for tc in m.tool_calls:
                    head += f"\n  TOOL_CALL {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
            lines.append(head)
        return "\n".join(lines)

    def _find_rule(self, messages: list[Message]) -> MockRule | None:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        transcript = self.format_transcript(messages)
        for rule in self.rules:
            if rule.times is not None and rule._used >= rule.times:
                continue
            target = transcript if rule.match_transcript else last_user
            if re.search(rule.match, target, re.IGNORECASE | re.DOTALL):
                rule._used += 1
                return rule
        return None

    def _apply_reply(self, rule: MockRule | None) -> tuple[str, list[ToolCall], dict[str, Any]]:
        if rule is None:
            return self.default_reply, [], {}
        if rule.error:
            raise LLMError(self.name, str(rule.error.get("message", "mock error")),
                           status=rule.error.get("status"))
        reply = rule.reply
        if isinstance(reply, str):
            return reply, [], {}
        if isinstance(reply, dict):
            if reply.get("refusal"):
                return DEFAULT_REFUSAL, [], {}
            tool_calls = [ToolCall(id=tc.get("id") or f"mock-{i}", name=tc["name"],
                                   arguments=tc.get("arguments") or {})
                          for i, tc in enumerate(reply.get("tool_calls") or [])]
            return reply.get("content", "") or "", tool_calls, {}
        return str(reply), [], {}

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
        start = time.perf_counter()
        rule = self._find_rule(messages)
        latency = (rule.latency_ms if rule else self.default_latency_ms) / 1000.0
        await self._sleep(latency)
        text, tool_calls, _ = self._apply_reply(rule)
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=TokenUsage(
                prompt_tokens=sum(self.estimate_tokens(m.content) for m in messages),
                completion_tokens=self.estimate_tokens(text),
            ),
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
            model=self.model,
        )

    async def stream(self, messages: list[Message], *, temperature: float = 0.0,
                     max_tokens: int = 512, stop: list[str] | None = None):
        resp = await self.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
        words = resp.text.split(" ")
        for i in range(0, len(words), 3):
            yield StreamChunk(text_delta=" ".join(words[i:i + 3]) + (" " if i + 3 < len(words) else ""))
        yield StreamChunk(finish_reason=resp.finish_reason)

    async def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        import asyncio
        await asyncio.sleep(seconds)
