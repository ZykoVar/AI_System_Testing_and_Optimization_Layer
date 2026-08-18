"""Anthropic Messages API 适配器（最小实现）。

注意：当前版本不支持 function calling（tools 参数会抛错）；
Agent/工具调用类测试请使用 OpenAI 兼容 Provider 或 Mock。
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from llmqa.clients.base import LLMClient, LLMError, LLMResponse, Message, TokenUsage


class AnthropicClient(LLMClient):
    def __init__(self, name: str, model: str, *, api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com",
                 timeout_seconds: float = 60.0, max_retries: int = 2):
        super().__init__(name, model)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": api_key or "",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, messages: list[Message], *, temperature: float = 0.0,
                       max_tokens: int = 512, stop: list[str] | None = None,
                       tools: list[dict[str, Any]] | None = None,
                       tool_choice: str | None = None) -> LLMResponse:
        if tools:
            raise LLMError(self.name, "AnthropicClient 暂不支持 function calling，请换用 OpenAI 兼容 Provider")
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages if m.role in ("user", "assistant")],
        }
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop
        start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/v1/messages", json=payload)
                if resp.status_code >= 400:
                    raise LLMError(self.name, f"HTTP {resp.status_code}: {resp.text[:300]}",
                                   status=resp.status_code)
                data = resp.json()
                content = "".join(b.get("text", "") for b in data.get("content") or [])
                usage_raw = data.get("usage") or {}
                return LLMResponse(
                    text=content,
                    finish_reason=data.get("stop_reason"),
                    usage=TokenUsage(prompt_tokens=usage_raw.get("input_tokens", 0),
                                     completion_tokens=usage_raw.get("output_tokens", 0)),
                    latency_ms=(time.perf_counter() - start) * 1000,
                    cost_usd=None,
                    model=data.get("model") or self.model,
                    raw=data,
                )
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise LLMError(self.name, f"请求失败: {last_exc}")

    async def aclose(self) -> None:
        await self._client.aclose()
