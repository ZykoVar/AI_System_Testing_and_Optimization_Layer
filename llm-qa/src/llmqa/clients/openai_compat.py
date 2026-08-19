"""OpenAI 兼容协议适配器（OpenAI / Azure / vLLM / Ollama / DeepSeek 等）。

支持：
- 非流式 chat/completions（含 function calling 工具）；
- 流式 SSE 输出（性能测试测首 token 延迟 TTFT 用）；
- 基于 usage 的粗粒度成本估算（需在 providers.yaml 配置 pricing）。
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from llmqa.clients.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from llmqa.config import Pricing


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    """把内部 Message 列表转换为 OpenAI chat/completions 的 messages 结构。"""
    out = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.name:
            item["name"] = m.name
        if m.tool_calls:
            # OpenAI 协议要求工具参数为 JSON 字符串；ensure_ascii=False 保留中文可读性。
            item["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        out.append(item)
    return out


class OpenAICompatClient(LLMClient):
    """OpenAI chat/completions 协议客户端，兼容各 OpenAI 风格端点。"""
    def __init__(self, name: str, model: str, base_url: str, *,
                 api_key: str | None = None, timeout_seconds: float = 60.0,
                 max_retries: int = 2, extra_headers: dict[str, str] | None = None,
                 pricing: Pricing | None = None):
        super().__init__(name, model)
        # 去掉尾部斜杠，避免拼接路径时出现 “//”，与 httpx base_url 组合更稳妥。
        self.base_url = base_url.rstrip("/")
        self.pricing = pricing or Pricing()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # 用户自定义头（如额外鉴权、租户标识）最后覆盖，优先级最高。
        headers.update(extra_headers or {})
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers,
            timeout=httpx.Timeout(timeout_seconds), follow_redirects=True,
        )
        self.max_retries = max_retries

    async def _post(self, path: str, payload: dict) -> dict:
        """带重试的 POST；HTTP 错误直接抛 LLMError，网络异常按退避重试。"""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(path, json=payload)
                if resp.status_code >= 400:
                    raise LLMError(self.name, f"HTTP {resp.status_code}: {resp.text[:300]}",
                                   status=resp.status_code)
                return resp.json()
            except LLMError:
                raise  # 4xx/5xx 是业务错误，重试无意义，直接向上抛出
            except Exception as exc:  # 网络层错误 → 重试
                last_exc = exc
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))  # 线性退避：1s、2s、…
        raise LLMError(self.name, f"请求失败: {last_exc}")

    def _estimate_cost(self, usage: TokenUsage) -> float | None:
        """按 usage 与 pricing 估算成本；未配置价格时返回 None 表示不可估算。"""
        if not (self.pricing.prompt_per_1m_usd or self.pricing.completion_per_1m_usd):
            return None
        return (usage.prompt_tokens * self.pricing.prompt_per_1m_usd
                + usage.completion_tokens * self.pricing.completion_per_1m_usd) / 1_000_000

    async def generate(self, messages: list[Message], *, temperature: float = 0.0,
                       max_tokens: int = 512, stop: list[str] | None = None,
                       tools: list[dict[str, Any]] | None = None,
                       tool_choice: str | None = None) -> LLMResponse:
        """非流式生成，返回统一 LLMResponse；支持工具调用与成本估算。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        start = time.perf_counter()
        data = await self._post("/chat/completions", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage_raw = data.get("usage") or {}
        usage = TokenUsage(prompt_tokens=usage_raw.get("prompt_tokens", 0),
                           completion_tokens=usage_raw.get("completion_tokens", 0))
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                # 个别模型返回非法 JSON，保底保留原文，避免整个响应解析失败。
                args = {"_raw": fn.get("arguments", "")}
            tool_calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name", ""), arguments=args))
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=self._estimate_cost(usage),
            model=data.get("model") or self.model,
            raw=data,
        )

    async def stream(self, messages: list[Message], *, temperature: float = 0.0,
                     max_tokens: int = 512, stop: list[str] | None = None) -> AsyncIterator[StreamChunk]:
        """真流式生成：解析 SSE 事件逐块产出，首块到达即为首 token 时延。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise LLMError(self.name, f"HTTP {resp.status_code}: {body[:300]}", status=resp.status_code)
            async for line in resp.aiter_lines():
                # SSE 事件可能夹杂空行或注释行，仅处理以 “data:” 开头的数据行。
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break  # 服务端流结束标记，正常退出
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue  # 忽略个别非法帧，保持流式输出鲁棒
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    yield StreamChunk(text_delta=delta["content"])
                if obj.get("choices") and obj["choices"][0].get("finish_reason"):
                    yield StreamChunk(finish_reason=obj["choices"][0]["finish_reason"])

    async def aclose(self) -> None:
        """释放底层 httpx 连接池，避免测试结束时泄漏连接。"""
        await self._client.aclose()
