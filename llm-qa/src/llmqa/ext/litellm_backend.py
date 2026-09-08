"""LiteLLM 后端适配：把 LiteLLM（100+ 模型统一接口、自动重试/fallback、成本核算）
接到本项目 LLMClient 契约，替换手写的单家适配器。

激活：pip install -e ".[ext]"；providers.yaml 中 kind: litellm。
未安装 litellm 时构造抛 LLMError（附安装指引），不影响离线 CI。
"""
from __future__ import annotations

import json
import time
from typing import Any

from llmqa.clients.base import LLMClient, LLMError, LLMResponse, Message, TokenUsage, ToolCall


def _import_litellm():
    try:
        import litellm  # noqa: F401
        return litellm
    except ImportError as e:  # pragma: no cover —— 依赖缺失路径由单测覆盖
        raise LLMError(
            "litellm",
            '未安装 litellm，无法使用 kind=litellm。安装：pip install -e ".[ext]"',
        ) from e


class LiteLLMClient(LLMClient):
    """LiteLLM 统一网关客户端：model 可用 "provider/model" 形式（如 openai/gpt-4o-mini）。"""

    def __init__(self, name: str, model: str, *, api_key: str | None = None,
                 temperature_default: float = 0.0, max_tokens_default: int = 512):
        super().__init__(name, model)
        self._litellm = _import_litellm()
        self.api_key = api_key
        self.temperature_default = temperature_default
        self.max_tokens_default = max_tokens_default

    def _to_messages(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            item: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                item["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in m.tool_calls]
            if m.tool_call_id:
                item["tool_call_id"] = m.tool_call_id
            out.append(item)
        return out

    async def generate(self, messages: list[Message], *, temperature: float = 0.0,
                       max_tokens: int = 512, stop: list[str] | None = None,
                       tools: list[dict[str, Any]] | None = None,
                       tool_choice: str | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if self.api_key:
            kwargs["api_key"] = self.api_key
        start = time.perf_counter()
        resp = await self._litellm.acompletion(**kwargs)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage_raw = resp.get("usage") or {}
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments", "")}
            tool_calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name", ""),
                                       arguments=args))
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(prompt_tokens=usage_raw.get("prompt_tokens", 0),
                             completion_tokens=usage_raw.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - start) * 1000,
            # LiteLLM 自带成本核算（resp._hidden_params），需要时在此扩展映射
            cost_usd=None,
            model=resp.get("model") or self.model,
            raw=dict(resp),
        )
