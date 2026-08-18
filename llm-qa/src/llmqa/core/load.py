"""并发负载生成工具：性能测试套件的基础设施。

run_load 以固定并发压 N 个请求，统计延迟分位、吞吐与 token 速率；
request_fn(i) 返回 LLMResponse（或其子集对象），异常计入 errors。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from llmqa.core.metrics import latency_stats


class LoadStats(BaseModel):
    requests: int = 0
    errors: int = 0
    error_rate: float = 0.0
    duration_ms: float = 0.0
    latency: dict[str, float] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    mean_cost_usd: float | None = None
    error_messages: list[str] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


async def run_load(
    request_fn: Callable[[int], Awaitable[Any]],
    *,
    concurrency: int = 10,
    count: int = 100,
    ramp_seconds: float = 0.0,
    per_request_timeout: float = 120.0,
) -> LoadStats:
    """并发压测。request_fn(i) -> 带 latency_ms/usage 属性的响应对象。"""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    error_messages: list[str] = []
    prompt_tokens = completion_tokens = 0
    costs: list[float] = []
    start = time.perf_counter()

    async def one(i: int) -> None:
        nonlocal prompt_tokens, completion_tokens
        async with sem:
            if ramp_seconds > 0 and i < concurrency:
                await asyncio.sleep(ramp_seconds * (i / max(1, concurrency)))
            try:
                resp = await asyncio.wait_for(request_fn(i), timeout=per_request_timeout)
            except Exception as e:  # noqa: BLE001 —— 压测中记录错误继续
                error_messages.append(f"{type(e).__name__}: {e}")
                return
            latencies.append(float(getattr(resp, "latency_ms", 0.0)))
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens += getattr(usage, "prompt_tokens", 0)
                completion_tokens += getattr(usage, "completion_tokens", 0)
            cost = getattr(resp, "cost_usd", None)
            if isinstance(cost, (int, float)):
                costs.append(float(cost))

    await asyncio.gather(*(one(i) for i in range(count)))
    duration_s = (time.perf_counter() - start)
    ok = len(latencies)
    return LoadStats(
        requests=count,
        errors=count - ok,
        error_rate=round((count - ok) / count, 4) if count else 0.0,
        duration_ms=duration_s * 1000,
        latency=latency_stats(latencies),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=round((prompt_tokens + completion_tokens) / duration_s, 2) if duration_s else 0.0,
        requests_per_second=round(count / duration_s, 2) if duration_s else 0.0,
        mean_cost_usd=round(sum(costs) / len(costs), 6) if costs else None,
        error_messages=error_messages[:5],
    )


async def run_ramp(
    request_fn: Callable[[int], Awaitable[Any]],
    *,
    levels: list[int] | None = None,
    per_level: int = 20,
) -> list[LoadStats]:
    """阶梯并发：依次以 1/2/5/10/20 并发压测，观察扩展性。"""
    levels = levels or [1, 2, 5, 10, 20]
    return [await run_load(request_fn, concurrency=c, count=per_level) for c in levels]
