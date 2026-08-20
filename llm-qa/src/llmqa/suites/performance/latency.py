"""性能测试 - 延迟：端到端 P95/P99 与流式首 token 延迟（TTFT）。

关注点：
- 端到端生成延迟的分位数是否满足全局阈值（settings.thresholds）；
- 流式输出首 token 延迟（Time To First Token）是否达标。

mock 默认延迟 2ms，阈值校验必然通过（用于离线验证测试逻辑）；
真实 Provider 下同一用例即为真实验收（规则不生效、直连真实模型）。
"""
from __future__ import annotations

import asyncio
import time

from llmqa.assertors import AssertionFailed
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.load import run_load
from llmqa.core.metrics import latency_stats
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


def _short_payload(ctx: TestContext) -> str:
    """加载短载荷（性能压测的轻量提示词）。"""
    return ctx.datasets.load("performance/payloads")["short"]


@test(id="perf-lat-001", suite="performance", name="端到端 P95 延迟达标",
      description="并发压测端到端生成，P95 延迟低于全局阈值 p95_latency_ms",
      tags=("性能", "延迟", "smoke"), severity=Severity.MEDIUM, timeout=60)
async def case_p95_latency(ctx: TestContext) -> None:
    """断言 50 次请求的 P95 端到端延迟 < thresholds.p95_latency_ms。"""
    payload = _short_payload(ctx)
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="延迟测试回复。")])
    stats = await run_load(lambda i: client.generate([Message.user(payload)]),
                           concurrency=8, count=50)
    # 先拦截请求错误：有错误时延迟分位已不可信，直接失败而非继续比较阈值
    if stats.errors:
        raise AssertionFailed(f"压测出现错误: {stats.error_messages[:3]}",
                              metrics={"errors": stats.errors, "error_rate": stats.error_rate})
    p95 = stats.latency.get("p95_ms", 0.0)
    limit = ctx.settings.thresholds.p95_latency_ms
    if p95 >= limit:
        raise AssertionFailed(f"P95 延迟 {p95:.1f}ms 超过阈值 {limit:.1f}ms",
                              metrics={"p95_ms": p95, "threshold_ms": limit})


@test(id="perf-lat-002", suite="performance", name="端到端 P99 延迟达标",
      description="并发压测端到端生成，P99 延迟低于全局阈值 p99_latency_ms",
      tags=("性能", "延迟"), severity=Severity.MEDIUM, timeout=60)
async def case_p99_latency(ctx: TestContext) -> None:
    """断言 50 次请求的 P99 端到端延迟 < thresholds.p99_latency_ms。"""
    payload = _short_payload(ctx)
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="延迟测试回复。")])
    stats = await run_load(lambda i: client.generate([Message.user(payload)]),
                           concurrency=8, count=50)
    if stats.errors:
        raise AssertionFailed(f"压测出现错误: {stats.error_messages[:3]}",
                              metrics={"errors": stats.errors, "error_rate": stats.error_rate})
    p99 = stats.latency.get("p99_ms", 0.0)
    limit = ctx.settings.thresholds.p99_latency_ms
    if p99 >= limit:
        raise AssertionFailed(f"P99 延迟 {p99:.1f}ms 超过阈值 {limit:.1f}ms",
                              metrics={"p99_ms": p99, "threshold_ms": limit})


async def _measure_ttft(client, text: str) -> float | None:
    """测量单次流式请求的首 token 延迟（毫秒）；无文本增量返回 None。"""
    start = time.perf_counter()
    async for chunk in client.stream([Message.user(text)]):
        # 首个非空文本增量即视为"首 token 到达"，返回其耗时（毫秒）
        if chunk.text_delta:
            return (time.perf_counter() - start) * 1000
        # 流提前结束而无文本增量时中断：返回 None 交由调用方过滤
        if chunk.finish_reason is not None:
            break
    return None


@test(id="perf-lat-003", suite="performance", name="流式首 token 延迟（TTFT）达标",
      description="流式请求首 chunk 到达延迟的 P95 低于全局阈值 ttft_p95_ms",
      tags=("性能", "延迟", "流式"), severity=Severity.MEDIUM, timeout=60)
async def case_ttft(ctx: TestContext) -> None:
    """断言 30 次流式请求首 token 延迟的 P95 < thresholds.ttft_p95_ms。"""
    payload = _short_payload(ctx)
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="流式首 token 测试回复。")])
    # 用信号量把并发流式请求限制在 8：既压出并发效果，又避免瞬时打满连接
    sem = asyncio.Semaphore(8)

    async def one(i: int) -> float | None:
        async with sem:
            return await _measure_ttft(client, payload)

    # 过滤掉无文本增量的 None 样本；若全部为空，说明流式实现未产出任何文本，直接判失败
    ttfts = [t for t in await asyncio.gather(*(one(i) for i in range(30))) if t is not None]
    if not ttfts:
        raise AssertionFailed("未能测得任何首 token 延迟（流式无文本增量）")
    p95 = latency_stats(ttfts)["p95_ms"]
    limit = ctx.settings.thresholds.ttft_p95_ms
    if p95 >= limit:
        raise AssertionFailed(f"首 token 延迟 P95 {p95:.1f}ms 超过阈值 {limit:.1f}ms",
                              metrics={"ttft_p95_ms": p95, "threshold_ms": limit,
                                       "samples": len(ttfts)})
