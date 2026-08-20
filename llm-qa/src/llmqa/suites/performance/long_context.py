"""性能测试 - 长上下文：长载荷下的请求成功与延迟退化。

关注点：
- 长提示（datasets/performance/payloads.yaml 的 long）能否正常返回；
- 长载荷下 P95 延迟是否仍低于 2× 全局 P95 阈值。

mock 下 2ms 延迟、不受载荷长度影响，校验必过；真实 Provider 下反映长上下文的真实代价。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.load import run_load
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


def _long_payload(ctx: TestContext) -> str:
    """加载长载荷（长上下文与输出长度对延迟影响的性能测试提示词）。"""
    return ctx.datasets.load("performance/payloads")["long"]


@test(id="perf-lc-001", suite="performance", name="长载荷请求成功",
      description="长提示请求正常返回非空回复",
      tags=("性能", "长上下文"), severity=Severity.LOW, timeout=60)
async def case_long_success(ctx: TestContext) -> None:
    """断言长载荷请求成功（返回非空文本）。"""
    payload = _long_payload(ctx)
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="长上下文处理成功。")])
    resp = await client.generate([Message.user(payload)], max_tokens=256)
    # 长提示请求的核心验收：能否正常返回非空文本（不被截断或直接失败）
    if not resp.text:
        raise AssertionFailed("长载荷请求返回空回复",
                              metrics={"prompt_tokens": resp.usage.prompt_tokens,
                                       "completion_tokens": resp.usage.completion_tokens})


@test(id="perf-lc-002", suite="performance", name="长载荷延迟退化可控",
      description="长载荷下 P95 延迟仍低于 2 倍全局 P95 阈值",
      tags=("性能", "长上下文"), severity=Severity.MEDIUM, timeout=60)
async def case_long_latency(ctx: TestContext) -> None:
    """断言长载荷压测 P95 延迟 < 2 × thresholds.p95_latency_ms。"""
    payload = _long_payload(ctx)
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="长上下文回复。")])
    stats = await run_load(lambda i: client.generate([Message.user(payload)], max_tokens=128),
                           concurrency=4, count=20)
    # 先拦截请求错误：有错误时延迟分位不可信，直接失败而非继续比较阈值
    if stats.errors:
        raise AssertionFailed(f"长载荷压测出现错误: {stats.error_messages[:3]}",
                              metrics={"errors": stats.errors, "error_rate": stats.error_rate})
    p95 = stats.latency.get("p95_ms", 0.0)
    # 长上下文允许更高延迟，但仍封顶在 2× 常规阈值，防止长载荷退化失控
    limit = 2 * ctx.settings.thresholds.p95_latency_ms
    if p95 >= limit:
        raise AssertionFailed(f"长载荷 P95 延迟 {p95:.1f}ms 超过 2× 阈值 {limit:.1f}ms",
                              metrics={"p95_ms": p95, "threshold_ms": limit})
