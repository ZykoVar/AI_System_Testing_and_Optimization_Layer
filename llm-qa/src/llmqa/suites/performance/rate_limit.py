"""性能测试 - 限流行为：429 错误注入、错误率统计与重试语义。

关注点：
- 429 限流错误是否被正确封装为 LLMError 并携带 status；
- run_load 是否能正确统计"部分失败"场景的错误率；
- times 限制耗尽后同一客户端能否恢复成功（模拟重试语义）。

注意：本模块为框架故障注入测试，被测对象是框架自身的错误/重试路径，
而非真实模型，因此直接使用 ctx.providers.get_mock 构造脚本化 mock
（scripted_or_real 在真实 Provider 下会忽略规则，导致 429 无法注入）。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed, assert_contains
from llmqa.clients import LLMError, Message, MockRule
from llmqa.core.load import run_load
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(id="perf-rat-001", suite="performance", name="429 限流错误正确抛出",
      description="429 故障注入应抛出 LLMError 且 status==429",
      tags=("性能", "限流"), severity=Severity.HIGH, timeout=60)
async def case_rate_limit_raises(ctx: TestContext) -> None:
    """断言 error 规则命中后抛出 LLMError 且 status==429。"""
    # 直接 get_mock：429 故障注入必须在 mock 上才生效（scripted_or_real 在真实 Provider 下会忽略规则）
    client = ctx.providers.get_mock(rules=[
        MockRule(match=".*", error={"status": 429, "message": "rate limited"})])
    try:
        await client.generate([Message.user("触发限流")])
    except LLMError as e:
        if e.status != 429:
            raise AssertionFailed(f"LLMError status 期望 429，实际 {e.status}",
                                  metrics={"status": e.status})
        return
    raise AssertionFailed("429 故障注入后未抛出 LLMError")


@test(id="perf-rat-002", suite="performance", name="部分限流错误率统计正确",
      description="一半 429 一半成功的批量压测，error_rate 应落在 0.4~0.6 区间",
      tags=("性能", "限流"), severity=Severity.MEDIUM, timeout=60)
async def case_partial_rate_limit(ctx: TestContext) -> None:
    """断言 50 请求中 25 个 429（times=25）+ 25 成功时 error_rate ∈ [0.4, 0.6]。"""
    # times=25：50 个请求里前 25 个命中 429，其余落入兜底成功规则，构造"部分失败"场景
    client = ctx.providers.get_mock(rules=[
        MockRule(match=".*", error={"status": 429, "message": "rate limited"}, times=25),
        MockRule(match=".*", reply="限流恢复后的成功回复。"),
    ])
    stats = await run_load(lambda i: client.generate([Message.user(f"请求 {i}")]),
                           concurrency=10, count=50)
    # 期望约 0.5 的错误率，用 [0.4, 0.6] 区间留出统计抖动余量
    if not (0.4 <= stats.error_rate <= 0.6):
        raise AssertionFailed(
            f"部分限流场景 error_rate={stats.error_rate:.4f} 未落在 [0.4, 0.6] 区间",
            metrics={"error_rate": stats.error_rate, "errors": stats.errors,
                     "requests": stats.requests, "error_messages": stats.error_messages[:3]})


@test(id="perf-rat-003", suite="performance", name="限流后重试成功",
      description="首次 429（times=1）后第二次调用应成功（模拟重试语义）",
      tags=("性能", "限流"), severity=Severity.MEDIUM, timeout=60)
async def case_retry_after_limit(ctx: TestContext) -> None:
    """断言 times=1 的 429 规则耗尽后，同一客户端的第二次调用成功。"""
    # times=1：首次 429 用尽后同一条规则被跳过，第二次调用落到成功规则，模拟重试恢复语义
    client = ctx.providers.get_mock(rules=[
        MockRule(match=".*", error={"status": 429, "message": "rate limited"}, times=1),
        MockRule(match=".*", reply="重试成功。"),
    ])
    try:
        await client.generate([Message.user("第一次请求")])
    except LLMError:
        pass
    else:
        raise AssertionFailed("第一次调用未抛出 429")
    resp = await client.generate([Message.user("第二次请求（重试）")])
    assert_contains(resp.text, "重试成功", message="限流耗尽后第二次调用未恢复成功")
