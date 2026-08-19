"""性能测试 - 吞吐：并发 10×50 请求的错误率、RPS 与 token 吞吐记录。

关注点：
- 固定并发下批量请求是否全部成功（error_rate == 0）；
- 每秒请求数（RPS）与每秒 token 吞吐是否被正确统计。

mock 下 2ms 延迟、无错误，验证统计口径正确；真实 Provider 下即真实吞吐验收。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.load import run_load
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


async def _basic_load(ctx: TestContext):
    """构造被测客户端并执行 10 并发 × 50 请求的压测。"""
    payload = ctx.datasets.load("performance/payloads")["medium"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="吞吐测试回复。")])
    return await run_load(lambda i: client.generate([Message.user(payload)]),
                          concurrency=10, count=50)


@test(id="perf-thr-001", suite="performance", name="并发 10×50 请求零错误",
      description="10 并发 50 请求的批量压测错误率为 0",
      tags=("性能", "吞吐", "smoke"), severity=Severity.HIGH, timeout=60)
async def case_zero_error(ctx: TestContext) -> None:
    """断言并发 10 × 50 请求 error_rate == 0。"""
    stats = await _basic_load(ctx)
    # mock 下应零错误，用精确比较（!= 0）而非区间，捕捉任何一次失败
    if stats.error_rate != 0:
        raise AssertionFailed(
            "批量压测出现错误（error_rate={:.4f}）: {}".format(stats.error_rate,
                                                               stats.error_messages[:3]),
            metrics={"error_rate": stats.error_rate, "errors": stats.errors,
                     "requests": stats.requests})


@test(id="perf-thr-002", suite="performance", name="吞吐 RPS 被正确记录",
      description="批量压测的每秒请求数（RPS）被记录且大于 0",
      tags=("性能", "吞吐"), severity=Severity.LOW, timeout=60)
async def case_rps_recorded(ctx: TestContext) -> None:
    """断言 requests_per_second > 0。"""
    stats = await _basic_load(ctx)
    rps = stats.requests_per_second
    # RPS 基于总请求数与墙钟耗时计算，<=0 说明统计口径或压测本身异常
    if rps <= 0:
        raise AssertionFailed("每秒请求数（RPS）未正确记录: {:.2f}".format(rps),
                              metrics={"requests_per_second": rps,
                                       "duration_ms": stats.duration_ms})


@test(id="perf-thr-003", suite="performance", name="token 吞吐被正确记录",
      description="批量压测的每秒 token 吞吐被记录且大于 0",
      tags=("性能", "吞吐"), severity=Severity.LOW, timeout=60)
async def case_token_throughput(ctx: TestContext) -> None:
    """断言 tokens_per_second > 0。"""
    stats = await _basic_load(ctx)
    tps = stats.tokens_per_second
    # token 吞吐依赖 usage 统计，<=0 说明 token 未计入或压测异常
    if tps <= 0:
        raise AssertionFailed(
            "每秒 token 吞吐未正确记录: {:.2f}".format(tps),
            metrics={"tokens_per_second": tps, "total_tokens": stats.total_tokens,
                     "duration_ms": stats.duration_ms})
