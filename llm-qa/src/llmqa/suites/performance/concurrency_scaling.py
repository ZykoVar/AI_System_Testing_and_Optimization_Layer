"""性能测试 - 并发扩展：阶梯并发下的稳定性、延迟退化与并行加速。

关注点：
- 阶梯并发 [1,2,5,10,20] 下是否全程无错误；
- 并发从 1 提升到 20 时 P95 延迟的退化倍数是否可控；
- 高并发是否带来真实加速（总耗时低于串行估算）。

mock 下 2ms 延迟、无锁竞争，扩展性断言必过；真实 Provider 下可反映真实扩展瓶颈。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.load import run_ramp
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


async def _ramp(ctx: TestContext):
    """构造被测客户端并执行阶梯并发 [1,2,5,10,20] 压测。"""
    payload = ctx.datasets.load("performance/payloads")["short"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="并发扩展测试回复。")])
    return await run_ramp(lambda i: client.generate([Message.user(payload)]),
                          levels=[1, 2, 5, 10, 20], per_level=20)


@test(id="perf-sca-001", suite="performance", name="阶梯并发全程零错误",
      description="阶梯并发 [1,2,5,10,20] 每一档错误率均为 0",
      tags=("性能", "并发"), severity=Severity.HIGH, timeout=60)
async def case_ramp_zero_error(ctx: TestContext) -> None:
    """断言所有并发档位 error_rate == 0。"""
    ramp = await _ramp(ctx)
    # 逐档收集有错误的档位及其前 2 条错误，便于定位是哪一档并发压出了失败
    bad = [(i, s.error_rate, s.error_messages[:2]) for i, s in enumerate(ramp)
           if s.error_rate != 0]
    if bad:
        raise AssertionFailed("阶梯并发出现错误: {}".format(bad),
                              metrics={"failed_levels": len(bad),
                                       "levels": [s.error_rate for s in ramp]})


@test(id="perf-sca-002", suite="performance", name="并发 20 延迟退化可控",
      description="并发 20 的 P95 延迟不超过单并发 P95 的 3 倍",
      tags=("性能", "并发"), severity=Severity.MEDIUM, timeout=60)
async def case_p95_scaling(ctx: TestContext) -> None:
    """断言并发 20 的 P95 延迟 ≤ 单并发 P95 的 3 倍。"""
    ramp = await _ramp(ctx)
    p95_1 = ramp[0].latency.get("p95_ms", 0.0)
    p95_20 = ramp[-1].latency.get("p95_ms", 0.0)
    # 单并发 P95 作为基准；为 0（无样本）时退化为 1.0 防除零，避免把退化比算成无穷
    baseline = p95_1 if p95_1 > 0 else 1.0  # 防除零
    # 允许并发带来一定延迟退化，但不得超过 3 倍，否则判定扩展性失控
    if p95_20 > baseline * 3.0:
        raise AssertionFailed(
            "并发 20 的 P95 {:.1f}ms 超过单并发 P95 {:.1f}ms 的 3 倍".format(p95_20, p95_1),
            metrics={"p95_1_ms": p95_1, "p95_20_ms": p95_20, "ratio": p95_20 / baseline})


@test(id="perf-sca-003", suite="performance", name="并发 20 真实加速",
      description="并发 20 的墙钟总耗时低于串行（平均延迟×请求数）估算",
      tags=("性能", "并发"), severity=Severity.LOW, timeout=60)
async def case_parallel_speedup(ctx: TestContext) -> None:
    """断言并发 20 档总耗时 < 串行估算（mean_ms × count）。"""
    ramp = await _ramp(ctx)
    lvl20 = ramp[-1]
    mean_ms = lvl20.latency.get("mean_ms", 0.0)
    # 串行估算 = 平均延迟 × 请求数；并发墙钟耗时若不低于它，说明并发未带来真实加速
    serial_ms = mean_ms * lvl20.requests
    if lvl20.duration_ms >= serial_ms:
        raise AssertionFailed(
            "并发 20 总耗时 {:.1f}ms 未低于串行估算 {:.1f}ms".format(
                lvl20.duration_ms, serial_ms),
            metrics={"duration_ms": lvl20.duration_ms, "serial_estimate_ms": serial_ms,
                     "mean_ms": mean_ms})
