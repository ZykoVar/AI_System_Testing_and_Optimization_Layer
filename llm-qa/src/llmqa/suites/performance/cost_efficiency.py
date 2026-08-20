"""性能测试 - 成本效率：单请求成本与批量成本预算。

关注点：
- 单次请求成本是否 ≤ 全局阈值 cost_per_request_usd；
- 批量 50 请求的估算总成本是否 ≤ 预算（0.5 USD）。

mock 下 cost_usd 恒为 0，成本校验必过；真实 Provider 依赖 providers.yaml 的 pricing
进行粗粒度估算（未配置 pricing 时无法估算，用例显式 SKIP）。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.load import run_load
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import SkipTest, test


@test(id="perf-cos-001", suite="performance", name="单请求成本达标",
      description="单次生成请求成本不超过全局阈值 cost_per_request_usd",
      tags=("性能", "成本"), severity=Severity.LOW, timeout=60)
async def case_single_cost(ctx: TestContext) -> None:
    """断言单请求 cost_usd ≤ thresholds.cost_per_request_usd。"""
    payload = ctx.datasets.load("performance/payloads")["short"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="成本测试回复。")])
    resp = await client.generate([Message.user(payload)])
    cost = resp.cost_usd
    # mock 下 cost 恒为 0；真实 Provider 未配 pricing 时无法估算，显式 SKIP 而非误判
    if cost is None:
        raise SkipTest("Provider 未配置 pricing，无法估算单请求成本")
    limit = ctx.settings.thresholds.cost_per_request_usd
    if cost > limit:
        raise AssertionFailed(f"单请求成本 {cost:.6f} USD 超过阈值 {limit:.6f} USD",
                              metrics={"cost_usd": cost, "threshold_usd": limit})


@test(id="perf-cos-002", suite="performance", name="批量请求成本在预算内",
      description="批量 50 请求的估算总成本不超过 0.5 USD 预算",
      tags=("性能", "成本"), severity=Severity.MEDIUM, timeout=60)
async def case_batch_cost(ctx: TestContext) -> None:
    """断言 mean_cost_usd × requests ≤ 0.5 USD。"""
    payload = ctx.datasets.load("performance/payloads")["medium"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="批量成本测试回复。")])
    stats = await run_load(lambda i: client.generate([Message.user(payload)]),
                           concurrency=8, count=50)
    mean = stats.mean_cost_usd
    # run_load 只有收集到 cost 才给出 mean；无 pricing 时 mean 为 None，显式 SKIP
    if mean is None:
        raise SkipTest("Provider 未配置 pricing，无法估算批量成本")
    total = mean * stats.requests
    budget = 0.5  # 批量 50 请求的成本护栏预算（固定 0.5 USD）
    if total > budget:
        raise AssertionFailed(
            f"批量估算总成本 {total:.6f} USD 超过预算 {budget:.2f} USD",
            metrics={"total_cost_usd": total, "budget_usd": budget,
                     "mean_cost_usd": mean, "requests": stats.requests})
