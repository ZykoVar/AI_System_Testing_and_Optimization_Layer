"""Agent 预算护栏专项：验证迭代预算与 token 预算能正确中止 Agent，正常任务不受影响。

关注点：
- 迭代预算耗尽（max_iterations）→ budget_exceeded；
- token 预算耗尽（max_total_tokens）→ budget_exceeded；
- 正常任务在预算范围内应成功完成。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness, Tool


def _weather_tool() -> Tool:
    return Tool(name="get_weather", description="查询城市天气",
                parameters={"type": "object", "required": ["city"],
                            "properties": {"city": {"type": "string"}}},
                handler=lambda city: "{} 晴 25 度".format(city))


@test(id="agt-bud-001", suite="agent", name="迭代预算耗尽中止",
      description="max_iterations 耗尽时应以 budget_exceeded 中止",
      tags=("budget-guardrail", "smoke"), severity=Severity.HIGH, timeout=60)
async def iteration_budget_exceeded(ctx: TestContext) -> None:
    """断言：无限工具调用 + max_iterations=3 → budget_exceeded。"""
    # 不设 times：规则无限命中；调高 stop_on_repeated_calls 关闭循环检测，聚焦迭代预算
    client = scripted_or_real(ctx, rules=[
        MockRule(match=".*", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}),
    ])
    harness = AgentHarness(client, [_weather_tool()], system_prompt="你是助手，可调用工具。",
                           max_iterations=3, stop_on_repeated_calls=9999)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.abort_reason == "budget_exceeded", "期望 budget_exceeded，实际: {}".format(trace.abort_reason)


@test(id="agt-bud-002", suite="agent", name="token 预算耗尽中止",
      description="max_total_tokens 设很小时应触发 budget_exceeded",
      tags=("budget-guardrail",), severity=Severity.MEDIUM, timeout=60)
async def token_budget_exceeded(ctx: TestContext) -> None:
    """断言：max_total_tokens=1 → 首轮即 budget_exceeded。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match=".*", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}),
    ])
    harness = AgentHarness(client, [_weather_tool()], system_prompt="你是助手，可调用工具。",
                           max_iterations=6, max_total_tokens=1)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.abort_reason == "budget_exceeded", "期望 budget_exceeded，实际: {}".format(trace.abort_reason)


@test(id="agt-bud-003", suite="agent", name="正常任务在预算内成功",
      description="预算充足时正常任务应成功完成且无中止原因",
      tags=("budget-guardrail",), severity=Severity.MEDIUM, timeout=60)
async def normal_task_within_budget(ctx: TestContext) -> None:
    """断言：正常任务 success 且 abort_reason 为 None。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="北京今天晴，25 度。", match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool()], system_prompt="你是助手，可调用工具。",
                           max_iterations=6, max_total_tokens=2000)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.success, "正常任务未成功完成"
    assert trace.abort_reason is None, "不应中止，实际: {}".format(trace.abort_reason)
    assert_contains(trace.final_answer, "晴")
