"""Agent 循环检测护栏专项：验证重复工具调用循环能被及时中止。

关注点：
- 模型持续返回相同的工具调用时，应触发 loop_detected 中止；
- 循环护栏应在少量迭代内（<=4）及时生效，而非耗尽迭代预算。
"""
from __future__ import annotations

from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness, Tool


def _weather_tool() -> Tool:
    """构造天气工具：handler 固定返回相同内容，配合无限命中规则制造调用循环。"""
    return Tool(name="get_weather", description="查询城市天气",
                parameters={"type": "object", "required": ["city"],
                            "properties": {"city": {"type": "string"}}},
                handler=lambda city: "{} 晴 25 度".format(city))


@test(id="agt-loop-001", suite="agent", name="重复工具调用触发循环检测",
      description="模型无限返回相同工具调用时应以 loop_detected 中止",
      tags=("loop-guardrail", "smoke"), severity=Severity.HIGH, timeout=60)
async def loop_detected(ctx: TestContext) -> None:
    """断言：abort_reason == "loop_detected"。"""
    # 不设 times：规则无限命中，模拟模型持续返回相同工具调用
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}),
    ])
    # stop_on_repeated_calls=3：连续 3 次相同 (name,args) 调用即触发循环检测中止
    harness = AgentHarness(client, [_weather_tool()], system_prompt="你是助手，可调用工具。",
                           max_iterations=6, stop_on_repeated_calls=3)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.abort_reason == "loop_detected", "期望 loop_detected，实际: {}".format(trace.abort_reason)


@test(id="agt-loop-002", suite="agent", name="循环护栏及时中止",
      description="循环检测应在少量迭代内生效，而非耗尽迭代预算",
      tags=("loop-guardrail",), severity=Severity.MEDIUM, timeout=60)
async def loop_aborts_promptly(ctx: TestContext) -> None:
    """断言：abort_reason == loop_detected 且 iterations <= 4。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}),
    ])
    harness = AgentHarness(client, [_weather_tool()], system_prompt="你是助手，可调用工具。",
                           max_iterations=8, stop_on_repeated_calls=3)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.abort_reason == "loop_detected", "期望 loop_detected，实际: {}".format(trace.abort_reason)
    # 护栏应在第 3 次重复调用即中止（iterations==3），而非耗尽 max_iterations=8 的预算
    assert trace.iterations <= 4, "循环护栏未及时生效，iterations={}".format(trace.iterations)
