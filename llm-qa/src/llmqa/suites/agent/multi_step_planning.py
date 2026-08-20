"""Agent 多步规划专项：验证多轮工具链、基于工具结果作答、完成任务后不再重复调用。

关注点：
- 多轮工具链按正确顺序执行（先查天气再计算）；
- 最终答案应基于工具观测内容；
- 任务完成后不应重复调用已用过的工具。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness, Tool


def _weather_tool() -> Tool:
    """构造天气工具：观测以"观测："开头，作为多步链的第一环。"""
    return Tool(name="get_weather", description="查询城市天气",
                parameters={"type": "object", "required": ["city"],
                            "properties": {"city": {"type": "string"}}},
                handler=lambda city: f"观测：{city} 晴 25 度")


def _calculator_tool() -> Tool:
    """构造计算器工具：观测以"观测："开头，作为多步链的第二环。"""
    return Tool(name="calculator", description="执行算术运算",
                parameters={"type": "object", "required": ["expression"],
                            "properties": {"expression": {"type": "string"}}},
                handler=lambda expression: "观测：计算结果为 96")


@test(id="agt-ms-001", suite="agent", name="两轮工具链顺序正确",
      description="多步任务应按顺序调用 get_weather 再 calculator",
      tags=("planning", "smoke"), severity=Severity.HIGH, timeout=60)
async def two_step_tool_chain(ctx: TestContext) -> None:
    """断言：tool_call_names == ["get_weather", "calculator"] 且成功。"""
    client = scripted_or_real(ctx, rules=[
        # 首条命中返回第一环工具调用；次条在 "[tool]" 观测行上命中并返回第二环调用，
        # 黄金脚本借此模拟"看到观测后决定下一步"的多步链
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply={"tool_calls": [
            {"id": "c2", "name": "calculator", "arguments": {"expression": "12*8"}}]},
            times=1, match_transcript=True),
        MockRule(match=r"\[tool\]", reply="综合结果：北京晴，12 乘以 8 等于 96。",
                 match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool()],
                           system_prompt="你是助手，可多步调用工具。", max_iterations=6)
    trace = await harness.run("先查一下北京的天气，然后计算 12 乘以 8 的结果")
    assert trace.success, "多步任务未成功完成"
    assert trace.tool_call_names == ["get_weather", "calculator"], f"工具链顺序错误: {trace.tool_call_names}"


@test(id="agt-ms-002", suite="agent", name="最终答案基于工具观测",
      description="Agent 应基于工具返回的观测内容组织最终答案",
      tags=("planning",), severity=Severity.MEDIUM, timeout=60)
async def answer_based_on_tool_result(ctx: TestContext) -> None:
    """断言：最终答案包含观测中的关键信息（气温 25 度）。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="根据工具观测，北京今天晴，气温 25 度。",
                 match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool()],
                           system_prompt="你是助手，基于工具结果作答。", max_iterations=4)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.success, "任务未成功完成"
    assert_contains(trace.tool_results[0].output, "25 度")
    # 最终答案须复现观测中的关键数值，证明答案源自工具结果而非凭空捏造
    assert_contains(trace.final_answer, "25 度")


@test(id="agt-ms-003", suite="agent", name="完成任务后不再重复调用工具",
      description="任务完成后不应重复调用已使用的工具",
      tags=("planning",), severity=Severity.MEDIUM, timeout=60)
async def no_repeated_tool_calls(ctx: TestContext) -> None:
    """断言：tool_call_names 无重复，且任务成功完成。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply={"tool_calls": [
            {"id": "c2", "name": "calculator", "arguments": {"expression": "1+1"}}]},
            times=1, match_transcript=True),
        MockRule(match=r"\[tool\]", reply="任务完成：北京晴，1+1=2。", match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool()],
                           system_prompt="你是助手，可多步调用工具。", max_iterations=6)
    trace = await harness.run("查北京天气，再算 1+1")
    assert trace.success, "任务未成功完成"
    # 用集合去重后的长度比对：相等即无重复，捕捉"任务完成后仍反复调用同一工具"的退化行为
    assert len(trace.tool_call_names) == len(set(trace.tool_call_names)), f"工具调用出现重复: {trace.tool_call_names}"
    assert trace.tool_call_names == ["get_weather", "calculator"], f"工具链顺序错误: {trace.tool_call_names}"
