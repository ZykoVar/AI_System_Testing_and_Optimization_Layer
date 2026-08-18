"""Agent 工具选择专项：验证模型能根据任务语义选择正确的工具、避免无关工具调用。

关注点：
- 实时信息类问题（天气）应调用天气工具；
- 算术类问题应调用计算器工具；
- 闲聊不应触发任何工具调用；
- 多工具并存时，能按任务语义命中正确工具。

黄金脚本模式：第一轮用"任务关键词"命中并返回 tool_calls（times=1），
之后用 match_transcript 匹配 "[tool] xxx" 观测行返回最终答案，避免陷入循环。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness, Tool


def _weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="查询指定城市的实时天气",
        parameters={"type": "object", "required": ["city"],
                    "properties": {"city": {"type": "string"}}},
        handler=lambda city: "{} 晴 25 度".format(city),
    )


def _calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="执行算术运算",
        parameters={"type": "object", "required": ["expression"],
                    "properties": {"expression": {"type": "string"}}},
        handler=lambda expression: "计算完成：{}".format(expression),
    )


def _search_tool() -> Tool:
    return Tool(
        name="search",
        description="联网搜索最新信息",
        parameters={"type": "object", "required": ["query"],
                    "properties": {"query": {"type": "string"}}},
        handler=lambda query: "关于「{}」的搜索结果……".format(query),
    )


@test(id="agt-sel-001", suite="agent", name="天气问题应选择天气工具",
      description="实时天气问题应触发 get_weather 工具调用并给出最终答案",
      tags=("tool-calling", "smoke"), severity=Severity.HIGH, timeout=60)
async def weather_selects_get_weather(ctx: TestContext) -> None:
    """断言：天气问题调用 get_weather，最终答案包含天气关键词。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="北京今天晴，25 度。", match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool()],
                           system_prompt="你是助手，需要实时数据时调用工具。",
                           max_iterations=4)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.tool_call_names == ["get_weather"], "工具选择错误: {}".format(trace.tool_call_names)
    assert_contains(trace.final_answer, "晴")


@test(id="agt-sel-002", suite="agent", name="算术问题应选择计算器工具",
      description="算术计算问题应触发 calculator 工具调用",
      tags=("tool-calling",), severity=Severity.MEDIUM, timeout=60)
async def math_selects_calculator(ctx: TestContext) -> None:
    """断言：算术问题调用 calculator，最终答案包含计算结果。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="12 乘以 8", reply={"tool_calls": [
            {"id": "c1", "name": "calculator", "arguments": {"expression": "12*8"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="12 乘以 8 等于 96。", match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool()],
                           system_prompt="你是助手，需要计算时调用计算器。",
                           max_iterations=4)
    trace = await harness.run("请计算 12 乘以 8 等于多少？")
    assert trace.tool_call_names == ["calculator"], "工具选择错误: {}".format(trace.tool_call_names)
    assert_contains(trace.final_answer, "96")


@test(id="agt-sel-003", suite="agent", name="闲聊不应触发工具调用",
      description="纯闲聊任务应直接作答，不调用任何工具",
      tags=("tool-calling",), severity=Severity.MEDIUM, timeout=60)
async def chitchat_calls_no_tool(ctx: TestContext) -> None:
    """断言：闲聊时 tool_call_names 为空，且 final_answer 正常。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="你好", reply="你好！我是你的智能助手，很高兴见到你。"),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool(), _search_tool()],
                           system_prompt="你是助手，需要实时数据或计算时才调用工具。",
                           max_iterations=4)
    trace = await harness.run("你好，今天过得怎么样？")
    assert trace.success, "闲聊任务未正常完成"
    assert trace.tool_call_names == [], "闲聊不应调用工具，实际: {}".format(trace.tool_call_names)
    assert_contains(trace.final_answer, "你好")


@test(id="agt-sel-004", suite="agent", name="多工具并存时命中搜索工具",
      description="同时注册天气/计算器/搜索三个工具，搜索类问题应命中 search",
      tags=("tool-calling",), severity=Severity.MEDIUM, timeout=60)
async def multi_tool_selects_search(ctx: TestContext) -> None:
    """断言：三个工具并存时，搜索问题调用 search。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="搜索", reply={"tool_calls": [
            {"id": "c1", "name": "search", "arguments": {"query": "Acme 最新动态"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="这是 Acme 最新动态的搜索结果。", match_transcript=True),
    ])
    harness = AgentHarness(client, [_weather_tool(), _calculator_tool(), _search_tool()],
                           system_prompt="你是助手，可调用天气、计算、搜索等工具。",
                           max_iterations=4)
    trace = await harness.run("帮我搜索 Acme 公司的最新动态")
    assert trace.tool_call_names == ["search"], "工具选择错误: {}".format(trace.tool_call_names)
    assert_contains(trace.final_answer, "搜索结果")
