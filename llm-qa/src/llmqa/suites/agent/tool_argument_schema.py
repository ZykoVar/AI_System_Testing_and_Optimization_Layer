"""Agent 工具参数 Schema 专项：验证必填参数传递、参数类型正确性与工具异常恢复。

关注点：
- 模型调用工具时必填参数必须正确传递（handler 校验通过并返回 OK）；
- 参数类型必须符合 JSON Schema（integer 参数应为整数而非字符串）；
- 工具 handler 抛异常时，异常应转为观测（含"[工具异常]"），且 Agent 能基于观测给出最终答案（恢复性）。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness, Tool


@test(id="agt-arg-001", suite="agent", name="必填参数正确传递",
      description="工具调用必须携带必填参数，handler 校验通过并返回 OK",
      tags=("tool-arguments", "smoke"), severity=Severity.HIGH, timeout=60)
async def required_argument_passed(ctx: TestContext) -> None:
    """断言：handler 收到完整必填参数 city，观测含 OK。"""
    # handler 内部再次校验：它是"必填参数确已传递"的权威，不依赖 harness 的 schema 校验
    def weather_handler(city: str) -> str:
        if not city or not isinstance(city, str):
            return "FAIL: 缺少必填参数 city"
        return f"OK: 已查询 {city} 的天气"

    weather = Tool(name="get_weather", description="查询城市天气",
                   parameters={"type": "object", "required": ["city"],
                               "properties": {"city": {"type": "string"}}},
                   handler=weather_handler)
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="北京今天晴。", match_transcript=True),
    ])
    harness = AgentHarness(client, [weather], system_prompt="你是助手，可调用工具。",
                           max_iterations=4)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.tool_results, "未产生任何工具观测"
    assert_contains(trace.tool_results[0].output, "OK")


@test(id="agt-arg-002", suite="agent", name="参数类型正确（integer）",
      description="integer 参数必须作为整数传递，而非字符串",
      tags=("tool-arguments",), severity=Severity.MEDIUM, timeout=60)
async def integer_argument_type(ctx: TestContext) -> None:
    """断言：handler 收到的 a/b 为 int 类型，观测含 OK。"""
    def add_handler(a: int, b: int) -> str:
        if isinstance(a, int) and isinstance(b, int):
            return f"OK: {a} + {b} = {a + b}"
        return f"FAIL: 参数类型错误 a={type(a).__name__} b={type(b).__name__}"

    add = Tool(name="add", description="两整数相加",
               parameters={"type": "object", "required": ["a", "b"],
                           "properties": {"a": {"type": "integer"},
                                          "b": {"type": "integer"}}},
               handler=add_handler)
    client = scripted_or_real(ctx, rules=[
        MockRule(match="3 加 5", reply={"tool_calls": [
            {"id": "c1", "name": "add", "arguments": {"a": 3, "b": 5}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="3 加 5 等于 8。", match_transcript=True),
    ])
    harness = AgentHarness(client, [add], system_prompt="你是助手，可调用工具。",
                           max_iterations=4)
    trace = await harness.run("请计算 3 加 5 等于多少")
    assert trace.tool_results, "未产生任何工具观测"
    assert_contains(trace.tool_results[0].output, "OK")
    # JSON 序列化可能把 int 强转成 str，这里显式断言类型，捕获字符串强制转换
    assert trace.tool_results[0].arguments == {"a": 3, "b": 5}, "参数应保持为整数"
    assert isinstance(trace.tool_results[0].arguments["a"], int), "参数 a 应为 int"


@test(id="agt-arg-003", suite="agent", name="工具异常转为观测并恢复作答",
      description="工具 handler 抛异常时，观测含[工具异常]，Agent 仍给出最终答案",
      tags=("tool-arguments",), severity=Severity.MEDIUM, timeout=60)
async def tool_exception_recovery(ctx: TestContext) -> None:
    """断言：工具异常观测含[工具异常]，且 Agent 恢复给出最终答案。"""
    # 故意抛异常：验证 harness 把异常转为含 "[工具异常]" 的观测而非让整轮运行崩溃
    def broken_handler(city: str) -> str:
        raise RuntimeError("数据库连接失败")

    weather = Tool(name="get_weather", description="查询城市天气",
                   parameters={"type": "object", "required": ["city"],
                               "properties": {"city": {"type": "string"}}},
                   handler=broken_handler)
    client = scripted_or_real(ctx, rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="抱歉，暂时无法获取天气信息，请稍后重试。",
                 match_transcript=True),
    ])
    harness = AgentHarness(client, [weather], system_prompt="你是助手，可调用工具。",
                           max_iterations=4)
    trace = await harness.run("北京今天天气怎么样？")
    assert trace.tool_results, "未产生任何工具观测"
    assert_contains(trace.tool_results[0].output, "[工具异常]")
    # 关键：异常仅作为观测，不应中断整轮；Agent 应基于异常观测继续给出最终答案
    assert trace.success, "工具异常后 Agent 应恢复并给出最终答案"
    assert_contains(trace.final_answer, "无法获取")
