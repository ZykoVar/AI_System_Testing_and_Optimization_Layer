"""AgentTrajectory 统一轨迹模型与行为断言 DSL 自测。"""
import pytest

from llmqa.assertors import (
    AssertionFailed,
    assert_max_cost,
    assert_max_steps,
    assert_requires_approval,
    assert_state_changed,
    assert_tool_args,
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_sequence,
)
from llmqa.clients.base import ToolCall
from llmqa.trajectory import AgentTrajectory, TrajectoryStep


def tc(name, args):
    return TrajectoryStep(index=0, kind="tool_call",
                          tool_call=ToolCall(id="c", name=name, arguments=args))


def make_traj(tool_names=None, args_map=None, extra_steps=None):
    steps = []
    for i, name in enumerate(tool_names or [], 1):
        steps.append(TrajectoryStep(
            index=i, kind="tool_call",
            tool_call=ToolCall(id=str(i), name=name,
                               arguments=(args_map or {}).get(name, {}))))
    steps.extend(extra_steps or [])
    return AgentTrajectory(task="t", steps=steps)


def test_tool_called_and_not_called():
    traj = make_traj(["search", "get_document"])
    assert_tool_called(traj, "search")
    assert_tool_not_called(traj, "delete")
    with pytest.raises(AssertionFailed):
        assert_tool_called(traj, "refund")
    with pytest.raises(AssertionFailed):
        assert_tool_not_called(traj, "search")


def test_tool_sequence_subsequence_and_strict():
    traj = make_traj(["search", "get_document", "answer", "search"])
    assert_tool_sequence(traj, ["get_document", "answer"])          # 子序列
    assert_tool_sequence(traj, ["search", "answer"], strict=False)
    with pytest.raises(AssertionFailed):
        assert_tool_sequence(traj, ["search", "answer"], strict=True)  # 中间隔了工具
    with pytest.raises(AssertionFailed):
        assert_tool_sequence(traj, ["answer", "get_document"])         # 顺序不符


def test_max_steps_and_cost():
    traj = make_traj(["a", "b"])
    assert_max_steps(traj, 2)
    with pytest.raises(AssertionFailed):
        assert_max_steps(traj, 1)
    traj.total_cost_usd = 0.05
    assert_max_cost(traj, 0.10)
    with pytest.raises(AssertionFailed):
        assert_max_cost(traj, 0.01)
    traj.total_cost_usd = None
    with pytest.raises(AssertionFailed):
        assert_max_cost(traj, 0.10)   # 无成本信息必须显式失败


def test_tool_args_subset():
    traj = make_traj(["refund_order"], args_map={"refund_order": {"order_id": "A1", "reason": "x"}})
    assert_tool_args(traj, "refund_order", {"order_id": "A1"})
    with pytest.raises(AssertionFailed):
        assert_tool_args(traj, "refund_order", {"order_id": "A2"})


def test_requires_approval():
    ok_steps = [
        TrajectoryStep(index=0, kind="approval_request"),
        tc("refund_order", {}),
    ]
    assert_requires_approval(AgentTrajectory(task="t", steps=ok_steps), "refund_order")
    bad = [tc("refund_order", {})]
    with pytest.raises(AssertionFailed):
        assert_requires_approval(AgentTrajectory(task="t", steps=bad), "refund_order")


def test_state_changed():
    steps = [tc("refund_order", {}),
             TrajectoryStep(index=1, kind="state_change",
                            state={"order.status": {"from": "pending", "to": "refunded"}})]
    traj = AgentTrajectory(task="t", steps=steps)
    assert_state_changed(traj, "order.status", from_="pending", to="refunded")
    assert_state_changed(traj, "order.status", to="refunded")        # from_ 省略
    with pytest.raises(AssertionFailed):
        assert_state_changed(traj, "order.status", from_="shipped", to="refunded")
    with pytest.raises(AssertionFailed):
        assert_state_changed(traj, "order.nonexistent", to="x")


def test_agent_trace_to_trajectory():
    import asyncio
    from llmqa.clients import MockClient, MockRule
    from llmqa.harnesses import AgentHarness, Tool
    client = MockClient(rules=[
        MockRule(match="天气", reply={"tool_calls": [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1),
        MockRule(match=r"\[tool\]", reply="晴。", match_transcript=True),
    ])
    weather = Tool(name="get_weather", description="查天气",
                   parameters={"type": "object", "properties": {}},
                   handler=lambda **kw: "晴")
    trace = asyncio.run(AgentHarness(client, [weather]).run("天气？"))
    traj = trace.to_trajectory()
    assert traj.source == "native"
    assert traj.finish_reason == "completed"
    assert traj.tool_call_names == ["get_weather"]
    kinds = [s.kind for s in traj.steps]
    assert "tool_call" in kinds and "observation" in kinds and "final" in kinds
