"""平台轨迹 Adapter 自测：Langfuse 完整映射（fixture 验证）。"""
import json

from llmqa.ext.trajectory_adapters import LangfuseAdapter, adapt_trajectory

# 模拟 Langfuse Public API trace 导出结构（按公开数据模型构造的 fixture）
FIXTURE = {
    "id": "trace-abc",
    "name": "refund-flow",
    "userId": "agent-7",
    "input": {"task": "给订单 A1 退款"},
    "metadata": {"termination": "completed"},
    "observations": [
        {
            "id": "gen-1", "type": "GENERATION", "name": "chat",
            "parentObservationId": None,
            "output": {"content": "需要先查订单再退款"},
            "usage": {"input": 50, "output": 20, "total": 70},
            "totalCost": 0.0005,
            "startTime": "2026-09-09T10:00:00Z", "endTime": "2026-09-09T10:00:01Z",
        },
        {
            "id": "span-1", "type": "SPAN", "name": "get_order",
            "parentObservationId": "gen-1",
            "input": {"order_id": "A1"},
            "output": {"content": "订单 A1 存在，状态 pending"},
            "usage": {"total": 0}, "totalCost": 0.0,
            "startTime": "2026-09-09T10:00:01Z", "endTime": "2026-09-09T10:00:02Z",
        },
        {
            "id": "evt-1", "type": "EVENT", "name": "state",
            "parentObservationId": "span-1",
            "output": "状态更新",
            "metadata": {"state_change": {"order.status": {"from": "pending", "to": "refunded"}}},
        },
        {
            "id": "gen-2", "type": "GENERATION", "name": "chat",
            "parentObservationId": "evt-1",
            "output": {"content": "退款已完成"},
            "usage": {"input": 30, "output": 10, "total": 40},
            "totalCost": 0.0003,
            "startTime": "2026-09-09T10:00:02Z", "endTime": "2026-09-09T10:00:03Z",
        },
    ],
}


def test_langfuse_adapt_steps():
    traj = LangfuseAdapter().adapt(FIXTURE)
    kinds = [(s.kind, getattr(s.tool_call, "name", None)) for s in traj.steps]
    assert ("llm_call", None) in kinds
    assert ("tool_call", "get_order") in kinds
    assert ("observation", None) in kinds
    assert ("state_change", None) in kinds
    assert ("final", None) in kinds
    final_step = next(s for s in traj.steps if s.kind == "final")
    assert final_step.content == "退款已完成"
    assert traj.tool_call_names == ["get_order"]
    assert traj.task == "给订单 A1 退款"
    assert traj.agent_id == "agent-7"
    assert traj.finish_reason == "completed"
    assert traj.source == "langfuse"


def test_langfuse_adapt_cost_and_capabilities():
    traj = LangfuseAdapter().adapt(FIXTURE)
    assert traj.total_tokens == 110                     # 70 + 40（EVENT/SPAN 无 usage）
    assert traj.total_cost_usd == 0.0008                # 0.0005 + 0.0003
    caps = traj.capabilities
    assert caps.cost is True
    assert caps.state_changes is True
    assert caps.approval_events is True


def test_langfuse_adapt_string_input():
    traj = LangfuseAdapter().adapt(json.dumps(FIXTURE, ensure_ascii=False))
    assert traj.task == "给订单 A1 退款"


def test_langfuse_adapt_state_change_assertable():
    from llmqa.assertors import assert_state_changed
    traj = LangfuseAdapter().adapt(FIXTURE)
    assert_state_changed(traj, "order.status", from_="pending", to="refunded")


def test_langfuse_adapt_behavior_hash():
    # 平台归一轨迹与 native 轨迹同构：behavior_hash 可直接参与跨平台回归
    traj = LangfuseAdapter().adapt(FIXTURE)
    assert len(traj.behavior_hash()) == 12


def test_adapt_trajectory_registry():
    traj = adapt_trajectory("langfuse", FIXTURE)
    assert traj.source == "langfuse"
    try:
        adapt_trajectory("unknown_platform", FIXTURE)
        raise AssertionError("未知平台应抛 KeyError")
    except KeyError:
        pass


def test_langfuse_no_cost_data():
    # 无 totalCost 的导出：成本 None + capabilities.cost=False（不冒充零成本）
    import copy
    fixture = copy.deepcopy(FIXTURE)
    for o in fixture["observations"]:
        o.pop("totalCost", None)
    traj = LangfuseAdapter().adapt(fixture)
    assert traj.total_cost_usd is None
    assert traj.capabilities.cost is False
