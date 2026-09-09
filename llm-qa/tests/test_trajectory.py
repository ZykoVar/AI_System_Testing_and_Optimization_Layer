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


def test_max_steps():
    traj = make_traj(["a", "b"])
    assert_max_steps(traj, 2)
    with pytest.raises(AssertionFailed):
        assert_max_steps(traj, 1)


def test_max_cost_capability_semantics():
    from llmqa.core.registry import SkipTest
    from llmqa.trajectory import TrajectoryCapabilities

    traj = make_traj(["a"])
    # 无成本能力 → SKIP(unsupported)，绝不误判为通过
    with pytest.raises(SkipTest):
        assert_max_cost(traj, 0.10)
    # 声明能力并给真实成本 → 正常判定
    traj.capabilities = TrajectoryCapabilities(cost=True)
    traj.total_cost_usd = 0.05
    assert_max_cost(traj, 0.10)
    with pytest.raises(AssertionFailed):
        assert_max_cost(traj, 0.01)
    # 能力声明了但数据缺失 → 同样 SKIP（0 与 None 必须区分）
    traj.total_cost_usd = None
    with pytest.raises(SkipTest):
        assert_max_cost(traj, 0.10)


def test_zero_cost_is_not_missing_cost():
    # 语义区分：0.0 = 成本确为 0（可判定），None = 无数据（SKIP）
    from llmqa.trajectory import TrajectoryCapabilities

    traj = make_traj(["a"])
    traj.capabilities = TrajectoryCapabilities(cost=True)
    traj.total_cost_usd = 0.0
    assert_max_cost(traj, 1.0)   # 0 成本在预算内 → 通过


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


def test_behavior_hash_canonicalization():
    # 指纹只依赖规范行为：工具名/参数/终止/状态迁移；模型原话与时间戳不参与
    t1 = AgentTrajectory(task="t", steps=[
        tc("get_order", {"order_id": "A1"}), tc("refund_order", {"order_id": "A1"})])
    t2 = AgentTrajectory(task="t", steps=[
        tc("get_order", {"order_id": "A1"}), tc("refund_order", {"order_id": "A1"})])
    assert t1.behavior_hash() == t2.behavior_hash()
    t3 = AgentTrajectory(task="t", steps=[
        tc("get_order", {"order_id": "A1"}), tc("refund_order", {"order_id": "A2"})])
    assert t1.behavior_hash() != t3.behavior_hash()          # 参数变化 → 指纹变化
    t4 = AgentTrajectory(task="t", steps=[tc("get_order", {})],
                         finish_reason="loop_detected")
    assert t1.behavior_hash() != t4.behavior_hash()          # 终止原因参与指纹


def test_canonicalizer_dynamic_args_noise():
    from llmqa.trajectory import ArgPolicy, BehaviorCanonicalizer

    # 动态参数（user_id/request_id/order_id）不参与指纹 → hash 稳定
    canon = BehaviorCanonicalizer(
        default_tool_args="exact",
        ignored_fields=["user_id", "request_id"],
        per_tool={"refund_order": ArgPolicy(mode="exact", ignore_args=["order_id"])},
    )
    t1 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "iphone", "user_id": "u1"}),
        tc("refund_order", {"order_id": "A1"})])
    t2 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "iphone", "user_id": "u2"}),
        tc("refund_order", {"order_id": "A999"})])
    assert t1.behavior_hash(canon) == t2.behavior_hash(canon)   # 噪音字段被忽略
    assert t1.behavior_hash() != t2.behavior_hash()             # 无规范化器时如实反映差异
    # 语义参数变化仍然可见
    t3 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "ipad", "user_id": "u1"}),
        tc("refund_order", {"order_id": "A1"})])
    assert t1.behavior_hash(canon) != t3.behavior_hash(canon)


def test_canonicalizer_normalized_mode():
    from llmqa.trajectory import ArgPolicy, BehaviorCanonicalizer

    canon = BehaviorCanonicalizer(
        default_tool_args="exact",
        per_tool={"search": ArgPolicy(mode="normalized", keep_args=["query"])},
    )
    t1 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "  iPhone 15 ", "language": "zh"})])
    t2 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "iphone 15", "language": "en"})])
    # 空白/大小写归一 + keep_args 白名单（language 不参与）→ hash 一致
    assert t1.behavior_hash(canon) == t2.behavior_hash(canon)
    # 语义变化仍可检测
    t3 = AgentTrajectory(task="t", steps=[tc("search", {"query": "ipad", "language": "en"})])
    assert t1.behavior_hash(canon) != t3.behavior_hash(canon)


def test_canonicalizer_ignore_mode_and_state_keys():
    from llmqa.trajectory import ArgPolicy, BehaviorCanonicalizer

    canon = BehaviorCanonicalizer(
        per_tool={"search": ArgPolicy(mode="ignore")},
        ignored_state_keys=["order.updated_at"],
    )
    t1 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "任意参数都不参与"}),
        TrajectoryStep(index=9, kind="state_change",
                       state={"order.updated_at": {"from": "t1", "to": "t2"}})])
    t2 = AgentTrajectory(task="t", steps=[
        tc("search", {"query": "完全不同"}),
        TrajectoryStep(index=9, kind="state_change",
                       state={"order.updated_at": {"from": "t3", "to": "t4"}})])
    assert t1.behavior_hash(canon) == t2.behavior_hash(canon)


def test_canonicalizer_load_default_from_repo():
    from llmqa.trajectory import BehaviorCanonicalizer
    canon = BehaviorCanonicalizer.load_default()
    # 仓库配置：search=normalized+keep(query)、refund_order 忽略 order_id
    assert canon.policy_for("search").mode == "normalized"
    assert canon.policy_for("refund_order").ignore_args == ["order_id"]
    assert "user_id" in canon.ignored_fields


def test_agent_run_model_and_capabilities():
    # native 轨迹的能力声明必须如实（无成本/状态/审批），AgentRun 承载轨迹为 evidence
    traj = make_traj(["search"])
    assert traj.capabilities.cost is False
    assert traj.capabilities.state_changes is False
    from llmqa.trajectory import AgentRun
    run = AgentRun(task="t", trajectory=traj, success=True,
                   final_answer="答案", source="native")
    assert run.trajectory.behavior_hash() == traj.behavior_hash()


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
