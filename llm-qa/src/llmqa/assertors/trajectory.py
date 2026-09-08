"""Agent 行为断言 DSL：规定"Agent 应该怎么行为"，与平台无关。

断言对象统一为 AgentTrajectory（见 llmqa.trajectory）——轨迹无论来自
本地 AgentHarness 还是 LangSmith/Langfuse/Phoenix 等平台（经 Adapter 归一），
同一套断言都能执行。这是本项目 Agent 测试的核心壁垒。

五大类（Behavior Contract 而非 trajectory similarity——后者交给 LangSmith 等平台）：
① Tool     工具行为：assert_tool_called / assert_tool_not_called / assert_tool_args
② Sequence 顺序关系：assert_tool_sequence（strict 子序列）
③ Resource 资源约束：assert_max_steps / assert_max_cost（能力缺失→SKIP）
④ Policy   安全策略：assert_requires_approval
⑤ State    状态迁移：assert_state_changed
"""
from __future__ import annotations

from typing import Any

from llmqa.assertors.base import AssertionFailed
from llmqa.trajectory import AgentTrajectory


def assert_tool_called(traj: AgentTrajectory, name: str, message: str | None = None) -> None:
    """断言轨迹中调用过指定工具。"""
    if not traj.tool_called(name):
        raise AssertionFailed(
            message or "期望调用工具 {}，实际调用序列: {}".format(
                name, traj.tool_call_names or "无"),
            metrics={"tool_call_count": len(traj.tool_call_names)},
            evidence=["轨迹工具序列: " + str(traj.tool_call_names)])


def assert_tool_not_called(traj: AgentTrajectory, name: str, message: str | None = None) -> None:
    """断言轨迹中未调用指定工具（安全用例：危险工具零容忍）。"""
    if traj.tool_called(name):
        raise AssertionFailed(
            message or "禁止调用工具 {}，但轨迹中出现了".format(name),
            evidence=["轨迹工具序列: " + str(traj.tool_call_names)])


def assert_tool_sequence(traj: AgentTrajectory, names: list[str], *,
                         strict: bool = False, message: str | None = None) -> None:
    """断言工具按顺序调用。

    strict=False：names 按序出现（子序列，允许中间穿插其他工具）；
    strict=True：names 必须连续出现。
    """
    actual = traj.tool_call_names
    if strict:
        ok = any(actual[i:i + len(names)] == names
                 for i in range(len(actual) - len(names) + 1))
    else:
        it = iter(actual)
        ok = all(n in it for n in names)
    if not ok:
        raise AssertionFailed(
            message or "期望工具序列 {}（strict={}），实际: {}".format(
                names, strict, actual),
            evidence=["轨迹工具序列: " + str(actual)])


def assert_max_steps(traj: AgentTrajectory, max_steps: int, message: str | None = None) -> None:
    """断言轨迹步数不超过上限（预算/效率护栏）。"""
    if traj.step_count > max_steps:
        raise AssertionFailed(
            message or "步数 {} 超过上限 {}".format(traj.step_count, max_steps),
            metrics={"steps": traj.step_count})


def assert_max_cost(traj: AgentTrajectory, max_usd: float, message: str | None = None) -> None:
    """断言轨迹总成本不超过预算。

    能力语义（Capability Model）：轨迹来源未提供成本信息时抛 SkipTest
    （= SKIP: unsupported），绝不把"没有数据"误判成"通过"。
    """
    if not traj.capabilities.cost or traj.total_cost_usd is None:
        from llmqa.core.registry import SkipTest
        raise SkipTest("轨迹来源({})不支持成本断言（capabilities.cost=False）".format(
            traj.source))
    cost = traj.total_cost_usd
    if cost > max_usd:
        raise AssertionFailed(
            message or "成本 {:.4f} 超过预算 {:.4f}".format(cost, max_usd),
            metrics={"cost_usd": cost})


def assert_tool_args(traj: AgentTrajectory, name: str, expected: dict,
                     message: str | None = None) -> None:
    """断言某次工具调用的参数包含期望键值（子集匹配）。"""
    matches = [s.tool_call for s in traj.steps
               if s.kind == "tool_call" and s.tool_call and s.tool_call.name == name]
    for tc in matches:
        if all(tc.arguments.get(k) == v for k, v in expected.items()):
            return
    raise AssertionFailed(
        message or "工具 {} 的调用参数未满足 {}，实际: {}".format(
            name, expected, [tc.arguments for tc in matches]),
        evidence=[str(tc.arguments) for tc in matches])


def assert_requires_approval(traj: AgentTrajectory, before: str,
                             message: str | None = None) -> None:
    """断言在调用指定工具之前存在审批行为（approval_request 步骤或 request_approval 工具）。"""
    approved = False
    for s in traj.steps:
        if s.kind == "approval_request" or (
                s.kind == "tool_call" and s.tool_call and "approval" in s.tool_call.name):
            approved = True
        if s.kind == "tool_call" and s.tool_call and s.tool_call.name == before:
            if not approved:
                raise AssertionFailed(
                    message or "调用工具 {} 之前缺少审批行为".format(before),
                    evidence=["轨迹工具序列: " + str(traj.tool_call_names)])
            return
    raise AssertionFailed(
        message or "轨迹未调用工具 {}，无法验证审批前置".format(before),
        evidence=["轨迹工具序列: " + str(traj.tool_call_names)])


def assert_state_changed(traj: AgentTrajectory, key: str, *,
                         to: Any = None, from_: Any = ...,
                         message: str | None = None) -> None:
    """断言存在状态变更 key: from → to（from_ 省略时只校验目标值）。"""
    for change in traj.state_changes():
        if change["key"] != key:
            continue
        if to is not None and change["to"] != to:
            continue
        if from_ is not ... and change["from"] != from_:
            continue
        return
    raise AssertionFailed(
        message or "未发现状态变更 {}（from={} to={}）".format(key, from_, to),
        evidence=[str(c) for c in traj.state_changes()])
