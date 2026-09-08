"""平台轨迹 Adapter 骨架：把 LangSmith/Langfuse/Phoenix 等平台的轨迹
导出结构归一为 AgentTrajectory（见 llmqa.trajectory）。

定位：adapter 只做结构映射，不做判定——判定一律由行为断言 DSL 完成。
各平台导出格式与版本有差异，故此处提供映射参考与骨架；
接入时按所用平台的导出结构补全 raw → steps 的字段映射。
"""
from __future__ import annotations

from typing import Any

from llmqa.clients.base import ToolCall
from llmqa.trajectory import AgentTrajectory, TrajectoryStep


class LangSmithAdapter:
    """LangSmith 轨迹归一骨架。

    参考映射（LangSmith run tree 导出，按版本核对字段）：
    - run.name / run.inputs["task"] → task
    - run.child_runs：type=llm → kind=llm_call（content=outputs.content）
      type=tool → kind=tool_call（name/inputs）+ 紧随 observation（outputs）
    - run.end_time - start_time → latency；token_usage → total_tokens
    """

    def adapt(self, raw: Any) -> AgentTrajectory:
        raise NotImplementedError(
            "LangSmithAdapter 是映射骨架：请按所用 LangSmith 导出结构补全 "
            "raw → AgentTrajectory 的字段映射（参考类 docstring）。")


class LangfuseAdapter:
    """Langfuse 轨迹归一骨架。

    参考映射（Langfuse trace/observations 导出）：
    - trace.name / trace.input → task
    - observations 按层级：generation → llm_call；span(type=tool) → tool_call，
      span 的 output → observation
    - usage/totalCost → total_tokens / total_cost_usd
    """

    def adapt(self, raw: Any) -> AgentTrajectory:
        raise NotImplementedError(
            "LangfuseAdapter 是映射骨架：请按所用 Langfuse 导出结构补全 "
            "raw → AgentTrajectory 的字段映射（参考类 docstring）。")


# 已注册的平台 adapter 一览（接入新平台在此登记）
ADAPTERS = {"langsmith": LangSmithAdapter, "langfuse": LangfuseAdapter}


def adapt_trajectory(platform: str, raw: Any) -> AgentTrajectory:
    """按平台名选择 adapter 归一轨迹；未知平台抛 KeyError 并列出已支持项。"""
    if platform not in ADAPTERS:
        raise KeyError("未知轨迹平台 {}，已登记: {}".format(platform, sorted(ADAPTERS)))
    return ADAPTERS[platform]().adapt(raw)
