"""AgentTrajectory 统一轨迹模型 + TrajectoryAdapter 协议。

战略定位：llm_learn 是"Testing & Optimization Layer"——外部平台（LangSmith/
Langfuse/Phoenix/Braintrust/Opik）负责记录与评估 Agent 发生了什么，
本项目负责"规定 Agent 应该怎么行为"。为此各平台轨迹统一归一为
AgentTrajectory，行为断言 DSL 不绑定任何平台（见 assertors/trajectory.py）。

来源：
- native：本地 AgentHarness 直接产出（离线可复现）；
- langsmith / langfuse / phoenix / ...：经 TrajectoryAdapter 归一。
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from llmqa.clients.base import ToolCall

# 轨迹步骤类型：llm_call(模型输出) / tool_call(工具调用) / observation(工具返回)
# / state_change(状态变更) / approval_request(请求审批) / final(最终回答)
FINISH_REASONS = ("completed", "loop_detected", "budget_exceeded",
                  "tool_policy_violation", "error")


class TrajectoryStep(BaseModel):
    """单步轨迹：跨平台统一表达（各 Adapter 负责把平台特有结构映射进来）。"""
    index: int
    kind: str = "llm_call"
    content: str = ""                     # 模型文本/最终回答
    tool_call: ToolCall | None = None     # kind=tool_call 时必填
    observation: str = ""                 # kind=observation 时的工具返回
    state: dict[str, dict[str, Any]] | None = None   # kind=state_change：{key: {"from":..,"to":..}}
    latency_ms: float = 0.0


class AgentTrajectory(BaseModel):
    """一次 Agent 任务的完整轨迹（平台无关）。"""
    agent_id: str = ""
    task: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    finish_reason: str = "completed"
    total_tokens: int = 0
    total_cost_usd: float = 0.0           # native 默认 0；外部平台接入后为真实成本
    source: str = "native"                # native | langsmith | langfuse | phoenix | ...

    # ---------- 便捷查询（DSL 断言的基础） ----------
    @property
    def tool_call_names(self) -> list[str]:
        return [s.tool_call.name for s in self.steps
                if s.kind == "tool_call" and s.tool_call is not None]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def tool_called(self, name: str) -> bool:
        return name in self.tool_call_names

    def state_changes(self) -> list[dict]:
        """展平全部状态变更：[{key, from, to, index}]。"""
        out = []
        for s in self.steps:
            if s.kind == "state_change" and s.state:
                for key, change in s.state.items():
                    out.append({"key": key,
                                "from": change.get("from"),
                                "to": change.get("to"),
                                "index": s.index})
        return out


class TrajectoryAdapter(Protocol):
    """平台轨迹归一协议：把任一平台的轨迹导出结构适配为 AgentTrajectory。

    实现约定：adapter 不应做判定，只做结构映射；判定一律由 DSL 断言完成。
    """

    def adapt(self, raw: Any) -> AgentTrajectory:
        ...
