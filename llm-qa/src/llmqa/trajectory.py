"""Agent 统一 Run 模型：AgentRun / AgentTrajectory / TrajectoryCapabilities。

架构约定（架构评审 §4/§6 收敛）：
- **AgentRun 是对外唯一 Run 模型**：execution metadata + trajectory（核心 evidence）
  + outcome + artifacts；AgentTrace 是 runtime 内部对象，随演进消失；
- **AgentTrajectory 是平台无关的轨迹 evidence**，携带 TrajectoryCapabilities
  声明"本来源提供哪些信息"——断言遇到能力缺失时 SKIP(unsupported)，
  绝不把"没有数据"误判成"通过"；
- **成本语义**：total_cost_usd 为 None 表示"无成本数据"，0.0 才表示"成本确实为 0"。

来源：native（本地 AgentHarness）/ langsmith / langfuse / phoenix / ...（Adapter 归一）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from llmqa.clients.base import TokenUsage, ToolCall

# 轨迹步骤类型：llm_call(模型输出) / tool_call(工具调用) / observation(工具返回)
# / state_change(状态变更) / approval_request(请求审批) / final(最终回答)
FINISH_REASONS = ("completed", "loop_detected", "budget_exceeded",
                  "tool_policy_violation", "error")


class TrajectoryCapabilities(BaseModel):
    """轨迹来源的能力声明：外部平台不可能提供完全一致的信息，
    断言据此判断"可断言 / 需 SKIP(unsupported)"，而非得到错误 PASS。"""
    tool_calls: bool = True
    tool_arguments: bool = True
    observations: bool = True
    state_changes: bool = False
    approval_events: bool = False
    tokens: bool = True
    cost: bool = False
    latency: bool = True
    errors: bool = True


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
    """一次 Agent 任务的完整轨迹（平台无关，AgentRun 的核心 evidence）。"""
    agent_id: str = ""
    task: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    finish_reason: str = "completed"
    total_tokens: int = 0
    total_cost_usd: float | None = None   # None=无成本数据；0.0=成本确为 0（语义必须区分）
    source: str = "native"
    capabilities: TrajectoryCapabilities = Field(default_factory=TrajectoryCapabilities)

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

    # ---------- Behavior Fingerprint（Agent Regression 基础） ----------
    def canonical_behavior(self) -> dict:
        """规范化行为：忽略 timestamp/trace_id/模型原话，
        保留工具名、参数、顺序、终止原因、状态迁移与策略事件。"""
        return {
            "task": self.task,
            "tools": self.tool_call_names,
            "tool_args": [s.tool_call.arguments for s in self.steps
                          if s.kind == "tool_call" and s.tool_call],
            "termination": self.finish_reason,
            "state_transitions": self.state_changes(),
            "approval_events": [s.tool_call.name for s in self.steps
                                if s.kind == "approval_request"
                                or (s.kind == "tool_call" and s.tool_call
                                    and "approval" in s.tool_call.name)],
        }

    def behavior_hash(self) -> str:
        """规范化行为的 sha256 指纹（前 12 位）：快速发现行为是否发生变化。"""
        payload = json.dumps(self.canonical_behavior(), ensure_ascii=False,
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class AgentRun(BaseModel):
    """对外的统一 Agent Run 模型：execution metadata + trajectory + outcome + artifacts。

    Runtime（AgentHarness 或外部平台经 Adapter/Instrumentation）→ AgentRun：
    trajectory 是核心 evidence，断言与回归都基于它。
    """
    run_id: str = ""
    agent_id: str = ""
    task: str = ""
    trajectory: AgentTrajectory = Field(default_factory=AgentTrajectory)
    termination: str = "completed"        # completed/loop_detected/budget_exceeded/tool_policy_violation/error
    success: bool = False
    final_answer: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    environment: dict[str, Any] = Field(default_factory=dict)  # runtime/platform 信息
    artifacts: dict[str, Any] = Field(default_factory=dict)    # trace_url/raw_trace_id 等外部引用
    source: str = "native"


class TrajectoryAdapter(Protocol):
    """平台轨迹归一协议：把任一平台的轨迹导出结构适配为 AgentTrajectory。

    实现约定：
    - adapter 只做结构映射，不做判定；判定一律由行为断言 DSL 完成；
    - adapt() 必须把来源能力写入 trajectory.capabilities（能力缺失≠数据为零）。
    """

    def adapt(self, raw: Any) -> AgentTrajectory:
        ...
