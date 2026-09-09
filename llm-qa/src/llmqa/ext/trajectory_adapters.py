"""平台轨迹 Adapter：把 LangSmith/Langfuse 等平台的轨迹导出结构
归一为 AgentTrajectory（见 llmqa.trajectory）。

定位：adapter 只做结构映射，不做判定——判定一律由行为断言 DSL 完成。
实现状态（如实声明）：
- LangfuseAdapter：按 Langfuse 公开数据模型（trace/observations：
  GENERATION/SPAN/EVENT）完整实现映射，已用 fixture 验证映射逻辑；
  未经真实平台实测（接入时按所用 Langfuse 版本核对字段）。
- LangSmithAdapter：映射骨架（参考映射见类 docstring）。
"""
from __future__ import annotations

from typing import Any

from llmqa.clients.base import ToolCall
from llmqa.trajectory import (
    AgentTrajectory,
    TrajectoryCapabilities,
    TrajectoryStep,
)

# Langfuse 无原生"工具调用"类型：约定 Agent 的工具调用以 SPAN 表达，
# 状态变更以 EVENT 表达（metadata 携带 state_change）
_LANGFUSE_STATE_KEY = "state_change"
_LANGFUSE_APPROVAL_KEY = "approval_request"


def _extract_content(value: Any) -> str:
    """Langfuse output/input 可能是字符串或 dict；dict 时取 content 字段。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            content = value["content"]
            return content if isinstance(content, str) else str(content)
        return str(value)
    return str(value)


def _extract_tool_calls(value: Any) -> list[ToolCall]:
    """从 GENERATION 的 output 中提取 OpenAI 风格 tool_calls（如有）。"""
    if not isinstance(value, dict):
        return []
    raw_calls = value.get("tool_calls") or []
    calls = []
    for i, tc in enumerate(raw_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = {"_raw": args}
        calls.append(ToolCall(id=tc.get("id") or "tc-{}".format(i),
                              name=fn.get("name", ""), arguments=args))
    return calls


class LangfuseAdapter:
    """Langfuse 轨迹归一（完整映射实现，fixture 验证）。

    输入：Langfuse trace 导出 JSON（Public API GET /api/public/traces/{id}
    或 SDK get_trace() 的序列化结构）：trace + observations 数组。

    映射约定：
    - trace.name / trace.input(字符串或 dict["task"]) → task；
    - GENERATION → llm_call（content=output.content；output 含 tool_calls 时
      紧跟 tool_call 步骤）；
    - SPAN → tool_call（name=span.name，arguments=input dict），
      其 output → observation 步骤；
    - EVENT → metadata 含 state_change/approval_request 时映射为对应步骤，
      否则 observation；
    - usage.total → total_tokens；observations 的 totalCost 求和 → total_cost_usd；
    - metadata["termination"]（可选）→ finish_reason，默认 completed；
    - capabilities：cost/latency/tokens 据实声明。
    """

    def adapt(self, raw: Any) -> AgentTrajectory:
        if isinstance(raw, str):
            import json
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("Langfuse trace 导出应为 dict（或 JSON 字符串）")

        observations = raw.get("observations") or []
        task = self._extract_task(raw)
        metadata = raw.get("metadata") or {}
        steps: list[TrajectoryStep] = []
        total_cost = 0.0
        total_tokens = 0
        index = 0
        has_cost = False

        # 按 parentObservationId 拓扑排序（同层级保持输入顺序）
        obs = sorted(observations, key=lambda o: self._depth(o, observations))
        for o in obs:
            otype = (o.get("type") or "").upper()
            usage = o.get("usage") or {}
            cost = o.get("totalCost")
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
                has_cost = True
            total_tokens += int(usage.get("total") or usage.get("output", 0) or 0)
            latency = 0.0
            if o.get("startTime") and o.get("endTime"):
                latency = self._iso_diff_ms(o["startTime"], o["endTime"])
            if otype == "GENERATION":
                index += 1
                content = _extract_content(o.get("output"))
                steps.append(TrajectoryStep(index=index, kind="llm_call",
                                            content=content, latency_ms=latency))
                for tc in _extract_tool_calls(o.get("output")):
                    index += 1
                    steps.append(TrajectoryStep(index=index, kind="tool_call",
                                                tool_call=tc, latency_ms=latency))
            elif otype == "SPAN":
                index += 1
                args = o.get("input") if isinstance(o.get("input"), dict) else {"_input": o.get("input")}
                steps.append(TrajectoryStep(
                    index=index, kind="tool_call",
                    tool_call=ToolCall(id=o.get("id") or "", name=o.get("name") or "",
                                       arguments=args),
                    latency_ms=latency))
                index += 1
                steps.append(TrajectoryStep(index=index, kind="observation",
                                            observation=_extract_content(o.get("output")),
                                            latency_ms=latency))
            elif otype == "EVENT":
                emeta = o.get("metadata") or {}
                if _LANGFUSE_STATE_KEY in emeta:
                    index += 1
                    steps.append(TrajectoryStep(index=index, kind="state_change",
                                                state=emeta[_LANGFUSE_STATE_KEY]))
                elif _LANGFUSE_APPROVAL_KEY in emeta:
                    index += 1
                    steps.append(TrajectoryStep(index=index, kind="approval_request",
                                                content=_extract_content(o.get("output"))))
                else:
                    index += 1
                    steps.append(TrajectoryStep(index=index, kind="observation",
                                                observation=_extract_content(o.get("output")),
                                                latency_ms=latency))
        # 最终回答：最后一个 llm_call 或 GENERATION 的内容
        final = next((s.content for s in reversed(steps) if s.kind == "llm_call"), "")
        if final:
            steps.append(TrajectoryStep(index=index + 1, kind="final", content=final))
        termination = metadata.get("termination") or "completed"
        return AgentTrajectory(
            agent_id=str(raw.get("userId") or metadata.get("agent_id") or ""),
            task=task,
            steps=steps,
            finish_reason=termination,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6) if has_cost else None,
            source="langfuse",
            capabilities=TrajectoryCapabilities(
                cost=has_cost, latency=True, tokens=True,
                state_changes=True, approval_events=True, errors=True),
        )

    @staticmethod
    def _extract_task(raw: dict) -> str:
        if raw.get("input"):
            inp = raw["input"]
            if isinstance(inp, str):
                return inp
            if isinstance(inp, dict):
                return str(inp.get("task") or list(inp.values())[0] if inp else "")
        return str(raw.get("name") or "")

    @staticmethod
    def _iso_diff_ms(start: str, end: str) -> float:
        try:
            from datetime import datetime
            a = datetime.fromisoformat(start.replace("Z", "+00:00"))
            b = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return (b - a).total_seconds() * 1000
        except Exception:  # noqa: BLE001 —— 时间解析失败不阻断映射
            return 0.0

    @staticmethod
    def _depth(o: dict, observations: list) -> int:
        depth = 0
        parent = o.get("parentObservationId")
        while parent:
            depth += 1
            parent = next((p.get("parentObservationId")
                           for p in observations if p.get("id") == parent), None)
        return depth


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


# 已注册的平台 adapter 一览（接入新平台在此登记）
ADAPTERS = {"langsmith": LangSmithAdapter, "langfuse": LangfuseAdapter}


def adapt_trajectory(platform: str, raw: Any) -> AgentTrajectory:
    """按平台名选择 adapter 归一轨迹；未知平台抛 KeyError 并列出已支持项。"""
    if platform not in ADAPTERS:
        raise KeyError("未知轨迹平台 {}，已登记: {}".format(platform, sorted(ADAPTERS)))
    return ADAPTERS[platform]().adapt(raw)
