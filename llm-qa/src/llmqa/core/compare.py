"""运行间结果对比：两次运行（或两个 Prompt 版本）逐用例 diff。

判定优劣分数：PASS(3) > SKIP(2) > FAIL(1) > ERROR(0)。
方向分类：regression（B 比 A 差）/ improvement / metric_drift（数值指标漂移）
/ message_change（失败消息变化）/ unchanged。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from llmqa.core.models import Severity, TestOutcome, Verdict

_VERDICT_SCORE = {Verdict.PASS: 3, Verdict.SKIP: 2, Verdict.FAIL: 1, Verdict.ERROR: 0}


class OutcomeDiff(BaseModel):
    """单个用例在两次运行间的差异。"""
    case_id: str
    name: str
    suite: str
    severity: Severity
    verdict_a: Verdict
    verdict_b: Verdict
    direction: str = "unchanged"   # regression | improvement | metric_drift | message_change | unchanged
    message_a: str = ""
    message_b: str = ""
    metric_diffs: dict[str, dict[str, float]] = Field(default_factory=dict)


def compare_outcomes(a: TestOutcome, b: TestOutcome) -> OutcomeDiff:
    """对比同一用例的两个结果：A 为基线（旧），B 为当前（新）。"""
    base = dict(case_id=a.case_id, name=a.name, suite=a.suite, severity=a.severity,
                verdict_a=a.verdict, verdict_b=b.verdict,
                message_a=a.message, message_b=b.message)
    sa, sb = _VERDICT_SCORE[a.verdict], _VERDICT_SCORE[b.verdict]
    if sb < sa:
        return OutcomeDiff(**base, direction="regression")
    if sb > sa:
        return OutcomeDiff(**base, direction="improvement")
    # 判定相同：先看数值指标漂移（延迟分位、judge 分数、检索指标等），再看失败消息变化
    diffs: dict[str, dict[str, float]] = {}
    for k in sorted(set(a.metrics) | set(b.metrics)):
        va, vb = a.metrics.get(k), b.metrics.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            diffs[k] = {"a": float(va), "b": float(vb)}
    if diffs:
        return OutcomeDiff(**base, direction="metric_drift", metric_diffs=diffs)
    if a.verdict in (Verdict.FAIL, Verdict.ERROR) and a.message != b.message:
        return OutcomeDiff(**base, direction="message_change")
    return OutcomeDiff(**base, direction="unchanged")


def summarize_diffs(diffs: list[OutcomeDiff]) -> dict[str, int]:
    """按方向汇总。"""
    counts = {"total": len(diffs), "unchanged": 0, "regressions": 0,
              "improvements": 0, "metric_drifts": 0, "message_changes": 0}
    for d in diffs:
        if d.direction == "regression":
            counts["regressions"] += 1
        elif d.direction == "improvement":
            counts["improvements"] += 1
        elif d.direction == "metric_drift":
            counts["metric_drifts"] += 1
        elif d.direction == "message_change":
            counts["message_changes"] += 1
        else:
            counts["unchanged"] += 1
    return counts
