"""运行间结果对比：两次运行（或两个 Prompt 版本）逐用例 diff。

判定优劣分数：PASS(3) > SKIP(2) > FAIL(1) > ERROR(0)。
方向分类：
- regression（B 比 A 差：判定劣化，或指标按策略显著劣化）
- improvement（反向变好）
- metric_drift（仅记录的指标变化：未声明策略或 neutral 方向）
- message_change（失败消息变化）/ unchanged
- neutral（任一侧因 budget/fail_fast 跳过：未执行，不参与回归判定）
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from llmqa.core.metrics_policy import classify_metric_delta
from llmqa.core.models import Severity, TestOutcome, Verdict

_VERDICT_SCORE = {Verdict.PASS: 3, Verdict.SKIP: 2, Verdict.FAIL: 1, Verdict.ERROR: 0}

# 中性的 SKIP 语义：这些"跳过"不代表被测对象变差，回归对比时不应计为回归
_NEUTRAL_SKIP_REASONS = {"budget", "fail_fast"}


class OutcomeDiff(BaseModel):
    """单个用例在两次运行间的差异。"""
    case_id: str
    name: str
    suite: str
    severity: Severity
    verdict_a: Verdict
    verdict_b: Verdict
    direction: str = "unchanged"   # regression | improvement | metric_drift | message_change | unchanged | neutral
    message_a: str = ""
    message_b: str = ""
    metric_diffs: dict[str, dict[str, float]] = Field(default_factory=dict)  # 显著变化的指标 {metric: {a, b}}
    metric_signs: dict[str, str] = Field(default_factory=dict)               # 每项指标的方向（regression/improvement/drift）


def compare_outcomes(a: TestOutcome, b: TestOutcome, *,
                     policies: dict[str, dict] | None = None) -> OutcomeDiff:
    """对比同一用例的两个结果：A 为基线（旧），B 为当前（新）。

    policies: 指标判定策略（见 core.metrics_policy），None 时全部指标按 drift 记录。
    """
    base = dict(case_id=a.case_id, name=a.name, suite=a.suite, severity=a.severity,
                verdict_a=a.verdict, verdict_b=b.verdict,
                message_a=a.message, message_b=b.message)
    # 中性 SKIP：任一侧因 budget/fail_fast 未执行 → 不参与回归判定（避免误报）
    if ((a.verdict == Verdict.SKIP and a.skip_reason in _NEUTRAL_SKIP_REASONS)
            or (b.verdict == Verdict.SKIP and b.skip_reason in _NEUTRAL_SKIP_REASONS)):
        return OutcomeDiff(**base, direction="neutral")
    sa, sb = _VERDICT_SCORE[a.verdict], _VERDICT_SCORE[b.verdict]
    if sb < sa:
        return OutcomeDiff(**base, direction="regression")
    if sb > sa:
        return OutcomeDiff(**base, direction="improvement")
    # 判定相同：按指标策略评估漂移（容差内不显著，越界按方向归为回归/改善）
    diffs: dict[str, dict[str, float]] = {}
    signs: dict[str, str] = {}
    for k in sorted(set(a.metrics) | set(b.metrics)):
        va, vb = a.metrics.get(k), b.metrics.get(k)
        if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))) or va == vb:
            continue
        sign = classify_metric_delta(k, float(va), float(vb), policies)
        if sign is None:
            continue   # 容差内：不显著，不记录
        diffs[k] = {"a": float(va), "b": float(vb)}
        signs[k] = sign
    if any(s == "regression" for s in signs.values()):
        return OutcomeDiff(**base, direction="regression",
                           metric_diffs=diffs, metric_signs=signs)
    if any(s == "improvement" for s in signs.values()):
        return OutcomeDiff(**base, direction="improvement",
                           metric_diffs=diffs, metric_signs=signs)
    if diffs:
        return OutcomeDiff(**base, direction="metric_drift",
                           metric_diffs=diffs, metric_signs=signs)
    if a.verdict in (Verdict.FAIL, Verdict.ERROR) and a.message != b.message:
        return OutcomeDiff(**base, direction="message_change")
    return OutcomeDiff(**base, direction="unchanged")


def summarize_diffs(diffs: list[OutcomeDiff]) -> dict[str, int]:
    """按方向汇总。"""
    counts = {"total": len(diffs), "unchanged": 0, "regressions": 0,
              "improvements": 0, "metric_drifts": 0, "message_changes": 0,
              "neutrals": 0}
    for d in diffs:
        if d.direction == "regression":
            counts["regressions"] += 1
        elif d.direction == "improvement":
            counts["improvements"] += 1
        elif d.direction == "metric_drift":
            counts["metric_drifts"] += 1
        elif d.direction == "message_change":
            counts["message_changes"] += 1
        elif d.direction == "neutral":
            counts["neutrals"] += 1
        else:
            counts["unchanged"] += 1
    return counts
