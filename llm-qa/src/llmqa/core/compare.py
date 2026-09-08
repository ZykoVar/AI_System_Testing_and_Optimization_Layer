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
    direction: str = "unchanged"   # regression | improvement | metric_drift | message_change | behavior_change | unchanged | neutral
    message_a: str = ""
    message_b: str = ""
    metric_diffs: dict[str, dict[str, float]] = Field(default_factory=dict)  # 显著变化的指标 {metric: {a, b}}
    metric_signs: dict[str, str] = Field(default_factory=dict)               # 每项指标的方向（regression/improvement/drift）
    behavior_hash_a: str = ""
    behavior_hash_b: str = ""        # behavior_change 方向时记录两侧行为指纹


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
    # 行为回归：Agent 用例的规范化行为指纹（工具序列/参数/终止/状态迁移）不同
    # → 单独报告 behavior_change，供详细 diff（行为细节见 canonical_behavior）
    hash_a = (a.artifacts or {}).get("behavior_hash")
    hash_b = (b.artifacts or {}).get("behavior_hash")
    if hash_a and hash_b and hash_a != hash_b:
        return OutcomeDiff(**base, direction="behavior_change",
                           behavior_hash_a=hash_a, behavior_hash_b=hash_b)
    return OutcomeDiff(**base, direction="unchanged")


def summarize_diffs(diffs: list[OutcomeDiff]) -> dict[str, int]:
    """按方向汇总。"""
    counts = {"total": len(diffs), "unchanged": 0, "regressions": 0,
              "improvements": 0, "metric_drifts": 0, "message_changes": 0,
              "behavior_changes": 0, "neutrals": 0}
    for d in diffs:
        if d.direction == "regression":
            counts["regressions"] += 1
        elif d.direction == "improvement":
            counts["improvements"] += 1
        elif d.direction == "metric_drift":
            counts["metric_drifts"] += 1
        elif d.direction == "message_change":
            counts["message_changes"] += 1
        elif d.direction == "behavior_change":
            counts["behavior_changes"] += 1
        elif d.direction == "neutral":
            counts["neutrals"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def split_by_identity(common: list[str], id_a: dict, id_b: dict) -> tuple[list[str], list[str]]:
    """Baseline compatibility 第一关：按用例源码指纹切分。

    同 case_id 但两侧指纹都存在且不同 → 身份失配（用例代码/数据已变），
    结果不可直接对比，单独列出而非计为回归。
    返回 (可对比列表, 身份失配列表)。
    """
    mismatched: list[str] = []
    comparable: list[str] = []
    for cid in common:
        hash_a, hash_b = id_a.get(cid), id_b.get(cid)
        if hash_a and hash_b and hash_a != hash_b:
            mismatched.append(cid)
        else:
            comparable.append(cid)
    return comparable, mismatched


def env_diff(prov_a: dict | None, prov_b: dict | None) -> list[str]:
    """报告 Prompt/数据集指纹差异行（环境差异提示：判定继续，但结论需谨慎）。"""
    lines: list[str] = []

    def prompt_set(p: dict | None) -> set:
        if not p:
            return set()
        return {(u.get("id"), u.get("version"), u.get("content_hash", ""))
                for u in (p.get("prompts_used") or [])}

    def dataset_set(p: dict | None) -> set:
        if not p:
            return set()
        return {(u.get("name"), u.get("content_hash", ""))
                for u in (p.get("datasets_used") or [])}

    pa, pb = prompt_set(prov_a), prompt_set(prov_b)
    for pid, ver, h in sorted(pa - pb):
        lines.append("Prompt 仅基线使用: {} v{} ({})".format(pid, ver, h[:6]))
    for pid, ver, h in sorted(pb - pa):
        lines.append("Prompt 仅当前使用: {} v{} ({})".format(pid, ver, h[:6]))
    da, db = dataset_set(prov_a), dataset_set(prov_b)
    for name, h in sorted(da - db):
        lines.append("数据集仅基线加载: {} ({})".format(name, h[:6]))
    for name, h in sorted(db - da):
        lines.append("数据集仅当前加载: {} ({})".format(name, h[:6]))
    return lines
