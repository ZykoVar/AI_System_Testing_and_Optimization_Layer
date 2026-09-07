"""运行间结果对比（core.compare）自测。"""
from llmqa.core import Severity, TestOutcome, Verdict
from llmqa.core.compare import compare_outcomes, summarize_diffs


def outcome(case_id, verdict, message="", metrics=None):
    return TestOutcome(case_id=case_id, name=case_id, suite="unit",
                       severity=Severity.MEDIUM, verdict=verdict,
                       message=message, metrics=metrics or {})


def test_regression_and_improvement():
    # A 通过、B 失败 → 回归；反之 → 改善
    reg = compare_outcomes(outcome("c1", Verdict.PASS), outcome("c1", Verdict.FAIL, "缺陷"))
    assert reg.direction == "regression"
    imp = compare_outcomes(outcome("c1", Verdict.FAIL, "缺陷"), outcome("c1", Verdict.PASS))
    assert imp.direction == "improvement"


def test_metric_drift_same_verdict():
    a = outcome("c2", Verdict.PASS, metrics={"p95_ms": 100.0, "recall": 0.9})
    b = outcome("c2", Verdict.PASS, metrics={"p95_ms": 180.0, "recall": 0.9})
    d = compare_outcomes(a, b)
    assert d.direction == "metric_drift"
    assert d.metric_diffs["p95_ms"] == {"a": 100.0, "b": 180.0}
    assert "recall" not in d.metric_diffs          # 未变化的指标不进入漂移清单


def test_message_change_and_unchanged():
    a = outcome("c3", Verdict.FAIL, "缺少关键词")
    b = outcome("c3", Verdict.FAIL, "JSON 解析失败")
    assert compare_outcomes(a, b).direction == "message_change"
    assert compare_outcomes(a, outcome("c3", Verdict.FAIL, "缺少关键词")).direction == "unchanged"


def test_summarize():
    diffs = [
        compare_outcomes(outcome("a", Verdict.PASS), outcome("a", Verdict.FAIL, "x")),
        compare_outcomes(outcome("b", Verdict.FAIL, "x"), outcome("b", Verdict.PASS)),
        compare_outcomes(outcome("c", Verdict.PASS, metrics={"m": 1.0}),
                         outcome("c", Verdict.PASS, metrics={"m": 2.0})),
        compare_outcomes(outcome("d", Verdict.PASS), outcome("d", Verdict.PASS)),
    ]
    counts = summarize_diffs(diffs)
    assert counts["total"] == 4
    assert counts["regressions"] == 1
    assert counts["improvements"] == 1
    assert counts["metric_drifts"] == 1
    assert counts["unchanged"] == 1
    assert counts["neutrals"] == 0


def test_neutral_skip_not_regression():
    # budget/fail_fast 跳过是"未执行"而非"变差"，不计回归
    a = outcome("n1", Verdict.PASS)
    b = TestOutcome(case_id="n1", name="n1", suite="unit",
                    severity=Severity.MEDIUM, verdict=Verdict.SKIP,
                    message="成本预算耗尽", skip_reason="budget")
    assert compare_outcomes(a, b).direction == "neutral"
    c = TestOutcome(case_id="n1", name="n1", suite="unit",
                    severity=Severity.MEDIUM, verdict=Verdict.SKIP,
                    message="fail-fast 未执行", skip_reason="fail_fast")
    assert compare_outcomes(c, a).direction == "neutral"


def test_intentional_skip_is_still_regression():
    # 有意跳过（覆盖丢失）仍按回归计，便于发现用例被悄悄禁用
    a = outcome("n2", Verdict.PASS)
    b = TestOutcome(case_id="n2", name="n2", suite="unit",
                    severity=Severity.MEDIUM, verdict=Verdict.SKIP,
                    message="环境不满足", skip_reason="intentional")
    assert compare_outcomes(a, b).direction == "regression"


def test_metric_policy_tolerance():
    policies = {"judge_score": {"direction": "higher_better", "tolerance": 0.3}}
    # 8.1 → 7.1 超出容差 → 指标回归
    d = compare_outcomes(outcome("p1", Verdict.PASS, metrics={"judge_score": 8.1}),
                         outcome("p1", Verdict.PASS, metrics={"judge_score": 7.1}),
                         policies=policies)
    assert d.direction == "regression"
    assert d.metric_signs["judge_score"] == "regression"
    # 8.1 → 8.0 在容差内 → 不显著
    d2 = compare_outcomes(outcome("p1", Verdict.PASS, metrics={"judge_score": 8.1}),
                          outcome("p1", Verdict.PASS, metrics={"judge_score": 8.0}),
                          policies=policies)
    assert d2.direction == "unchanged"


def test_metric_policy_relative_tolerance():
    policies = {"p95_ms": {"direction": "lower_better", "tolerance": "20%"}}
    # 100 → 115（+15%）在 20% 容差内 → 不显著
    d = compare_outcomes(outcome("p2", Verdict.PASS, metrics={"p95_ms": 100.0}),
                         outcome("p2", Verdict.PASS, metrics={"p95_ms": 115.0}),
                         policies=policies)
    assert d.direction == "unchanged"
    # 100 → 150（+50%）越界且方向劣化 → 回归
    d2 = compare_outcomes(outcome("p2", Verdict.PASS, metrics={"p95_ms": 100.0}),
                          outcome("p2", Verdict.PASS, metrics={"p95_ms": 150.0}),
                          policies=policies)
    assert d2.direction == "regression"
    # 150 → 100（方向改善）→ improvement
    d3 = compare_outcomes(outcome("p2", Verdict.PASS, metrics={"p95_ms": 150.0}),
                          outcome("p2", Verdict.PASS, metrics={"p95_ms": 100.0}),
                          policies=policies)
    assert d3.direction == "improvement"
