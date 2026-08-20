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
