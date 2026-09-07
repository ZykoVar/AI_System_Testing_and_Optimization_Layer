"""报告器自测。"""
import json
import xml.etree.ElementTree as ET

from llmqa.core import Severity, TestOutcome, TestReport, Verdict


def make_report():
    """构造含 PASS/FAIL/ERROR 三种裁决的样例报告，供各格式断言复用。"""
    outcomes = [
        TestOutcome(case_id="a-1", name="通过", suite="unit", severity=Severity.LOW,
                    verdict=Verdict.PASS, message="通过", duration_ms=10),
        TestOutcome(case_id="a-2", name="失败", suite="unit", severity=Severity.HIGH,
                    verdict=Verdict.FAIL, message="缺少关键词", duration_ms=20),
        TestOutcome(case_id="a-3", name="错误", suite="unit", severity=Severity.MEDIUM,
                    verdict=Verdict.ERROR, message="超时", duration_ms=30,
                    traceback="Traceback..."),
    ]
    return TestReport(run_id="test-run", provider="mock",
                      started_at="2025-01-01T00:00:00", outcomes=outcomes)


def test_summary_text():
    report = make_report()
    text = report.summary_text()
    assert "通过 1" in text
    assert "失败 1" in text


def test_reporter_writes_all_formats(tmp_path):
    from llmqa.core.reporter import Reporter
    reporter = Reporter(tmp_path, no_color=True)
    files = reporter.finalize(make_report())
    for name in ("json", "markdown", "html", "junit"):
        assert name in files
        assert files[name].exists()


def test_junit_structure(tmp_path):
    from llmqa.core.reporter import Reporter
    reporter = Reporter(tmp_path, no_color=True)
    files = reporter.finalize(make_report())
    # JUnit XML 需可被 CI 解析：验证根元素与失败/错误计数
    tree = ET.parse(files["junit"])
    root = tree.getroot()
    assert root.tag == "testsuite"
    assert root.get("failures") == "1"
    assert root.get("errors") == "1"
    assert len(root.findall("testcase")) == 3


def test_json_report_metrics(tmp_path):
    from llmqa.core.reporter import Reporter
    reporter = Reporter(tmp_path, no_color=True)
    files = reporter.finalize(make_report())
    # JSON 报告按裁决聚合计数，且按原始顺序保留每条 outcome
    data = json.loads(files["json"].read_text(encoding="utf-8"))
    assert data["counts"]["PASS"] == 1
    assert data["outcomes"][1]["severity"] == "HIGH"


def test_json_report_carries_evidence(tmp_path):
    # evidence 端到端：TestOutcome → report.json 不得丢失
    from llmqa.core.reporter import Reporter
    from llmqa.core import TestReport
    outcome = TestOutcome(case_id="e-1", name="证据", suite="unit",
                          severity=Severity.HIGH, verdict=Verdict.FAIL,
                          message="低分", metrics={"judge_score": 3.0},
                          evidence=["裁判理由: 答案错误"])
    report = TestReport(run_id="r", provider="mock", started_at="t",
                        outcomes=[outcome])
    files = Reporter(tmp_path, no_color=True).finalize(report)
    data = json.loads(files["json"].read_text(encoding="utf-8"))
    assert data["outcomes"][0]["evidence"] == ["裁判理由: 答案错误"]
