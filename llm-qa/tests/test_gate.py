"""严重级门禁（scripts/gate.py）自测。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gate import gate  # noqa: E402


def make_report(tmp_path, outcomes):
    run_dir = tmp_path / "20250101-000000-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps({
        "counts": {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0},
        "pass_rate": 0.0,
        "outcomes": outcomes,
    }, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def case(sev, verdict="FAIL"):
    return {"case_id": "c-" + sev.lower(), "name": "x", "severity": sev,
            "verdict": verdict, "message": "失败"}


def test_high_threshold_blocks_critical_and_high(tmp_path, capsys):
    report = make_report(tmp_path, [
        case("CRITICAL"), case("HIGH"), case("MEDIUM"), case("LOW")])
    assert gate(report, "HIGH") == 1
    out = capsys.readouterr().out
    assert "2 个 ≥HIGH 失败" in out


def test_medium_warning_does_not_block_by_default(tmp_path, capsys):
    # P0-2 核心语义：默认阈值下 MEDIUM 只告警、退出 0——CI 不会在 gate 前误拦
    report = make_report(tmp_path, [case("MEDIUM"), case("LOW")])
    assert gate(report, "HIGH") == 0
    assert "仅告警" in capsys.readouterr().out


def test_medium_threshold_blocks_medium(tmp_path):
    report = make_report(tmp_path, [case("MEDIUM")])
    assert gate(report, "MEDIUM") == 1


def test_all_pass(tmp_path):
    report = make_report(tmp_path, [])
    assert gate(report, "HIGH") == 0
