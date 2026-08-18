"""CI 严重级门禁：解析最新 report.json，按严重级判定退出码。

退出码：0=通过；1=CRITICAL/HIGH 失败（拦截）；2=MEDIUM 失败（告警）。
用法：python scripts/gate.py reports
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def latest_report(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.glob("*/report.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("gate: 未找到任何 report.json")
        sys.exit(3)
    return candidates[0]


def main() -> int:
    reports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports")
    report_path = latest_report(reports_dir)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    print("gate: 检查报告 {}".format(report_path))
    print("gate: 通过 {PASS} 失败 {FAIL} 错误 {ERROR} 跳过 {SKIP} 通过率 {rate:.1%}".format(
        rate=data["pass_rate"], **data["counts"]))
    critical = []
    medium = []
    low = []
    for o in data["outcomes"]:
        if o["verdict"] not in ("FAIL", "ERROR"):
            continue
        sev = o["severity"]
        (critical if sev in ("CRITICAL", "HIGH") else
         medium if sev == "MEDIUM" else low).append(o)
    for o in critical:
        print("  [拦截] {} {} — {}: {}".format(o["severity"], o["case_id"], o["name"], o["message"][:100]))
    for o in medium:
        print("  [告警] {} {} — {}: {}".format(o["severity"], o["case_id"], o["name"], o["message"][:100]))
    if critical:
        print("gate: {} 个 CRITICAL/HIGH 失败 → 拦截".format(len(critical)))
        return 1
    if medium:
        print("gate: {} 个 MEDIUM 失败 → 告警".format(len(medium)))
        return 2
    print("gate: 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
