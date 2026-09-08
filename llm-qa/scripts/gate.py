"""CI 严重级门禁：解析最新 report.json，按严重级阈值判定退出码。

退出码语义（CI 唯一的事实来源）：
- 0 = 通过（含低于阈值的失败，以 [告警] 打印但不拦截）
- 1 = 存在 ≥ 阈值的 FAIL/ERROR（拦截）
- 3 = 未找到任何报告（配置错误）

用法：
    python scripts/gate.py reports                    # 默认阈值 HIGH：CRITICAL/HIGH 拦截
    python scripts/gate.py reports --threshold MEDIUM # 收紧：MEDIUM 也拦截

退出码归属设计：CI 中 llmqa run 使用 --soft（执行结果不直接决定 job 成败），
由 gate.py 独占严重级判定；回归判定由 llmqa report compare 独占。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SEVERITY_RANK = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def latest_report(reports_dir: Path) -> Path:
    """按修改时间取最新一次运行的 report.json（排除 compare/baselines 子目录）。"""
    candidates = sorted(
        (p for p in reports_dir.glob("*/report.json")
         if p.parent.name not in ("compare", "baselines")),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("gate: 未找到任何 report.json")
        raise SystemExit(3)
    return candidates[0]


def gate(reports_dir: Path, threshold: str = "HIGH") -> int:
    """按严重级阈值判定门禁；返回 0=通过 / 1=拦截。可独立测试的纯逻辑。"""
    threshold_rank = _SEVERITY_RANK[threshold.upper()]
    report_path = latest_report(reports_dir)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    print("gate: 检查报告 {}（阈值 {}）".format(report_path, threshold.upper()))
    print("gate: 通过 {PASS} 失败 {FAIL} 错误 {ERROR} 跳过 {SKIP} 通过率 {rate:.1%}".format(
        rate=data["pass_rate"], **data["counts"]))
    blocking, warning = [], []
    for o in data["outcomes"]:
        if o["verdict"] not in ("FAIL", "ERROR"):
            continue
        sev = o["severity"]
        if _SEVERITY_RANK.get(sev, 0) >= threshold_rank:
            blocking.append(o)
        else:
            warning.append(o)
    for o in blocking:
        print("  [拦截] {} {} — {}: {}".format(
            o["severity"], o["case_id"], o["name"], o["message"][:100]))
    for o in warning:
        print("  [告警] {} {} — {}: {}".format(
            o["severity"], o["case_id"], o["name"], o["message"][:100]))
    if blocking:
        print("gate: {} 个 ≥{} 失败 → 拦截".format(len(blocking), threshold.upper()))
        return 1
    if warning:
        print("gate: {} 个低于阈值的失败 → 仅告警，不拦截".format(len(warning)))
    print("gate: 通过")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="严重级门禁")
    parser.add_argument("reports_dir", default="reports", nargs="?")
    parser.add_argument("--threshold", default="HIGH",
                        choices=["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        help="拦截阈值（默认 HIGH：CRITICAL/HIGH 拦截，MEDIUM 仅告警）")
    args = parser.parse_args()
    return gate(Path(args.reports_dir), args.threshold)


if __name__ == "__main__":
    sys.exit(main())
