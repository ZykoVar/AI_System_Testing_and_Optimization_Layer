"""临时分析脚本：解析一次运行的报告关键数据（用后可删）。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "reports/20260908-170533-50a8/report.json"
d = json.load(open(path, encoding="utf-8"))
print("=== 溯源 ===")
p = d.get("provenance") or {}
print("model:", p.get("model"), "| commit:", (p.get("git_commit") or "")[:8],
      "| dirty:", p.get("git_dirty"))
print("prompts:", p.get("prompts_used"))
print("datasets:", len(p.get("datasets_used", [])), "个")
print()
print("=== 耗时 TOP 10 ===")
for o in sorted(d["outcomes"], key=lambda o: -o["duration_ms"])[:10]:
    print("  {:<18} {:>8.0f} ms  {}".format(o["case_id"], o["duration_ms"], o["name"][:22]))
print()
with_metrics = [o for o in d["outcomes"] if o.get("metrics")]
print("=== 带指标的用例:", len(with_metrics), "例 ===")
for o in with_metrics[:12]:
    print("  ", o["case_id"], o["metrics"])
with_ev = [o for o in d["outcomes"] if o.get("evidence")]
print("有证据的用例:", len(with_ev), "例")
total = sum(o["duration_ms"] for o in d["outcomes"])
print("总耗时: {:.0f}s / 136 例".format(total / 1000))
