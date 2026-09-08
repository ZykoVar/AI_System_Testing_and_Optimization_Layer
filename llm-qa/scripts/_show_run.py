"""临时：打印一次运行的全部用例明细。"""
import json
import sys

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
print("run:", d["run_id"], "| provider:", d["provider"], "| counts:", d["counts"])
print()
for o in d["outcomes"]:
    status = o["verdict"]
    print("[{}] {:<14} ({:.0f}ms) {}".format(status, o["case_id"], o["duration_ms"], o["name"]))
    if o.get("metrics"):
        print("      metrics:", o["metrics"])
    if status in ("FAIL", "ERROR"):
        print("      message:", o["message"][:250])
    for e in (o.get("evidence") or [])[:3]:
        print("      evidence:", e[:200])
    print()
