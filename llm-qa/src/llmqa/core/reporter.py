"""报告器：控制台进度 + JSON / Markdown / HTML / JUnit XML 多格式输出。

输出目录：<report_dir>/<run_id>/report.{json,md,html} 与 junit.xml
"""
from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from llmqa.core.models import TestOutcome, Verdict
from llmqa.core.runner import TestReport

_COLORS = {
    Verdict.PASS: "\033[32m",   # 绿
    Verdict.FAIL: "\033[31m",   # 红
    Verdict.ERROR: "\033[35m",  # 紫
    Verdict.SKIP: "\033[33m",   # 黄
}
_RESET = "\033[0m"


def _ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，强制 UTF-8 输出避免 UnicodeEncodeError。"""
    import sys
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 —— 重定向等场景下忽略
        pass


class Reporter:
    """报告器：作为运行器的进度回调输出控制台，并负责落盘多格式报告。"""

    def __init__(self, report_dir: str | Path, *, no_color: bool = False):
        self.report_dir = Path(report_dir)
        self.no_color = no_color
        _ensure_utf8_stdout()

    def on_case_done(self, outcome: TestOutcome) -> None:
        """单个用例完成时的回调（传给 TestRunner 的 progress），实时打印一行结果。"""
        tag = outcome.verdict.value
        if not self.no_color:
            tag = _COLORS[outcome.verdict] + tag + _RESET
        line = "[{tag}] {sev:<8} {cid:<34} {name} ({dur:.0f}ms)".format(
            tag=tag, sev=outcome.severity.value, cid=outcome.case_id,
            name=outcome.name, dur=outcome.duration_ms)
        if outcome.verdict in (Verdict.FAIL, Verdict.ERROR):
            line += "\n      -> " + outcome.message[:200]
        print(line)

    def finalize(self, report: TestReport) -> dict[str, Path]:
        """落盘全部报告格式，返回 {格式: 路径}。"""
        out_dir = self.report_dir / report.run_id  # 每次运行独立子目录，避免相互覆盖。
        out_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        files["json"] = out_dir / "report.json"
        files["json"].write_text(json.dumps(
            _to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
        files["markdown"] = out_dir / "report.md"
        files["markdown"].write_text(_render_markdown(report), encoding="utf-8")
        files["html"] = out_dir / "report.html"
        files["html"].write_text(_render_html(report), encoding="utf-8")
        files["junit"] = out_dir / "junit.xml"
        files["junit"].write_text(_render_junit(report), encoding="utf-8")
        return files


def _to_dict(report: TestReport) -> dict:
    return {
        "run_id": report.run_id,
        "provider": report.provider,
        "started_at": report.started_at,
        "duration_ms": report.duration_ms,
        "counts": report.counts,
        "pass_rate": report.pass_rate,
        "outcomes": [o.model_dump(mode="json") for o in report.outcomes],
    }


def _render_markdown(report: TestReport) -> str:
    c = report.counts
    bt = chr(96)  # 反引号，避免转义混乱
    lines = [
        "# LLM-QA 测试报告 " + bt + report.run_id + bt,
        "",
        "- Provider: " + bt + report.provider + bt,
        "- 开始时间: " + report.started_at,
        "- 总耗时: {:.0f} ms".format(report.duration_ms),
        "- 通过率: **{:.1%}** （通过 {} / 失败 {} / 错误 {} / 跳过 {}）".format(
            report.pass_rate, c["PASS"], c["FAIL"], c["ERROR"], c["SKIP"]),
        "",
        "## 失败明细",
        "",
    ]
    fails = sorted(report.failures, key=lambda o: -o.severity.rank)
    if not fails:
        lines.append("无失败用例 ✅")
    for o in fails:
        lines.append("### [{v}] {s} — {i} {n}".format(
            v=o.verdict.value, s=o.severity.value, i=o.case_id, n=o.name))
        lines.append("- " + o.message)
        if o.metrics:
            lines.append("- 指标: " + bt + json.dumps(o.metrics, ensure_ascii=False) + bt)
        if o.traceback:
            fence = bt * 3
            lines.append(fence + "\n" + o.traceback + "\n" + fence)
        lines.append("")
    lines += ["## 全部结果", "", "| 用例 | 套件 | 严重级 | 结果 | 耗时 | 说明 |",
               "| --- | --- | --- | --- | --- | --- |"]
    for o in report.outcomes:
        msg = o.message.replace("|", "\\|")[:80]  # 转义竖线并截断，防止破坏 Markdown 表格列。
        lines.append("| {} | {} | {} | {} | {:.0f}ms | {} |".format(
            o.case_id, o.suite, o.severity.value, o.verdict.value, o.duration_ms, msg))
    return "\n".join(lines) + "\n"


def _render_html(report: TestReport) -> str:
    c = report.counts
    rows = []
    # HTML 输出须对所有用户文本转义，防止注入破坏页面结构。
    for o in sorted(report.outcomes, key=lambda o: -o.severity.rank):
        color = {"PASS": "#2e7d32", "FAIL": "#c62828", "ERROR": "#6a1b9a", "SKIP": "#f9a825"}[o.verdict.value]
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td>"
            "<td style='color:{};font-weight:600'>{}</td>"
            "<td>{:.0f}ms</td><td>{}</td></tr>".format(
                html.escape(o.case_id), o.suite, o.severity.value, color, o.verdict.value,
                o.duration_ms, html.escape(o.message[:150])))
    body = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>LLM-QA 报告 {run_id}</title>
<style>
body{{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;margin:32px;color:#222}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}}
.cards{{display:flex;gap:12px}} .card{{border:1px solid #ddd;border-radius:8px;padding:12px 20px;min-width:110px}}
.card b{{font-size:22px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}}
th,td{{border:1px solid #e0e0e0;padding:6px 10px;text-align:left}}
th{{background:#f5f5f5}}
</style></head><body>
<h1>LLM-QA 测试报告 <code>{run_id}</code></h1>
<p>Provider: {provider} · 开始: {started} · 耗时: {dur:.0f}ms</p>
<div class="cards">
<div class="card">通过<b>{p}</b></div><div class="card">失败<b>{f}</b></div>
<div class="card">错误<b>{e}</b></div><div class="card">跳过<b>{s}</b></div>
<div class="card">通过率<b>{rate:.0%}</b></div></div>
<h2>用例结果</h2>
<table><tr><th>用例</th><th>套件</th><th>严重级</th><th>结果</th><th>耗时</th><th>说明</th></tr>
{rows}</table>
</body></html>
""".format(
        run_id=report.run_id, provider=html.escape(report.provider),
        started=report.started_at, dur=report.duration_ms,
        p=c["PASS"], f=c["FAIL"], e=c["ERROR"], s=c["SKIP"], rate=report.pass_rate,
        rows="".join(rows))
    return body


def _render_junit(report: TestReport) -> str:
    """渲染 JUnit XML：FAIL→failure、ERROR→error、SKIP→skipped，便于 CI 采集。"""
    suite = ET.Element("testsuite", {
        "name": "llmqa", "tests": str(len(report.outcomes)),
        "failures": str(report.counts["FAIL"]), "errors": str(report.counts["ERROR"]),
        "skipped": str(report.counts["SKIP"]),
        "time": "{:.3f}".format(report.duration_ms / 1000),
    })
    for o in report.outcomes:
        case = ET.SubElement(suite, "testcase", {
            "classname": o.suite, "name": "{} — {}".format(o.case_id, o.name),
            "time": "{:.3f}".format(o.duration_ms / 1000),
        })
        if o.verdict == Verdict.FAIL:
            ET.SubElement(case, "failure", {"message": o.message[:500]}).text = o.message
        elif o.verdict == Verdict.ERROR:
            ET.SubElement(case, "error", {"message": o.message[:500]}).text = o.traceback or o.message
        elif o.verdict == Verdict.SKIP:
            ET.SubElement(case, "skipped").text = o.message
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)
