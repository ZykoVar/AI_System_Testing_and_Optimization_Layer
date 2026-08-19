"""Prompt A/B 测试：同一批用例在两个 Prompt 版本下对比运行。

原理：
- PromptManager.pin(prompt_id, version) 全局钉住默认渲染版本；
- 测试用例中未显式指定 version 的 render() 调用解析到被钉住的版本；
- 显式指定 version 的用例（如安全套件钉住 v2）不受影响。

流程：
1. 以 version_a 钉住运行选中用例 → 报告 A（含标准四格式报告）；
2. 以 version_b 钉住运行同一批用例 → 报告 B；
3. 逐用例对比判定（回归/改善）、失败消息、数值指标漂移；
4. 产出 A/B 对比报告：reports/abtest/abtest-<时间戳>-<prompt>-vA-vs-vB.{md,json}。
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmqa.core.models import Severity, TestContext, TestOutcome, Verdict
from llmqa.core.registry import discover
from llmqa.core.reporter import Reporter
from llmqa.core.runner import TestRunner
from llmqa.prompts.manager import PromptManager
from llmqa.suites import DEFAULT_PACKAGES

# 判定优劣分数：PASS 最好，ERROR 最差
_VERDICT_SCORE = {Verdict.PASS: 3, Verdict.SKIP: 2, Verdict.FAIL: 1, Verdict.ERROR: 0}


class ABChange(BaseModel):
    """单个用例在两个版本下的差异。"""
    case_id: str
    name: str
    suite: str
    severity: Severity
    verdict_a: Verdict
    verdict_b: Verdict
    direction: str = "unchanged"   # regression | improvement | metric_drift | message_change | unchanged
    message_a: str = ""
    message_b: str = ""
    metric_diffs: dict[str, dict[str, float]] = Field(default_factory=dict)   # 数值指标漂移（metric → {a,b}）


class ABTestResult(BaseModel):
    """一次 A/B 对比的汇总：计数、运行标识与逐用例差异列表。"""

    prompt_id: str
    version_a: int
    version_b: int
    provider: str
    run_id_a: str = ""
    run_id_b: str = ""
    total: int = 0
    unchanged: int = 0
    regressions: int = 0
    improvements: int = 0
    metric_drifts: int = 0
    changes: list[ABChange] = Field(default_factory=list)

    def summary_text(self) -> str:
        """一行中文摘要，供报告标题与日志输出使用。"""
        return ("Prompt {pid} v{a} vs v{b} | 共 {t} 例 | 未变化 {u} | 回归 {r} | "
                "改善 {i} | 指标漂移 {m}").format(
            pid=self.prompt_id, a=self.version_a, b=self.version_b, t=self.total,
            u=self.unchanged, r=self.regressions, i=self.improvements, m=self.metric_drifts)

    def by_direction(self, direction: str) -> list[ABChange]:
        """按方向（regression/improvement/metric_drift/message_change/unchanged）筛选差异。"""
        return [c for c in self.changes if c.direction == direction]


def _compare(a: TestOutcome, b: TestOutcome) -> ABChange:
    """比较同一用例在 A/B 两次运行的结果，判定方向（回归/改善/指标漂移/消息变化/不变）。"""
    base = dict(case_id=a.case_id, name=a.name, suite=a.suite, severity=a.severity,
                verdict_a=a.verdict, verdict_b=b.verdict, message_a=a.message, message_b=b.message)
    sa, sb = _VERDICT_SCORE[a.verdict], _VERDICT_SCORE[b.verdict]
    if sb < sa:
        return ABChange(**base, direction="regression")
    if sb > sa:
        return ABChange(**base, direction="improvement")
    # 判定相同：先看数值指标漂移，再看失败消息变化
    diffs: dict[str, dict[str, float]] = {}
    for k in sorted(set(a.metrics) | set(b.metrics)):
        va, vb = a.metrics.get(k), b.metrics.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            diffs[k] = {"a": float(va), "b": float(vb)}
    if diffs:
        return ABChange(**base, direction="metric_drift", metric_diffs=diffs)
    if a.verdict in (Verdict.FAIL, Verdict.ERROR) and a.message != b.message:
        return ABChange(**base, direction="message_change")
    return ABChange(**base, direction="unchanged")


def run_abtest(root: Path, settings: Any, pool: Any, datasets: Any,
               prompt_id: str, version_a: int, version_b: int, *,
               suites: set[str] | None = None, tags: set[str] | None = None,
               exclude_tags: set[str] | None = None,
               min_severity: Severity | None = None,
               concurrency: int | None = None, provider: str | None = None,
               include_demo: bool = False, progress: bool = True) -> ABTestResult:
    """执行 A/B 对比。root 为仓库根；settings/pool/datasets 复用调用方构造的资源。"""
    if version_a == version_b:
        raise ValueError("两个版本号相同，无法对比")
    packages = list(DEFAULT_PACKAGES)
    if include_demo:
        packages.append("llmqa.demo")
    cases = [c for c in discover(packages) if c.matches(
        suites=suites, tags=tags, exclude_tags=exclude_tags, min_severity=min_severity)]
    if not cases:
        raise ValueError("没有匹配的用例（请检查 suite/tag/severity 过滤条件）")
    provider_name = provider or settings.default_provider

    def make_ctx_factory(pin_version: int, tag: str):
        pm = PromptManager(root / "prompts").load()
        # 钉住全局默认渲染版本：未显式指定 version 的 render() 会解析到该版本。
        pm.pin(prompt_id, pin_version)

        def ctx_factory() -> TestContext:
            return TestContext(run_id=tag + "-" + uuid.uuid4().hex[:6],
                               settings=settings, providers=pool,
                               prompts=pm, datasets=datasets)
        return ctx_factory

    reporter = Reporter(root / settings.report_dir, no_color=True) if progress else None
    on_done = reporter.on_case_done if reporter else None

    def one_run(pin_version: int, tag: str):
        runner = TestRunner(make_ctx_factory(pin_version, tag),
                            concurrency=concurrency or settings.concurrency,
                            retries_on_error=settings.retries_on_error,
                            default_timeout=settings.timeout_per_test,
                            progress=on_done)
        report = runner.run_sync(cases, provider_name=provider_name)
        if reporter is not None:
            reporter.finalize(report)   # 每次运行都留标准报告，可审计
        return report

    report_a = one_run(version_a, "ab-a")
    report_b = one_run(version_b, "ab-b")
    outcomes_a = {o.case_id: o for o in report_a.outcomes}
    outcomes_b = {o.case_id: o for o in report_b.outcomes}
    # 只对比两次运行都实际产出的用例，避免某侧缺失被误判为回归。
    common = sorted(set(outcomes_a) & set(outcomes_b))
    changes = [_compare(outcomes_a[cid], outcomes_b[cid]) for cid in common]
    result = ABTestResult(
        prompt_id=prompt_id, version_a=version_a, version_b=version_b,
        provider=provider_name, run_id_a=report_a.run_id, run_id_b=report_b.run_id,
        total=len(common),
        unchanged=sum(1 for c in changes if c.direction == "unchanged"),
        regressions=sum(1 for c in changes if c.direction == "regression"),
        improvements=sum(1 for c in changes if c.direction == "improvement"),
        metric_drifts=sum(1 for c in changes if c.direction == "metric_drift"),
        changes=changes,
    )
    return result


def render_ab_report(result: ABTestResult, out_dir: Path) -> dict[str, Path]:
    """输出 A/B 对比报告（markdown + json）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # prompt_id 中的 "/" 被替换为 "-"，避免路径嵌套导致文件名无法生成。
    stem = "abtest-{}-{}-v{}-vs-v{}".format(
        stamp, result.prompt_id.replace("/", "-"), result.version_a, result.version_b)
    md_path = out_dir / (stem + ".md")
    json_path = out_dir / (stem + ".json")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(
        result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def _render_markdown(result: ABTestResult) -> str:
    """把 A/B 结果渲染为 Markdown 报告正文。"""
    lines = [
        "# Prompt A/B 测试报告",
        "",
        "- Prompt: {}  **v{} vs v{}**".format(result.prompt_id, result.version_a, result.version_b),
        "- Provider: {}".format(result.provider),
        "- 运行 A: {}（v{}）".format(result.run_id_a, result.version_a),
        "- 运行 B: {}（v{}）".format(result.run_id_b, result.version_b),
        "- 结论: {}".format(result.summary_text()),
        "",
    ]

    def section(title: str, changes: list[ABChange]) -> None:
        lines.append("## {}（{} 例）".format(title, len(changes)))
        lines.append("")
        if not changes:
            lines.append("无")
        for c in changes:
            lines.append("- **{}** {}（{}）: {} → {}".format(
                c.case_id, c.name, c.severity.value,
                c.verdict_a.value, c.verdict_b.value))
            if c.message_a or c.message_b:
                lines.append("  - A: {}".format(c.message_a[:160]))
                lines.append("  - B: {}".format(c.message_b[:160]))
            for metric, pair in c.metric_diffs.items():
                lines.append("  - {}: {} → {}".format(metric, pair["a"], pair["b"]))
        lines.append("")

    section("回归（A 通过/更优 → B 失败/更差）", result.by_direction("regression"))
    section("改善", result.by_direction("improvement"))
    section("指标漂移", result.by_direction("metric_drift"))
    section("失败消息变化", result.by_direction("message_change"))
    lines += ["## 全部对照", "",
              "| 用例 | 套件 | 严重级 | v{} | v{} | 方向 |".format(
                  result.version_a, result.version_b),
              "| --- | --- | --- | --- | --- | --- |"]
    for c in result.changes:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            c.case_id, c.suite, c.severity.value, c.verdict_a.value,
            c.verdict_b.value, c.direction))
    return "\n".join(lines) + "\n"
