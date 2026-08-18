"""llmqa 命令行入口。

用法示例：
    llmqa run --suite security --tag smoke --severity MEDIUM
    llmqa run --provider openai --concurrency 16
    llmqa prompts list / show / validate / scan / diff / promote
    llmqa datasets list
    llmqa demo
"""
from __future__ import annotations

import argparse
import sys


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmqa", description="企业级 LLM/Agent 质量保障测试框架")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="运行测试套件")
    p_run.add_argument("--suite", action="append", default=[],
                       help="只跑指定套件（llm/rag/agent/security/performance），可多次指定")
    p_run.add_argument("--tag", action="append", default=[], help="按标签过滤（需全部命中）")
    p_run.add_argument("--exclude-tag", action="append", default=[])
    p_run.add_argument("--severity", default=None,
                       help="最低严重级（INFO/LOW/MEDIUM/HIGH/CRITICAL）")
    p_run.add_argument("--provider", default=None,
                       help="Provider 名称（默认 settings.default_provider）")
    p_run.add_argument("--concurrency", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None, help="单用例超时（秒）")
    p_run.add_argument("--fail-fast", action="store_true")
    p_run.add_argument("--config", default=None, help="config 目录（默认仓库根/config）")
    p_run.add_argument("--report-dir", default=None)
    p_run.add_argument("--no-color", action="store_true")
    p_run.add_argument("--soft", action="store_true", help="失败时仍返回退出码 0")
    p_run.add_argument("--list", action="store_true", help="只列出匹配的用例不执行")

    p_pr = sub.add_parser("prompts", help="Prompt 管理")
    pr_sub = p_pr.add_subparsers(dest="prompt_action", required=True)
    pr_sub.add_parser("list", help="列出全部 Prompt")
    pr_sub.add_parser("validate", help="全库校验")
    pr_sub.add_parser("scan", help="全库注入扫描")
    p_show = pr_sub.add_parser("show", help="查看单个 Prompt")
    p_show.add_argument("prompt_id")
    p_show.add_argument("--version", type=int, default=None)
    p_diff = pr_sub.add_parser("diff", help="版本对比")
    p_diff.add_argument("prompt_id")
    p_diff.add_argument("v1", type=int)
    p_diff.add_argument("v2", type=int)
    p_promo = pr_sub.add_parser("promote", help="状态流转 draft/active/deprecated")
    p_promo.add_argument("prompt_id")
    p_promo.add_argument("status")
    p_promo.add_argument("--version", type=int, default=None)

    p_ds = sub.add_parser("datasets", help="数据集管理")
    p_ds.add_argument("--list", action="store_true", help="列出全部数据集")

    sub.add_parser("demo", help="运行内置演示套件（mock Provider）")
    return parser


def _run(args: argparse.Namespace) -> int:
    from llmqa.clients import ClientPool
    from llmqa.config import Settings, repo_root
    from llmqa.core.models import Severity, TestContext
    from llmqa.core.registry import discover
    from llmqa.core.reporter import Reporter
    from llmqa.core.runner import TestRunner
    from llmqa.datasets import DatasetManager
    from llmqa.prompts import PromptManager
    from llmqa.suites import DEFAULT_PACKAGES

    root = repo_root()
    settings = Settings.load(args.config or (root / "config"))
    prompts = PromptManager(root / "prompts").load()
    datasets = DatasetManager(root / "datasets")
    pool = ClientPool(settings)
    provider_name = args.provider or settings.default_provider

    def ctx_factory() -> TestContext:
        import uuid
        return TestContext(run_id="run-" + uuid.uuid4().hex[:8],
                           settings=settings, providers=pool,
                           prompts=prompts, datasets=datasets)

    cases = discover(DEFAULT_PACKAGES)
    min_sev = Severity(args.severity.upper()) if args.severity else None
    cases = [c for c in cases if c.matches(
        suites=set(args.suite) or None, tags=set(args.tag) or None,
        exclude_tags=set(args.exclude_tag) or None, min_severity=min_sev)]

    if args.list:
        for c in sorted(cases, key=lambda c: (c.suite, c.id)):
            print("{:<12} {:<38} {} [{}]".format(
                c.suite, c.id, c.name, ",".join(sorted(c.tags)) or "-"))
        return 0
    print("匹配 {} 个用例，Provider: {}".format(len(cases), provider_name))

    reporter = Reporter(args.report_dir or (root / settings.report_dir),
                        no_color=args.no_color)
    runner = TestRunner(
        ctx_factory,
        concurrency=args.concurrency or settings.concurrency,
        fail_fast=args.fail_fast or settings.fail_fast,
        retries_on_error=settings.retries_on_error,
        default_timeout=args.timeout or settings.timeout_per_test,
        progress=reporter.on_case_done,
    )
    report = runner.run_sync(cases, provider_name=provider_name)
    files = reporter.finalize(report)
    print()
    print(report.summary_text())
    print("报告: " + ", ".join("{} → {}".format(k, v) for k, v in files.items()))
    if args.soft:
        return 0
    return 0 if not report.failures else 1


def _prompts(args: argparse.Namespace) -> int:
    from llmqa.config import repo_root
    from llmqa.prompts import PromptManager, PromptScanner
    root = repo_root()
    manager = PromptManager(root / "prompts").load()
    action = args.prompt_action
    if action == "list":
        for t in manager.list():
            print("{:<28} v{:<3} {:<10} {}".format(t.id, t.version, t.status, t.name))
    elif action == "validate":
        problems = manager.validate()
        if problems:
            print("发现 {} 个问题:".format(len(problems)))
            for p in problems:
                print("  - " + p)
            return 1
        print("校验通过：全部 Prompt 变量声明一致")
    elif action == "scan":
        scanner = PromptScanner()
        total = 0
        for pid, report in scanner.scan_library(manager).items():
            if report.findings:
                total += len(report.findings)
                print("{} 风险 {}：".format(pid, report.highest_risk))
                for f in report.findings:
                    print("  [{}/{}] {} → {}".format(f.risk, f.rule, f.location, f.matched))
        print("共 {} 条风险发现".format(total))
        return 0 if total == 0 else 1
    elif action == "show":
        t = manager.get(args.prompt_id, args.version)
        print("id: {}  name: {}  version: {}  status: {}".format(
            t.id, t.name, t.version, t.status))
        for m in t.messages:
            print("--- [{}] ---".format(m.role))
            print(m.content)
    elif action == "diff":
        print(manager.diff(args.prompt_id, args.v1, args.v2))
    elif action == "promote":
        manager.promote(args.prompt_id, args.status, args.version)
        print("已流转 {} → {}".format(args.prompt_id, args.status))
    return 0


def _datasets(args: argparse.Namespace) -> int:
    from llmqa.config import repo_root
    from llmqa.datasets import DatasetManager
    for name in DatasetManager(repo_root() / "datasets").list():
        print(name)
    return 0


def _demo(args: argparse.Namespace) -> int:
    from llmqa import demo  # noqa: F401 —— 导入即注册演示用例
    return demo.run()


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "prompts":
        return _prompts(args)
    if args.command == "datasets":
        return _datasets(args)
    if args.command == "demo":
        return _demo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
