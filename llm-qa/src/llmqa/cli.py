"""llmqa 命令行入口。

用法示例：
    llmqa run --suite security --tag smoke --severity MEDIUM
    llmqa run --provider openai --concurrency 16
    llmqa prompts list / show / validate / scan / diff / promote
    llmqa prompts ab-test <prompt_id> <版本A> <版本B> [--include-demo]
    llmqa datasets list
    llmqa demo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_root(config_dir: str | None = None) -> Path:
    """--config 指向 config 目录；未提供时自动定位仓库根（含安装包回退）。"""
    from llmqa.config import repo_root
    if config_dir:
        # --config 传入的是 config 目录本身，仓库根取其父目录
        return Path(config_dir).resolve().parent
    return repo_root()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造参数解析器：run / prompts / datasets / demo 四个子命令。"""
    parser = argparse.ArgumentParser(
        prog="llmqa", description="企业级 LLM/Agent 质量保障测试框架")
    parser.add_argument("--config", default=None,
                        help="config 目录路径（默认自动定位仓库根/config，支持任意目录运行）")
    # required=True：不带任何子命令时直接报错，避免静默无操作
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
    p_run.add_argument("--max-cost", type=int, default=None,
                       help="成本预算上限（按用例声明的 cost 单位累计，超预算用例 SKIP）")
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

    p_ab = pr_sub.add_parser("ab-test", help="同一套用例在两个 Prompt 版本下对比运行")
    p_ab.add_argument("prompt_id")
    p_ab.add_argument("version_a", type=int)
    p_ab.add_argument("version_b", type=int)
    p_ab.add_argument("--suite", action="append", default=[])
    p_ab.add_argument("--tag", action="append", default=[])
    p_ab.add_argument("--exclude-tag", action="append", default=[])
    p_ab.add_argument("--severity", default=None)
    p_ab.add_argument("--provider", default=None)
    p_ab.add_argument("--concurrency", type=int, default=None)
    p_ab.add_argument("--include-demo", action="store_true",
                      help="把内置演示用例纳入对比（demo-006 对版本敏感）")

    p_ds = sub.add_parser("datasets", help="数据集管理")
    p_ds.add_argument("--list", action="store_true", help="列出全部数据集")

    p_rep = sub.add_parser("report", help="报告查看与运行间对比")
    rep_sub = p_rep.add_subparsers(dest="report_action", required=True)
    rep_sub.add_parser("list", help="列出历史运行")
    p_cmp = rep_sub.add_parser("compare", help="对比两次运行（A=基线/旧，B=当前/新）")
    p_cmp.add_argument("run_a", nargs="?", default=None, help="基线运行 ID（reports/ 下目录名）")
    p_cmp.add_argument("run_b", nargs="?", default=None, help="当前运行 ID")
    p_cmp.add_argument("--last", action="store_true", help="自动取最近两次运行对比")
    p_cmp.add_argument("--baseline", action="store_true",
                       help="A 侧取已登记的基线（llmqa report baseline-set 设置）")
    p_cmp.add_argument("--baseline-name", default="production",
                       help="基线名称（默认 production，配合 --baseline 使用）")
    p_cmp.add_argument("--report-dir", default=None, help="报告根目录（默认 settings.report_dir）")
    p_bset = rep_sub.add_parser("baseline-set", help="把某次运行登记为命名基线")
    p_bset.add_argument("run_id", help="运行 ID（reports/ 下目录名）")
    p_bset.add_argument("--name", default="production",
                        help="基线名称（production/staging/security/performance 等）")
    p_bset.add_argument("--report-dir", default=None)
    p_bshow = rep_sub.add_parser("baseline-show", help="查看基线")
    p_bshow.add_argument("--name", default="production", help="基线名称")
    p_bshow.add_argument("--report-dir", default=None)
    rep_sub.add_parser("baseline-list", help="列出全部命名基线")
    p_bens = rep_sub.add_parser("baseline-ensure", help="基线不存在时用最近一次运行引导登记（CI 首跑用）")
    p_bens.add_argument("--name", default="production", help="基线名称")
    p_bens.add_argument("--report-dir", default=None)

    sub.add_parser("demo", help="运行内置演示套件（mock Provider）")
    return parser


def _run(args: argparse.Namespace) -> int:
    """执行 run 子命令：加载配置→发现用例→过滤→并发运行→产出报告并返回退出码。"""
    from llmqa.clients import ClientPool
    from llmqa.config import Settings
    from llmqa.core.models import Severity, TestContext
    from llmqa.core.registry import discover
    from llmqa.core.reporter import Reporter
    from llmqa.core.runner import TestRunner
    from llmqa.datasets import DatasetManager
    from llmqa.prompts import PromptManager
    from llmqa.suites import DEFAULT_PACKAGES

    root = _resolve_root(args.config)
    # config 未显式指定时回退到仓库默认 config 目录
    settings = Settings.load(args.config or (root / "config"))
    prompts = PromptManager(root / "prompts").load()
    datasets = DatasetManager(root / "datasets")
    pool = ClientPool(settings)
    # CLI 显式指定优先，否则用 settings 默认 Provider
    provider_name = args.provider or settings.default_provider

    def ctx_factory() -> TestContext:
        import uuid
        return TestContext(run_id="run-" + uuid.uuid4().hex[:8],
                           settings=settings, providers=pool,
                           prompts=prompts, datasets=datasets)

    cases = discover(DEFAULT_PACKAGES)
    # 未指定严重级时传 None 表示不过滤；统一转大写以匹配枚举名
    min_sev = Severity(args.severity.upper()) if args.severity else None
    # 空集合折叠为 None，语义是"该维度不过滤"
    cases = [c for c in cases if c.matches(
        suites=set(args.suite) or None, tags=set(args.tag) or None,
        exclude_tags=set(args.exclude_tag) or None, min_severity=min_sev)]

    if args.list:
        # --list 只列出匹配用例，不执行，直接返回
        for c in sorted(cases, key=lambda c: (c.suite, c.id)):
            print("{:<12} {:<38} {} [{}]".format(
                c.suite, c.id, c.name, ",".join(sorted(c.tags)) or "-"))
        return 0
    print(f"匹配 {len(cases)} 个用例，Provider: {provider_name}")

    reporter = Reporter(args.report_dir or (root / settings.report_dir),
                        no_color=args.no_color)
    runner = TestRunner(
        ctx_factory,
        concurrency=args.concurrency or settings.concurrency,
        fail_fast=args.fail_fast or settings.fail_fast,
        retries_on_error=settings.retries_on_error,
        # 显式 --timeout 优先，否则回退到配置默认
        default_timeout=args.timeout or settings.timeout_per_test,
        progress=reporter.on_case_done,
        max_cost=args.max_cost,
    )
    report = runner.run_sync(cases, provider_name=provider_name)
    # 运行溯源：git commit / Prompt 版本+指纹 / 数据集指纹 / 模型 / 用例指纹
    from llmqa.core.provenance import attach_provenance
    attach_provenance(report, root, prompts, datasets,
                      settings=settings, provider_name=provider_name, cases=cases)
    files = reporter.finalize(report)
    pool.close_sync()   # 运行结束释放真实 Provider 连接池（按 run 生命周期而非进程）
    print()
    print(report.summary_text())
    print("报告: " + ", ".join(f"{k} → {v}" for k, v in files.items()))
    if args.soft:
        # --soft：无论是否失败都返回 0，供门禁外的软性检查使用
        return 0
    return 0 if not report.failures else 1


def _prompts(args: argparse.Namespace) -> int:
    """执行 prompts 子命令：按 prompt_action 分派到各 Prompt 管理操作。"""
    from llmqa.prompts import PromptManager, PromptScanner
    root = _resolve_root(args.config)
    manager = PromptManager(root / "prompts").load()
    action = args.prompt_action
    if action == "list":
        for t in manager.list():
            print(f"{t.id:<28} v{t.version:<3} {t.status:<10} {t.name}")
    elif action == "validate":
        problems = manager.validate()
        if problems:
            print(f"发现 {len(problems)} 个问题:")
            for p in problems:
                print("  - " + p)
            return 1
        print("校验通过：全部 Prompt 变量声明一致")
    elif action == "scan":
        scanner = PromptScanner()
        total = 0
        # 跨全库累计风险条数，用于最终退出码判定
        for pid, report in scanner.scan_library(manager).items():
            if report.findings:
                total += len(report.findings)
                print(f"{pid} 风险 {report.highest_risk}：")
                for f in report.findings:
                    print(f"  [{f.risk}/{f.rule}] {f.location} → {f.matched}")
        print(f"共 {total} 条风险发现")
        return 0 if total == 0 else 1
    elif action == "show":
        t = manager.get(args.prompt_id, args.version)
        print(f"id: {t.id}  name: {t.name}  version: {t.version}  status: {t.status}")
        for m in t.messages:
            print(f"--- [{m.role}] ---")
            print(m.content)
    elif action == "diff":
        print(manager.diff(args.prompt_id, args.v1, args.v2))
    elif action == "promote":
        manager.promote(args.prompt_id, args.status, args.version)
        print(f"已流转 {args.prompt_id} → {args.status}")
    elif action == "ab-test":
        return _prompts_abtest(args)
    return 0


def _prompts_abtest(args: argparse.Namespace) -> int:
    """A/B 对比：两个 Prompt 版本各跑一遍同一批用例，产出对比报告。"""
    from llmqa.clients import ClientPool
    from llmqa.config import Settings
    from llmqa.core.models import Severity
    from llmqa.datasets import DatasetManager
    from llmqa.prompts import render_ab_report, run_abtest

    root = _resolve_root(args.config)
    settings = Settings.load(args.config or (root / "config"))
    pool = ClientPool(settings)
    datasets = DatasetManager(root / "datasets")
    min_sev = Severity(args.severity.upper()) if args.severity else None
    result = run_abtest(
        root, settings, pool, datasets, args.prompt_id,
        args.version_a, args.version_b,
        suites=set(args.suite) or None, tags=set(args.tag) or None,
        exclude_tags=set(args.exclude_tag) or None, min_severity=min_sev,
        concurrency=args.concurrency, provider=args.provider,
        include_demo=args.include_demo)
    files = render_ab_report(result, root / settings.report_dir / "abtest")
    print()
    print(result.summary_text())
    print("A/B 报告: " + ", ".join(f"{k} → {v}" for k, v in files.items()))
    print(f"标准报告: reports/{result.run_id_a} 与 reports/{result.run_id_b}")
    # 出现回归即判失败，供 CI 门禁使用
    return 1 if result.regressions else 0


def _report(args: argparse.Namespace) -> int:
    """report 子命令：list 历史运行 / compare 两次运行回归对比。"""
    import datetime as dt
    import json

    from llmqa.config import Settings
    from llmqa.core.compare import compare_outcomes, summarize_diffs
    from llmqa.core.models import TestOutcome

    root = _resolve_root(args.config)
    settings = Settings.load(args.config or (root / "config"))
    # --report-dir 只在 compare/baseline-set 子命令定义，list 等无此参数，故用 getattr 兜底
    report_dir = Path(getattr(args, "report_dir", None) or "") if getattr(
        args, "report_dir", None) else (root / settings.report_dir)

    # ---- baseline 管理：命名基线（production/staging/security/performance 并存） ----
    baselines_dir = report_dir / "baselines"
    legacy_file = report_dir / ".baseline.json"   # 旧版单基线文件，首次访问时迁移

    def baseline_path(name: str) -> Path:
        return baselines_dir / (name + ".json")

    def migrate_legacy() -> None:
        """旧 .baseline.json → baselines/production.json（一次性迁移）。"""
        if legacy_file.exists() and not baseline_path("production").exists():
            data = json.loads(legacy_file.read_text(encoding="utf-8"))
            _write_baseline("production", data.get("run_id", ""),
                            set_at=data.get("set_at", "?"))
            legacy_file.unlink(missing_ok=True)

    def _write_baseline(name: str, run_id: str, *, set_at: str | None = None) -> None:
        """从运行报告提取溯源信息，写出富元数据基线文件。"""
        baselines_dir.mkdir(parents=True, exist_ok=True)
        run_path = report_dir / run_id / "report.json"
        report_data = json.loads(run_path.read_text(encoding="utf-8"))
        prov = report_data.get("provenance") or {}
        created_by = (os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
        baseline_path(name).write_text(json.dumps({
            "schema_version": 1,
            "name": name,
            "run_id": run_id,
            "git_commit": prov.get("git_commit", ""),
            "prompts_used": prov.get("prompts_used", []),
            "datasets_used": prov.get("datasets_used", []),
            "created_at": set_at or dt.datetime.now().isoformat(timespec="seconds"),
            "created_by": created_by,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.report_action in ("baseline-set", "baseline-show", "baseline-list",
                               "baseline-ensure"):
        migrate_legacy()
        if args.report_action == "baseline-ensure":
            # CI 首跑引导：基线缺失时用最近一次运行登记；已存在则无操作
            path = baseline_path(args.name)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                print("基线已存在: {} → {}（无操作）".format(args.name, data.get("run_id")))
                return 0
            runs = sorted((d for d in report_dir.glob("*/report.json")
                           if d.parent.name not in ("compare", "baselines")),
                          key=lambda p: p.stat().st_mtime)
            if not runs:
                print("没有可引导基线的运行记录")
                return 2
            _write_baseline(args.name, runs[-1].parent.name)
            print("引导基线: {} → {}（首次运行自动登记）".format(
                args.name, runs[-1].parent.name))
            return 0
        if args.report_action == "baseline-list":
            if not baselines_dir.exists():
                print("尚未登记任何基线")
                return 2
            for p in sorted(baselines_dir.glob("*.json")):
                data = json.loads(p.read_text(encoding="utf-8"))
                commit = (data.get("git_commit") or "")[:8]
                print("{:<14} run={}  commit={}  at={}  by={}".format(
                    data.get("name"), data.get("run_id"), commit or "-",
                    data.get("created_at", "?"), data.get("created_by", "?")))
            return 0
        if args.report_action == "baseline-show":
            path = baseline_path(args.name)
            if not path.exists():
                print("基线 {} 不存在（llmqa report baseline-set <run_id> --name {}）".format(
                    args.name, args.name))
                return 2
            data = json.loads(path.read_text(encoding="utf-8"))
            print("基线: {}  run: {}  commit: {}  at: {}  by: {}".format(
                data.get("name"), data.get("run_id"),
                (data.get("git_commit") or "-")[:8],
                data.get("created_at", "?"), data.get("created_by", "?")))
            if data.get("prompts_used"):
                print("Prompt: " + ", ".join(
                    "{}:v{}@{}".format(u.get("id"), u.get("version"),
                                       (u.get("content_hash") or "")[:6])
                    for u in data["prompts_used"]))
            return 0
        # baseline-set
        path = report_dir / args.run_id / "report.json"
        if not path.exists():
            print("报告不存在: " + str(path))
            return 2
        _write_baseline(args.name, args.run_id)
        print("已登记基线 {} → {}".format(args.name, args.run_id))
        return 0

    # ---- list：按时间倒序列出全部运行 ----
    if args.report_action == "list":
        runs = sorted((d for d in report_dir.glob("*/report.json") if d.parent.name != "compare"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"共 {len(runs)} 次运行:")
        for p in runs:
            data = json.loads(p.read_text(encoding="utf-8"))
            counts = data["counts"]
            print("  {:<22} {}  通过 {}/{}/{}/{}/{:<3} 通过率 {:.0%}".format(
                p.parent.name, data.get("provider", "?"), counts["PASS"], counts["FAIL"],
                counts["ERROR"], counts["SKIP"], len(data["outcomes"]), data["pass_rate"]))
        return 0

    # ---- compare：A（基线） vs B（当前） ----
    if args.last:
        runs = sorted((d for d in report_dir.glob("*/report.json") if d.parent.name != "compare"),
                      key=lambda p: p.stat().st_mtime)
        if len(runs) < 2:
            print("历史运行不足 2 次，无法对比")
            return 2
        run_a, run_b = runs[-2].parent.name, runs[-1].parent.name
    elif args.baseline:
        # A 侧取命名基线，B 侧取指定运行或最近一次
        migrate_legacy()
        path = baseline_path(args.baseline_name)
        if not path.exists():
            print("基线 {} 不存在（llmqa report baseline-set <run_id> --name {}）".format(
                args.baseline_name, args.baseline_name))
            return 2
        run_a = json.loads(path.read_text(encoding="utf-8"))["run_id"]
        if args.run_b:
            run_b = args.run_b
        else:
            runs = sorted((d for d in report_dir.glob("*/report.json")
                           if d.parent.name not in ("compare", "baselines", run_a)),
                          key=lambda p: p.stat().st_mtime)
            if not runs:
                print("除基线外没有其他运行可对比")
                return 2
            run_b = runs[-1].parent.name
    elif args.run_a and args.run_b:
        run_a, run_b = args.run_a, args.run_b
    else:
        print("用法: llmqa report compare <基线ID> <当前ID> | --last | --baseline [当前ID]")
        return 2

    def load(run_id: str) -> dict:
        path = report_dir / run_id / "report.json"
        if not path.exists():
            print("报告不存在: " + str(path))
            raise SystemExit(2)
        return json.loads(path.read_text(encoding="utf-8"))

    data_a, data_b = load(run_a), load(run_b)
    # 溯源对齐提示：commit 不同意味着代码基线已变，回归需要结合 commit 判断
    prov_a, prov_b = data_a.get("provenance"), data_b.get("provenance")
    if prov_a and prov_b and prov_a.get("git_commit") != prov_b.get("git_commit"):
        print("代码基线: A={} B={}".format(
            (prov_a.get("git_commit") or "?")[:8], (prov_b.get("git_commit") or "?")[:8]))
    outcomes_a = {o["case_id"]: TestOutcome.model_validate(o) for o in data_a["outcomes"]}
    outcomes_b = {o["case_id"]: TestOutcome.model_validate(o) for o in data_b["outcomes"]}
    common = sorted(set(outcomes_a) & set(outcomes_b))
    only_a = sorted(set(outcomes_a) - set(outcomes_b))
    only_b = sorted(set(outcomes_b) - set(outcomes_a))

    # ---- Baseline compatibility："能不能比"先于"比的结果是什么" ----
    from llmqa.core.compare import env_diff, split_by_identity
    id_a = (prov_a or {}).get("test_identity", {})
    id_b = (prov_b or {}).get("test_identity", {})
    comparable, mismatched = split_by_identity(common, id_a, id_b)
    for line in env_diff(prov_a, prov_b):
        print("环境差异: " + line)
    if not comparable:
        print("基线不可对比：{} 例身份失配（用例代码/数据已变）、{} 例新增、{} 例移除".format(
            len(mismatched), len(only_b), len(only_a)))
        print("请人工复核后重新登记基线（llmqa report baseline-set ...）")
        return 2

    # 指标判定策略（方向 + 容差），见 config/metrics_policy.yaml
    from llmqa.core.metrics_policy import load_policies
    policies = load_policies(root)
    diffs = [compare_outcomes(outcomes_a[cid], outcomes_b[cid], policies=policies)
             for cid in comparable]
    counts = summarize_diffs(diffs)
    counts["identity_mismatches"] = len(mismatched)

    # 输出对比报告（markdown + json）
    out_dir = report_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"compare-{stamp}-{run_a}-vs-{run_b}"
    md_path = out_dir / (stem + ".md")
    json_path = out_dir / (stem + ".json")
    md_path.write_text(_render_compare_md(run_a, run_b, diffs, counts,
                                          only_a, only_b, mismatched),
                       encoding="utf-8")
    json_path.write_text(json.dumps({
        "run_a": run_a, "run_b": run_b, "counts": counts,
        "only_in_a": only_a, "only_in_b": only_b,
        "identity_mismatches": mismatched,
        "diffs": [d.model_dump(mode="json") for d in diffs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("对比 {a}（基线） vs {b}（当前） | 共 {t} 例 | 回归 {r} | 改善 {i} | "
          "指标漂移 {m} | 消息变化 {c} | 中性 {n} | 身份失配 {x} | 未变化 {u}".format(
              a=run_a, b=run_b, t=counts["total"], r=counts["regressions"],
              x=counts.get("identity_mismatches", 0),
              i=counts["improvements"], m=counts["metric_drifts"],
              c=counts["message_changes"], n=counts["neutrals"],
              u=counts["unchanged"]))
    if only_a:
        print("覆盖变化-移除: " + ", ".join(only_a[:10]))
    if only_b:
        print("覆盖变化-新增: " + ", ".join(only_b[:10]))
    if mismatched:
        print("身份失配（不可对比）: " + ", ".join(mismatched[:10]))
    print("对比报告: " + str(md_path))
    return 1 if counts["regressions"] else 0


def _render_compare_md(run_a: str, run_b: str, diffs: list,
                       counts: dict, only_a: list, only_b: list,
                       mismatched: list | None = None) -> str:
    """渲染运行间对比 Markdown 报告。"""
    lines = [
        "# 运行对比报告",
        "",
        f"- 基线（旧）: {run_a}  当前（新）: {run_b}",
        "- 结论: 共 {total} 例 | 回归 {regressions} | 改善 {improvements} | "
        "指标漂移 {metric_drifts} | 消息变化 {message_changes} | 中性 {neutrals} | "
        "未变化 {unchanged}".format(**counts),
        "",
    ]

    def section(title: str, items: list) -> None:
        lines.append(f"## {title}（{len(items)} 例）")
        lines.append("")
        if not items:
            lines.append("无")
        for d in items:
            lines.append(f"- **{d.case_id}** {d.name}（{d.severity.value}）: {d.verdict_a.value} → {d.verdict_b.value}")
            if d.message_a or d.message_b:
                lines.append(f"  - A: {d.message_a[:150]}")
                lines.append(f"  - B: {d.message_b[:150]}")
            for metric, pair in d.metric_diffs.items():
                sign = d.metric_signs.get(metric, "drift")
                lines.append("  - {}: {} → {}（{}）".format(
                    metric, pair["a"], pair["b"], sign))
        lines.append("")

    section("回归（判定劣化或指标按策略显著劣化）",
            [d for d in diffs if d.direction == "regression"])
    section("改善", [d for d in diffs if d.direction == "improvement"])
    section("指标漂移（仅记录，未声明策略或中性方向）",
            [d for d in diffs if d.direction == "metric_drift"])
    section("失败消息变化", [d for d in diffs if d.direction == "message_change"])
    section("中性（budget/fail_fast 跳过，不计回归）",
            [d for d in diffs if d.direction == "neutral"])
    if mismatched:
        lines.append("## 身份失配（同 id 但用例代码/数据已变，不可直接对比）")
        lines.append("")
        lines.append(", ".join(mismatched))
        lines.append("")
    if only_a:
        lines.append("## 覆盖变化-移除（仅基线存在）")
        lines.append("")
        lines.append(", ".join(only_a))
        lines.append("")
    if only_b:
        lines.append("## 覆盖变化-新增（仅当前存在）")
        lines.append("")
        lines.append(", ".join(only_b))
        lines.append("")
    return "\n".join(lines) + "\n"


def _datasets(args: argparse.Namespace) -> int:
    """执行 datasets 子命令：逐行打印全部数据集名称。"""
    from llmqa.datasets import DatasetManager
    for name in DatasetManager(_resolve_root(args.config) / "datasets").list():
        print(name)
    return 0


def _demo(args: argparse.Namespace) -> int:
    """执行 demo 子命令：导入演示模块触发 @test 注册，再运行演示套件。"""
    from llmqa import demo
    return demo.run()



def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并按子命令分派，返回进程退出码。"""
    args = _build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "prompts":
        return _prompts(args)
    if args.command == "datasets":
        return _datasets(args)
    if args.command == "report":
        return _report(args)
    if args.command == "demo":
        return _demo(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
