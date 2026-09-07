"""测试运行器：并发调度、超时、重试、fail-fast 与报告汇总。"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
import traceback
from collections.abc import Callable

from pydantic import BaseModel, Field

from llmqa.assertors.base import AssertionFailed
from llmqa.core.provenance import RunProvenance
from llmqa.core.retry import error_kind, is_retryable_error
from llmqa.core.models import Severity, TestContext, TestOutcome, Verdict
from llmqa.core.registry import SkipTest, TestCaseDef


class TestReport(BaseModel):
    """一次运行的完整报告。"""
    run_id: str
    provider: str
    started_at: str
    duration_ms: float = 0.0
    outcomes: list[TestOutcome] = Field(default_factory=list)
    max_cost: int | None = None   # --max-cost 预算上限（None 表示不限）
    used_cost: int = 0            # 实际消耗的成本单位（按用例声明 cost 累计）
    schema_version: int = 2       # 报告结构版本：v2 起 provenance 含 content_hash/模型/用例指纹
    provenance: RunProvenance | None = None  # 运行溯源（git/Prompt/数据集版本），见 core.provenance

    @property
    def counts(self) -> dict[str, int]:
        result = {v.value: 0 for v in Verdict}
        for o in self.outcomes:
            result[o.verdict.value] += 1
        return result

    @property
    def pass_rate(self) -> float:
        total = len(self.outcomes)
        if total == 0:
            # 空报告无失败，按 100% 通过处理，避免除零。
            return 1.0
        return self.counts["PASS"] / total

    @property
    def failures(self) -> list[TestOutcome]:
        return [o for o in self.outcomes if o.verdict in (Verdict.FAIL, Verdict.ERROR)]

    def failures_at_or_above(self, severity: Severity) -> list[TestOutcome]:
        """返回不低于指定等级的失败（含 FAIL 与 ERROR），供 exit-code 判定使用。"""
        return [o for o in self.failures if o.severity >= severity]

    def summary_text(self) -> str:
        c = self.counts
        lines = [
            f"运行 {len(self.outcomes)} 个用例 | 通过 {c['PASS']} | 失败 {c['FAIL']} | "
            f"错误 {c['ERROR']} | 跳过 {c['SKIP']} | 通过率 {self.pass_rate:.1%}",
        ]
        if self.max_cost is not None:
            lines.append(f"成本预算: 已用 {self.used_cost} / 上限 {self.max_cost}")
        # 按严重度降序排列，只展示前 10 条，避免控制台被刷屏。
        fails = sorted(self.failures, key=lambda o: -o.severity.rank)
        for o in fails[:10]:
            lines.append(f"  [{o.verdict.value}] {o.severity.value:<8} {o.case_id} — {o.name}: {o.message[:120]}")
        if len(fails) > 10:
            lines.append(f"  ... 另有 {len(fails) - 10} 个失败，详见报告文件")
        return "\n".join(lines)


class TestRunner:
    """异步并发执行注册用例。

    ctx_factory: 每个用例独立上下文（保证用例间隔离；客户端连接由 ClientPool 缓存复用）。
    """

    def __init__(
        self,
        ctx_factory: Callable[[], TestContext],
        *,
        concurrency: int = 8,
        fail_fast: bool = False,
        retries_on_error: int = 1,
        default_timeout: float = 90.0,
        progress: Callable[[TestOutcome], None] | None = None,
        max_cost: int | None = None,
    ):
        self.ctx_factory = ctx_factory
        self.concurrency = max(1, concurrency)  # 并发至少为 1，避免信号量阻塞全部任务。
        self.fail_fast = fail_fast
        self.retries_on_error = retries_on_error
        self.default_timeout = default_timeout
        self.progress = progress or (lambda o: None)
        # 成本预算：按用例声明 cost 累计，超预算的用例记为 SKIP（真实模型夜间回归防烧钱）
        self.max_cost = max_cost
        self.used_cost = 0
        self._cost_lock = asyncio.Lock()

    async def _run_one(self, case: TestCaseDef) -> TestOutcome:
        """执行单个用例并归一化为 TestOutcome；任何异常都被捕获，绝不外抛。"""
        timeout = case.timeout or self.default_timeout
        retries = case.retries if case.retries is not None else self.retries_on_error
        start = time.perf_counter()
        verdict, message, tb = Verdict.ERROR, "", None
        metrics: dict = {}
        evidence: list[str] = []   # 断言附带的证据（裁判理由/引用），随结果落盘
        skip_reason = ""           # SKIP 语义分类（intentional/budget/fail_fast）
        if case.skip:
            verdict, message, skip_reason = Verdict.SKIP, "用例标记为 skip", "intentional"
        else:
            attempt = 0
            while True:
                try:
                    ctx = self.ctx_factory()
                    await asyncio.wait_for(case.fn(ctx), timeout=timeout)
                    verdict, message = Verdict.PASS, "通过"
                    break
                except SkipTest as e:
                    verdict, message, skip_reason = Verdict.SKIP, str(e), "intentional"
                    break
                except AssertionFailed as e:
                    verdict, message = Verdict.FAIL, str(e)
                    # 断言可附带数值指标与证据，两者都透传到结果（evidence 不得在此丢失）
                    metrics = dict(getattr(e, "metrics", {}) or {})
                    evidence = list(getattr(e, "evidence", []) or [])
                    break
                except AssertionError as e:  # 普通 assert 失败 → FAIL（不重试）
                    verdict, message = Verdict.FAIL, f"断言失败: {e}"
                    break
                except asyncio.TimeoutError:
                    # 超时属于基础设施问题，不重试（避免放大长时间挂起的影响）。
                    verdict, message = Verdict.ERROR, f"超时（>{timeout:g}s）"
                    break
                except Exception as e:  # noqa: BLE001 —— 基础设施/用例代码错误
                    # 重试分桶：只有基础设施故障（429/5xx/网络层）才重试；
                    # 代码缺陷（KeyError/TypeError 等）重试无意义，直接判 ERROR。
                    if attempt < retries and is_retryable_error(e):
                        attempt += 1
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    verdict, message = Verdict.ERROR, "{}: {}: {}".format(
                        error_kind(e), type(e).__name__, e)
                    tb = traceback.format_exc(limit=8)
                    break
        outcome = TestOutcome(
            case_id=case.id, name=case.name, suite=case.suite,
            description=case.description, tags=sorted(case.tags),
            severity=case.severity, verdict=verdict,
            duration_ms=(time.perf_counter() - start) * 1000,
            message=message, metrics=metrics, traceback=tb,
            evidence=evidence,
            retries_used=attempt,   # flaky 可见性：0=一次通过，N=重试 N 次后判定
            skip_reason=skip_reason,
        )
        self.progress(outcome)
        return outcome

    async def run_all(self, cases: list[TestCaseDef], provider_name: str = "default") -> TestReport:
        sem = asyncio.Semaphore(self.concurrency)  # 限制同时运行的用例数。
        stop_event = asyncio.Event()  # fail-fast 广播信号，触发后跳过余下用例。
        outcomes: list[TestOutcome] = []

        async def worker(case: TestCaseDef) -> None:
            async with sem:
                if stop_event.is_set():
                    # fail-fast 触发后，尚未执行的用例统一记为 SKIP，不实际调用被测对象。
                    outcomes.append(TestOutcome(
                        case_id=case.id, name=case.name, suite=case.suite,
                        tags=sorted(case.tags), severity=case.severity,
                        verdict=Verdict.SKIP, message="fail-fast：前置高危失败，未执行",
                        skip_reason="fail_fast",
                    ))
                    return
                if self.max_cost is not None:
                    # 预算扣减必须先于执行（原子操作），避免并发下超支。
                    async with self._cost_lock:
                        if self.used_cost + case.cost > self.max_cost:
                            outcomes.append(TestOutcome(
                                case_id=case.id, name=case.name, suite=case.suite,
                                tags=sorted(case.tags), severity=case.severity,
                                verdict=Verdict.SKIP,
                                message=f"成本预算耗尽：已用 {self.used_cost}/{self.max_cost}，本用例需 {case.cost}",
                                skip_reason="budget",
                            ))
                            return
                        self.used_cost += case.cost
                outcome = await self._run_one(case)
                outcomes.append(outcome)
                if (self.fail_fast and outcome.verdict in (Verdict.FAIL, Verdict.ERROR)
                        and outcome.severity >= Severity.HIGH):
                    stop_event.set()

        start = time.perf_counter()
        await asyncio.gather(*(worker(c) for c in cases))
        import uuid  # 局部导入仅为生成 run_id 的随机后缀，避免污染模块顶部导入区。
        return TestReport(
            run_id=dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4],
            provider=provider_name,
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
            duration_ms=(time.perf_counter() - start) * 1000,
            outcomes=sorted(outcomes, key=lambda o: (o.suite, o.case_id)),
            max_cost=self.max_cost,
            used_cost=self.used_cost,
        )

    def run_sync(self, cases: list[TestCaseDef], provider_name: str = "default") -> TestReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 无运行中事件循环：创建新循环同步执行（适合 CLI / 脚本入口）。
            return asyncio.run(self.run_all(cases, provider_name=provider_name))
        raise RuntimeError("TestRunner.run_sync 不能在已有事件循环中调用，请使用 run_all")
