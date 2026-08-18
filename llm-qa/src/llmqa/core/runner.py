"""测试运行器：并发调度、超时、重试、fail-fast 与报告汇总。"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
import traceback
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from llmqa.assertors.base import AssertionFailed
from llmqa.core.models import Severity, TestContext, TestOutcome, Verdict
from llmqa.core.registry import SkipTest, TestCaseDef


class TestReport(BaseModel):
    """一次运行的完整报告。"""
    run_id: str
    provider: str
    started_at: str
    duration_ms: float = 0.0
    outcomes: list[TestOutcome] = Field(default_factory=list)

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
            return 1.0
        return self.counts["PASS"] / total

    @property
    def failures(self) -> list[TestOutcome]:
        return [o for o in self.outcomes if o.verdict in (Verdict.FAIL, Verdict.ERROR)]

    def failures_at_or_above(self, severity: Severity) -> list[TestOutcome]:
        return [o for o in self.failures if o.severity >= severity]

    def summary_text(self) -> str:
        c = self.counts
        lines = [
            f"运行 {len(self.outcomes)} 个用例 | 通过 {c['PASS']} | 失败 {c['FAIL']} | "
            f"错误 {c['ERROR']} | 跳过 {c['SKIP']} | 通过率 {self.pass_rate:.1%}",
        ]
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
    ):
        self.ctx_factory = ctx_factory
        self.concurrency = max(1, concurrency)
        self.fail_fast = fail_fast
        self.retries_on_error = retries_on_error
        self.default_timeout = default_timeout
        self.progress = progress or (lambda o: None)

    async def _run_one(self, case: TestCaseDef) -> TestOutcome:
        timeout = case.timeout or self.default_timeout
        retries = case.retries if case.retries is not None else self.retries_on_error
        start = time.perf_counter()
        verdict, message, tb = Verdict.ERROR, "", None
        metrics: dict = {}
        if case.skip:
            verdict, message = Verdict.SKIP, "用例标记为 skip"
        else:
            attempt = 0
            while True:
                try:
                    ctx = self.ctx_factory()
                    await asyncio.wait_for(case.fn(ctx), timeout=timeout)
                    verdict, message = Verdict.PASS, "通过"
                    break
                except SkipTest as e:
                    verdict, message = Verdict.SKIP, str(e)
                    break
                except AssertionFailed as e:
                    verdict, message = Verdict.FAIL, str(e)
                    metrics = dict(getattr(e, "metrics", {}) or {})
                    break
                except AssertionError as e:  # 普通 assert 失败 → FAIL（不重试）
                    verdict, message = Verdict.FAIL, "断言失败: {}".format(e)
                    break
                except asyncio.TimeoutError:
                    verdict, message = Verdict.ERROR, f"超时（>{timeout:g}s）"
                    break
                except Exception as e:  # noqa: BLE001 —— 基础设施/用例代码错误
                    if attempt < retries:
                        attempt += 1
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    verdict, message = Verdict.ERROR, f"{type(e).__name__}: {e}"
                    tb = traceback.format_exc(limit=8)
                    break
        outcome = TestOutcome(
            case_id=case.id, name=case.name, suite=case.suite,
            description=case.description, tags=sorted(case.tags),
            severity=case.severity, verdict=verdict,
            duration_ms=(time.perf_counter() - start) * 1000,
            message=message, metrics=metrics, traceback=tb,
        )
        self.progress(outcome)
        return outcome

    async def run_all(self, cases: list[TestCaseDef], provider_name: str = "default") -> TestReport:
        sem = asyncio.Semaphore(self.concurrency)
        stop_event = asyncio.Event()
        outcomes: list[TestOutcome] = []

        async def worker(case: TestCaseDef) -> None:
            async with sem:
                if stop_event.is_set():
                    outcomes.append(TestOutcome(
                        case_id=case.id, name=case.name, suite=case.suite,
                        tags=sorted(case.tags), severity=case.severity,
                        verdict=Verdict.SKIP, message="fail-fast：前置高危失败，未执行",
                    ))
                    return
                outcome = await self._run_one(case)
                outcomes.append(outcome)
                if (self.fail_fast and outcome.verdict in (Verdict.FAIL, Verdict.ERROR)
                        and outcome.severity >= Severity.HIGH):
                    stop_event.set()

        start = time.perf_counter()
        await asyncio.gather(*(worker(c) for c in cases))
        return TestReport(
            run_id=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
            provider=provider_name,
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
            duration_ms=(time.perf_counter() - start) * 1000,
            outcomes=sorted(outcomes, key=lambda o: (o.suite, o.case_id)),
        )

    def run_sync(self, cases: list[TestCaseDef], provider_name: str = "default") -> TestReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_all(cases, provider_name=provider_name))
        raise RuntimeError("TestRunner.run_sync 不能在已有事件循环中调用，请使用 run_all")
