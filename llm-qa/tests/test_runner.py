"""测试运行器与注册表自测。"""
import asyncio

from llmqa.assertors import AssertionFailed
from llmqa.clients import ClientPool
from llmqa.config import Settings
from llmqa.core import (
    Severity,
    SkipTest,
    TestContext,
    TestRunner,
    Verdict,
    clear_registry,
    get_registered_cases,
)
from llmqa.core import (
    test as register_test,  # 别名避免 pytest 把装饰器当测试收集
)
from llmqa.datasets import DatasetManager
from llmqa.prompts import PromptManager


def make_ctx():
    """构造一个指向不存在资源的最小 TestContext，只用于单元用例不触达 IO。"""
    settings = Settings(default_provider="mock", providers={})
    return TestContext(
        run_id="t", settings=settings,
        providers=ClientPool(settings),
        prompts=PromptManager("__nonexistent__"),
        datasets=DatasetManager("__nonexistent__"),
    )


def test_registry_and_runner_verdicts():
    clear_registry()

    @register_test(id="unit-001", suite="unit", name="通过用例", severity=Severity.LOW)
    async def case_pass(ctx):
        pass

    @register_test(id="unit-002", suite="unit", name="失败用例")
    async def case_fail(ctx):
        raise AssertionFailed("预期失败")

    @register_test(id="unit-003", suite="unit", name="错误用例", retries=0)
    async def case_error(ctx):
        raise RuntimeError("boom")

    @register_test(id="unit-004", suite="unit", name="跳过用例")
    async def case_skip(ctx):
        raise SkipTest("环境不满足")

    cases = get_registered_cases()
    assert len(cases) == 4
    runner = TestRunner(make_ctx, concurrency=4, retries_on_error=0)
    report = asyncio.run(runner.run_all(cases))
    # 断言四类用例分别映射到 PASS/FAIL/ERROR/SKIP 四种裁决
    verdicts = {o.case_id: o.verdict for o in report.outcomes}
    assert verdicts["unit-001"] == Verdict.PASS
    assert verdicts["unit-002"] == Verdict.FAIL
    assert verdicts["unit-003"] == Verdict.ERROR
    assert verdicts["unit-004"] == Verdict.SKIP
    assert report.pass_rate == 0.25
    clear_registry()


def test_plain_assert_is_fail_not_error():
    clear_registry()

    @register_test(id="unit-005", suite="unit", name="普通断言")
    async def case_plain(ctx):
        assert 1 == 2, "普通断言失败"

    # retries_on_error=3 不会改变断言失败性质：普通 AssertionError 判 FAIL 而非 ERROR
    runner = TestRunner(make_ctx, retries_on_error=3)
    report = asyncio.run(runner.run_all(get_registered_cases()))
    outcome = report.outcomes[0]
    assert outcome.verdict == Verdict.FAIL
    assert "普通断言失败" in outcome.message
    clear_registry()


def test_evidence_propagates_to_outcome():
    # 端到端：AssertionFailed 携带的 evidence 必须透传到 TestOutcome（此前曾在此丢失）
    clear_registry()

    @register_test(id="unit-006c", suite="unit", name="证据用例", retries=0)
    async def case_with_evidence(ctx):
        raise AssertionFailed("低分", metrics={"judge_score": 3.0},
                              evidence=["裁判理由: 答案错误", "各次投票分: [3.0]"])

    report = asyncio.run(TestRunner(make_ctx, retries_on_error=0).run_all(
        get_registered_cases()))
    outcome = report.outcomes[0]
    assert outcome.verdict == Verdict.FAIL
    assert outcome.evidence == ["裁判理由: 答案错误", "各次投票分: [3.0]"]
    assert outcome.metrics["judge_score"] == 3.0
    clear_registry()


def test_retries_used_recorded():
    # 基础设施故障（5xx）重试应写入 outcome.retries_used，flaky 可见
    clear_registry()
    attempts = []
    from llmqa.clients import LLMError

    @register_test(id="unit-006b", suite="unit", name="重试用例", retries=2)
    async def case_flaky(ctx):
        attempts.append(1)
        raise LLMError("mock", "服务器抖动", status=503)

    runner = TestRunner(make_ctx, retries_on_error=2)
    report = asyncio.run(runner.run_all(get_registered_cases()))
    outcome = report.outcomes[0]
    assert outcome.verdict == Verdict.ERROR
    assert outcome.retries_used == 2
    assert len(attempts) == 3          # 首次 + 两次重试
    clear_registry()


def test_code_bug_is_not_retried():
    # 代码缺陷（TypeError）不在重试分桶内：立即 ERROR 且 retries_used=0
    clear_registry()
    attempts = []

    @register_test(id="unit-006d", suite="unit", name="缺陷用例", retries=3)
    async def case_bug(ctx):
        attempts.append(1)
        raise TypeError("int + str")

    runner = TestRunner(make_ctx, retries_on_error=3)
    report = asyncio.run(runner.run_all(get_registered_cases()))
    outcome = report.outcomes[0]
    assert outcome.verdict == Verdict.ERROR
    assert outcome.retries_used == 0
    assert len(attempts) == 1          # 不重试
    assert "代码缺陷" in outcome.message
    clear_registry()


def test_timeout_is_error():
    clear_registry()

    # timeout=0.1 而用例睡眠 1 秒，必然触发超时；retries=0 避免重试拖慢测试
    @register_test(id="unit-006", suite="unit", name="超时用例", timeout=0.1, retries=0)
    async def case_slow(ctx):
        await asyncio.sleep(1)

    runner = TestRunner(make_ctx, retries_on_error=0)
    report = asyncio.run(runner.run_all(get_registered_cases()))
    assert report.outcomes[0].verdict == Verdict.ERROR
    assert "超时" in report.outcomes[0].message
    clear_registry()


def test_filtering():
    clear_registry()

    @register_test(id="unit-007", suite="unit-a", tags=("smoke", "fast"))
    async def case_a(ctx):
        pass

    @register_test(id="unit-008", suite="unit-b", tags=("slow",))
    async def case_b(ctx):
        pass

    defs = {d.id: d for d in get_registered_cases()}
    # None 表示该维度不过滤，只有显式给出的集合才参与匹配
    a, b = defs["unit-007"], defs["unit-008"]
    assert a.matches(suites={"unit-a"}, tags=None, exclude_tags=None, min_severity=None)
    assert not a.matches(suites={"unit-b"}, tags=None, exclude_tags=None, min_severity=None)
    assert a.matches(suites=None, tags={"smoke"}, exclude_tags=None, min_severity=None)
    assert not b.matches(suites=None, tags={"smoke"}, exclude_tags=None, min_severity=None)
    clear_registry()


def test_cost_budget_skips_over_budget_cases():
    clear_registry()

    @register_test(id="unit-010", suite="unit", name="预算内", cost=1)
    async def case_cheap(ctx):
        pass

    @register_test(id="unit-011", suite="unit", name="预算外", cost=5)
    async def case_expensive(ctx):
        pass

    # 预算 3：第一个用例（cost=1）可跑，第二个（cost=5）超支 → SKIP
    runner = TestRunner(make_ctx, max_cost=3, retries_on_error=0)
    report = asyncio.run(runner.run_all(get_registered_cases()))
    verdicts = {o.case_id: o.verdict for o in report.outcomes}
    assert verdicts["unit-010"] == Verdict.PASS
    assert verdicts["unit-011"] == Verdict.SKIP
    assert "预算耗尽" in next(o.message for o in report.outcomes if o.case_id == "unit-011")
    assert report.max_cost == 3
    assert report.used_cost == 1
    clear_registry()
