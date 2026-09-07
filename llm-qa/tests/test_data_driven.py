"""数据驱动注册器（data_driven）自测。"""
from llmqa.core import Severity, clear_registry, data_driven, get_registered_cases

DATASET = {
    "items": [
        {"id": "r-1", "case_name": "记录一", "severity": "HIGH", "tags": ["a", "smoke"],
         "mock_match": "特征一", "payload": "p1"},
        {"id": "r-2", "case_name": "记录二", "payload": "p2"},   # 用装饰器默认元数据
    ],
}


def test_generates_one_case_per_record():
    clear_registry()

    @data_driven(DATASET, suite="unit", id_prefix="dd", name_field="case_name")
    async def case_unit(ctx, item):
        assert item["payload"]

    defs = {d.id: d for d in get_registered_cases()}
    assert set(defs) == {"dd-001", "dd-002"}
    assert defs["dd-001"].name == "记录一"
    assert defs["dd-001"].severity == Severity.HIGH          # 记录级覆盖
    assert defs["dd-001"].tags == frozenset({"a", "smoke"})
    assert defs["dd-002"].severity == Severity.MEDIUM        # 回退到默认
    assert defs["dd-002"].tags == frozenset()
    clear_registry()


def test_missing_name_field_raises():
    clear_registry()
    import pytest
    with pytest.raises(ValueError):
        @data_driven({"items": [{"id": "x"}]}, suite="unit", id_prefix="dd")
        async def case_bad(ctx, item):
            pass
    clear_registry()


def test_generated_case_executes_with_item():
    import asyncio
    clear_registry()
    seen = []

    @data_driven(DATASET, suite="unit", id_prefix="dd", name_field="case_name")
    async def case_unit(ctx, item):
        seen.append(item["id"])

    from llmqa.core import TestRunner, Verdict
    from llmqa.clients import ClientPool
    from llmqa.config import Settings
    from llmqa.datasets import DatasetManager
    from llmqa.prompts import PromptManager

    settings = Settings(default_provider="mock", providers={})

    def make_ctx():
        return __import__("llmqa.core.models", fromlist=["TestContext"]).TestContext(
            run_id="t", settings=settings, providers=ClientPool(settings),
            prompts=PromptManager("__nonexistent__"),
            datasets=DatasetManager("__nonexistent__"))

    report = asyncio.run(TestRunner(make_ctx, retries_on_error=0).run_all(
        get_registered_cases()))
    assert all(o.verdict == Verdict.PASS for o in report.outcomes)
    assert sorted(seen) == ["r-1", "r-2"]
    clear_registry()
