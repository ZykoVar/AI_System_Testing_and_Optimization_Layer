"""运行溯源（Provenance）自测：内容指纹、模型与用例身份。"""
from pathlib import Path

from llmqa.core.provenance import collect_provenance
from llmqa.datasets import DatasetManager
from llmqa.prompts import PromptManager

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_usage_tracking_with_hash():
    pm = PromptManager(ROOT / "prompts").load()
    pm.render("support-agent", {"company": "Acme", "headquarters": "上海",
                                "secret_value": "C", "question": "q"}, version=2)
    pm.render("rag/answer", {"context": "c", "question": "q"})
    used = {u["id"]: u for u in pm.used_prompts()}
    assert set(used) == {"rag/answer", "support-agent"}
    assert used["support-agent"]["version"] == 2        # 记录的是实际解析版本
    assert len(used["support-agent"]["content_hash"]) == 16   # 内容指纹防篡改


def test_dataset_usage_tracking_with_hash():
    dm = DatasetManager(ROOT / "datasets")
    dm.load("golden_qa")
    dm.load("rag/corpus")
    used = {u["name"]: u for u in dm.used_datasets()}
    assert set(used) == {"golden_qa", "rag/corpus"}
    assert len(used["rag/corpus"]["content_hash"]) == 16


def test_collect_provenance_fields():
    prov = collect_provenance(ROOT)
    assert prov.python_version
    assert prov.platform
    assert prov.timezone
    # git 字段：本仓库是 git 环境时应能读到提交
    if prov.git_commit:
        assert len(prov.git_commit) >= 7


def test_collect_provenance_with_usage_and_model():
    from llmqa.config import Settings
    pm = PromptManager(ROOT / "prompts").load()
    pm.render("rag/answer", {"context": "c", "question": "q"})
    dm = DatasetManager(ROOT / "datasets")
    dm.load("rag/corpus")
    settings = Settings.load(ROOT / "config")
    prov = collect_provenance(ROOT, pm, dm, settings=settings)
    assert {u.model_dump()["id"] for u in prov.prompts_used} == {"rag/answer"}
    assert [u.name for u in prov.datasets_used] == ["rag/corpus"]
    assert prov.model is not None
    assert prov.model.provider == "mock"
    assert prov.model.model == "mock-1"


def test_collect_provenance_test_identity():
    # 用例源码指纹：同一用例 id 的内容变化在溯源中可见
    from llmqa.core.registry import TestCaseDef
    cases = [TestCaseDef(fn=lambda: None, id="sec-inj-001", suite="security",
                         name="x", description="", tags=frozenset(),
                         severity="MEDIUM", timeout=None, retries=None, skip=False,
                         source_hash="abc123")]
    prov = collect_provenance(ROOT, cases=cases)
    assert prov.test_identity == {"sec-inj-001": "abc123"}


def test_attach_provenance_to_report():
    from llmqa.core.provenance import attach_provenance
    from llmqa.core.runner import TestReport
    report = TestReport(run_id="r", provider="mock", started_at="2025-01-01")
    attach_provenance(report, ROOT)
    assert report.provenance is not None
    assert report.schema_version == 2   # v2：provenance 含 content_hash/模型/用例指纹
    # 报告可序列化（reporter._to_dict 依赖）
    assert report.provenance.model_dump(mode="json")["python_version"]
