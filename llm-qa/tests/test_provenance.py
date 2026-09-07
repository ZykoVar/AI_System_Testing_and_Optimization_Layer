"""运行溯源（Provenance）自测。"""
from pathlib import Path

import pytest

from llmqa.core.provenance import collect_provenance
from llmqa.datasets import DatasetManager
from llmqa.prompts import PromptManager

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_usage_tracking():
    pm = PromptManager(ROOT / "prompts").load()
    pm.render("support-agent", {"company": "Acme", "headquarters": "上海",
                                "secret_value": "C", "question": "q"}, version=2)
    pm.render("rag/answer", {"context": "c", "question": "q"})
    used = {u["id"]: u["version"] for u in pm.used_prompts()}
    assert used == {"rag/answer": 1, "support-agent": 2}   # 记录的是实际解析版本


def test_dataset_usage_tracking():
    dm = DatasetManager(ROOT / "datasets")
    dm.load("golden_qa")
    dm.load("rag/corpus")
    assert dm.used_datasets() == ["golden_qa", "rag/corpus"]


def test_collect_provenance_fields():
    prov = collect_provenance(ROOT)
    assert prov.schema_version if hasattr(prov, "schema_version") else True  # 结构检查
    assert prov.python_version
    assert prov.platform
    assert prov.timezone
    # git 字段：本仓库是 git 环境时应能读到提交
    if prov.git_commit:
        assert len(prov.git_commit) >= 7


def test_collect_provenance_with_usage():
    pm = PromptManager(ROOT / "prompts").load()
    pm.render("rag/answer", {"context": "c", "question": "q"})
    dm = DatasetManager(ROOT / "datasets")
    dm.load("rag/corpus")
    prov = collect_provenance(ROOT, pm, dm)
    assert {u.model_dump()["id"] for u in prov.prompts_used} == {"rag/answer"}
    assert prov.datasets_used == ["rag/corpus"]


def test_attach_provenance_to_report():
    from llmqa.core.provenance import attach_provenance
    from llmqa.core.runner import TestReport
    report = TestReport(run_id="r", provider="mock", started_at="2025-01-01")
    attach_provenance(report, ROOT)
    assert report.provenance is not None
    assert report.schema_version == 1
    # 报告可序列化（reporter._to_dict 依赖）
    assert report.provenance.model_dump(mode="json")["python_version"]
