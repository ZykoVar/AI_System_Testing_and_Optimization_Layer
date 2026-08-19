"""Prompt A/B 测试引擎自测。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmqa.clients import ClientPool
from llmqa.config import Settings
from llmqa.datasets import DatasetManager
from llmqa.prompts import run_abtest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def resources():
    settings = Settings.load(ROOT / "config")
    pool = ClientPool(settings)
    datasets = DatasetManager(ROOT / "datasets")
    return settings, pool, datasets


def test_same_versions_rejected(resources):
    settings, pool, datasets = resources
    with pytest.raises(ValueError):
        run_abtest(ROOT, settings, pool, datasets, "support-agent", 2, 2,
                   progress=False)


def test_abtest_detects_version_regression(resources):
    settings, pool, datasets = resources
    result = run_abtest(ROOT, settings, pool, datasets, "support-agent", 2, 3,
                        tags={"demo"}, include_demo=True, progress=False)
    assert result.total == 6
    assert result.regressions == 1
    assert result.improvements == 0
    reg = result.by_direction("regression")[0]
    assert reg.case_id == "demo-006"
    assert reg.verdict_a.value == "PASS"
    assert reg.verdict_b.value == "FAIL"
    # 反向（v3 → v2）应记为改善
    reverse = run_abtest(ROOT, settings, pool, datasets, "support-agent", 3, 2,
                         tags={"demo"}, include_demo=True, progress=False)
    assert reverse.improvements == 1
    assert reverse.regressions == 0
