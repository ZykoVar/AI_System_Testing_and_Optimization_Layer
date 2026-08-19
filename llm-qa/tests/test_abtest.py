"""Prompt A/B 测试引擎自测。"""
import sys
from pathlib import Path

import pytest

# 独立运行时确保能导入未安装的 src/llmqa（与 conftest 等价）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmqa.clients import ClientPool
from llmqa.config import Settings
from llmqa.datasets import DatasetManager
from llmqa.prompts import run_abtest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def resources():
    """模块级复用：加载一次配置、连接池与数据集，避免重复 IO。"""
    settings = Settings.load(ROOT / "config")
    pool = ClientPool(settings)
    datasets = DatasetManager(ROOT / "datasets")
    return settings, pool, datasets


def test_same_versions_rejected(resources):
    settings, pool, datasets = resources
    # 两个版本号相同无对比意义，引擎应显式拒绝
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
    # demo-006 对版本敏感：v3 语义使其从 PASS 翻转为 FAIL，正是回归检测目标
    reg = result.by_direction("regression")[0]
    assert reg.case_id == "demo-006"
    assert reg.verdict_a.value == "PASS"
    assert reg.verdict_b.value == "FAIL"
    # 反向（v3 → v2）应记为改善
    reverse = run_abtest(ROOT, settings, pool, datasets, "support-agent", 3, 2,
                         tags={"demo"}, include_demo=True, progress=False)
    assert reverse.improvements == 1
    assert reverse.regressions == 0
