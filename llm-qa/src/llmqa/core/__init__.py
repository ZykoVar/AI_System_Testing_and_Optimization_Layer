"""测试执行引擎：用例模型、注册表、运行器、报告器、指标与负载工具。"""
# 本模块是 llmqa.core 的公共 API 面：所有对外符号在此汇总再导出，外部只应从此处 import。
from llmqa.core.load import LoadStats, run_load, run_ramp
from llmqa.core.metrics import latency_stats, percentile
from llmqa.core.models import Severity, TestContext, TestOutcome, Verdict
from llmqa.core.registry import (
    SkipTest,
    TestCaseDef,
    clear_registry,
    data_driven,
    discover,
    get_registered_cases,
    test,
)
from llmqa.core.reporter import Reporter
from llmqa.core.runner import TestReport, TestRunner

__all__ = [
    "LoadStats",
    "Reporter",
    "Severity",
    "SkipTest",
    "TestCaseDef",
    "TestContext",
    "TestOutcome",
    "TestReport",
    "TestRunner",
    "Verdict",
    "clear_registry",
    "data_driven",
    "discover",
    "get_registered_cases",
    "latency_stats",
    "percentile",
    "run_load",
    "run_ramp",
    "test",
]
