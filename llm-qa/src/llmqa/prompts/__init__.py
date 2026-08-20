"""Prompt 管理子系统：版本化注册表、渲染校验、diff、注入扫描与 A/B 测试。"""
from llmqa.prompts.abtest import ABChange, ABTestResult, render_ab_report, run_abtest
from llmqa.prompts.guards import InjectionFinding, PromptScanner, ScanReport
from llmqa.prompts.manager import (
    PromptError,
    PromptManager,
    PromptMessage,
    PromptNotFound,
    PromptRenderError,
    PromptTemplate,
    VariableSpec,
)

# 只暴露稳定公共 API；内部实现细节（如 _compare、_RULES）不在此重导出。
__all__ = [
    "ABChange",
    "ABTestResult",
    "InjectionFinding",
    "PromptError",
    "PromptManager",
    "PromptMessage",
    "PromptNotFound",
    "PromptRenderError",
    "PromptScanner",
    "PromptTemplate",
    "ScanReport",
    "VariableSpec",
    "render_ab_report",
    "run_abtest",
]
