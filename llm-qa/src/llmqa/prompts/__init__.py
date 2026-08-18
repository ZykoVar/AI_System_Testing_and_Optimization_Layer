"""Prompt 管理子系统：版本化注册表、渲染校验、diff 与注入扫描。"""
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

__all__ = [
    "PromptManager", "PromptTemplate", "PromptMessage", "VariableSpec",
    "PromptError", "PromptNotFound", "PromptRenderError",
    "PromptScanner", "ScanReport", "InjectionFinding",
]
