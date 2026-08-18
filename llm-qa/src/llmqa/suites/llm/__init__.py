"""LLM 功能与质量测试套件：格式、指令遵循、事实准确性、幻觉、一致性等。

每个子模块文件顶部有中文 docstring 说明关注点；用例通过 @test 装饰器注册，
本包导入子模块即触发注册（discover 通过 DEFAULT_PACKAGES 导入本包）。
"""
from llmqa.suites.llm import (  # noqa: F401
    consistency,
    factual_accuracy,
    format_compliance,
    hallucination_refusal,
    instruction_following,
    length_constraints,
    multilingual,
    safety_tone,
)

__all__ = [
    "format_compliance",
    "instruction_following",
    "factual_accuracy",
    "hallucination_refusal",
    "consistency",
    "multilingual",
    "safety_tone",
    "length_constraints",
]
