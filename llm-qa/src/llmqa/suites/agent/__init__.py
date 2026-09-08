"""Agent 测试套件：工具选择、参数 Schema、多步规划、会话状态、循环与预算护栏。

子模块：
- tool_selection:        工具选择正确性（天气/算术/闲聊/多工具）
- tool_argument_schema:  工具参数 Schema 与异常恢复
- multi_step_planning:   多步规划与工具链
- state_memory:          会话状态与指代消解
- loop_detection:        循环检测护栏
- budget_guardrails:     迭代/token 预算护栏
- tool_refusal:          工具调用策略（未注册/白名单/allowlist_only）
"""
# 显式导入各子模块以触发 @test 装饰器注册：discover 只 import 包本身，需在此级联导入
from . import (
    behavior_dsl,
    budget_guardrails,
    loop_detection,
    multi_step_planning,
    state_memory,
    tool_argument_schema,
    tool_refusal,
    tool_selection,
)

__all__ = [
    "behavior_dsl",
    "budget_guardrails",
    "loop_detection",
    "multi_step_planning",
    "state_memory",
    "tool_argument_schema",
    "tool_refusal",
    "tool_selection",
]
