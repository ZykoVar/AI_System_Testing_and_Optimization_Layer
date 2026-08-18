"""Agent 会话状态与记忆专项：验证 session_history 传递与指代消解。

关注点：
- 多轮对话中，前文信息应通过 session_history 正确传递并被模型引用；
- 指代（"它"）应被正确消解为前文中的实体。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import AgentHarness


@test(id="agt-st-001", suite="agent", name="会话历史引用前文信息",
      description="Agent 应能引用 session_history 中的前文信息作答",
      tags=("memory",), severity=Severity.MEDIUM, timeout=60)
async def session_history_referenced(ctx: TestContext) -> None:
    """断言：最终答案引用历史对话中的关键信息（颜色偏好）。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match=r"蓝色", reply="根据我们的历史对话，你最喜欢的颜色是蓝色。",
                 match_transcript=True),
    ])
    harness = AgentHarness(client, [], system_prompt="你是助手，能记住对话历史。",
                           max_iterations=4)
    history = [Message.user("我最喜欢的颜色是蓝色。"),
               Message.assistant("好的，我记住了。")]
    trace = await harness.run("我之前说过我最喜欢什么颜色？", session_history=history)
    assert trace.success, "任务未成功完成"
    assert_contains(trace.final_answer, "蓝色")


@test(id="agt-st-002", suite="agent", name="指代消解到前文实体",
      description="模糊指代（它）应被消解为前文中的实体对象",
      tags=("memory",), severity=Severity.MEDIUM, timeout=60)
async def pronoun_resolution(ctx: TestContext) -> None:
    """断言：最终答案包含被指代对象名（iPhone 17）。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match=r"iPhone 17", reply="iPhone 17 性能强劲，续航优秀。",
                 match_transcript=True),
    ])
    harness = AgentHarness(client, [], system_prompt="你是助手，能结合上下文理解指代。",
                           max_iterations=4)
    history = [Message.user("最近发布的 iPhone 17 怎么样？"),
               Message.assistant("iPhone 17 是一款新手机。")]
    trace = await harness.run("那它的续航怎么样？", session_history=history)
    assert trace.success, "任务未成功完成"
    assert_contains(trace.final_answer, "iPhone")
