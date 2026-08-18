"""性能测试 - token 效率：输出 token 上限、回复简洁性与输入 token 估算。

关注点：
- 输出 completion_tokens 是否 ≤ max_tokens 参数；
- 短问题回复是否过度冗长（≤ 200 词）；
- 长提示的输入 token 是否被正确统计（prompt_tokens > 0）。

mock 下回复短小、token 由 estimate_tokens 估算（约 4 字符/token），校验必过；
真实 Provider 下依赖服务端 usage 统计。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed, assert_word_count
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(id="perf-tok-001", suite="performance", name="输出 token 不超上限",
      description="输出的 completion_tokens 不超过 max_tokens 参数",
      tags=("性能", "token效率"), severity=Severity.MEDIUM, timeout=60)
async def case_completion_within_limit(ctx: TestContext) -> None:
    """断言 completion_tokens ≤ max_tokens（此处 max_tokens=64）。"""
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="简短回复。")])
    max_tokens = 64
    resp = await client.generate([Message.user("请简短回答")], max_tokens=max_tokens)
    comp = resp.usage.completion_tokens
    if comp > max_tokens:
        raise AssertionFailed("输出 {:,} token 超过 max_tokens={}".format(comp, max_tokens),
                              metrics={"completion_tokens": comp, "max_tokens": max_tokens})


@test(id="perf-tok-002", suite="performance", name="短问题回复不过度冗长",
      description="短问题回复不超过 200 词",
      tags=("性能", "token效率"), severity=Severity.MEDIUM, timeout=60)
async def case_brief_reply(ctx: TestContext) -> None:
    """断言短问题回复词数 ≤ 200（mock 分支使用短回复验证测试逻辑）。"""
    payload = ctx.datasets.load("performance/payloads")["short"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="这是针对短问题的简短回复。")])
    resp = await client.generate([Message.user(payload)], max_tokens=200)
    assert_word_count(resp.text, max_words=200, message="短问题回复超过 200 词")


@test(id="perf-tok-003", suite="performance", name="长提示输入 token 统计合理",
      description="长提示的输入 token（prompt_tokens）被正确统计且大于 0",
      tags=("性能", "token效率"), severity=Severity.LOW, timeout=60)
async def case_prompt_tokens(ctx: TestContext) -> None:
    """断言长载荷请求的 prompt_tokens > 0。"""
    payload = ctx.datasets.load("performance/payloads")["long"]
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="长输入回复。")])
    resp = await client.generate([Message.user(payload)])
    pt = resp.usage.prompt_tokens
    if pt <= 0:
        raise AssertionFailed("长提示输入 token 未被正确统计（prompt_tokens={}）".format(pt),
                              metrics={"prompt_tokens": pt, "payload_chars": len(payload)})
