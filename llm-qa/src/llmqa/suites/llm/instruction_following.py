"""指令遵循测试套件（llm-ins-）：验证模型对自然语言指令的遵循能力。

关注点：
- 回复词数上限（assert_word_count）；
- 必含关键词 / 否定约束（不得出现某词）；
- 角色口吻（客服身份，通过 prompts/support-agent 渲染系统提示）；
- 多约束同时满足；
- 指定输出顺序（首先/其次）。

断言均为确定性断言；mock 分支用脚本模拟"理想遵循"以验证测试逻辑，
真实 Provider 下由真实模型遵循指令。
"""
from __future__ import annotations

import re

from llmqa.assertors import (
    assert_contains,
    assert_matches,
    assert_not_contains,
    assert_word_count,
)
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(
    id="llm-ins-001",
    suite="llm",
    name="回复词数不超过上限",
    description="要求模型用不超过 N 个英文单词作答，验证词数上限约束。",
    tags=("instruction", "smoke"),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def word_count_limit(ctx: TestContext) -> None:
    """断言：回复词数（空白切分）不超过 15。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="15 个英文单词", reply="Seven days return policy with refund in three days."),
    ])
    resp = await client.generate(
        [Message.user("请用不超过 15 个英文单词回答：Acme 的退货政策是什么？")],
        temperature=0.0, max_tokens=512,
    )
    assert_word_count(resp.text, max_words=15)


@test(
    id="llm-ins-002",
    suite="llm",
    name="回复必含指定关键词",
    description="要求模型在回答中务必包含指定关键词，验证关键词命中。",
    tags=("instruction",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def must_contain(ctx: TestContext) -> None:
    """断言：回复包含关键词"保障"。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="保障", reply="Acme 为用户提供数据加密与隐私保障，信息仅限授权访问。"),
    ])
    resp = await client.generate(
        [Message.user("请介绍 Acme 的隐私保护措施，务必包含关键词'保障'。")],
        temperature=0.0, max_tokens=512,
    )
    assert_contains(resp.text, "保障")


@test(
    id="llm-ins-003",
    suite="llm",
    name="遵守否定约束不出现禁用词",
    description="要求模型回答时不得出现某类词，验证否定约束生效。",
    tags=("instruction",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def must_not_contain(ctx: TestContext) -> None:
    """断言：回复不含价格相关词（价格/299/元）。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="不要提及", reply="Acme 是一家成立于 2015 年的云计算与客服服务公司。"),
    ])
    resp = await client.generate(
        [Message.user("请介绍 Acme，但不要提及价格相关的任何数字。")],
        temperature=0.0, max_tokens=512,
    )
    assert_not_contains(resp.text, "价格", "299", "元")


@test(
    id="llm-ins-004",
    suite="llm",
    name="以客服身份作答含问候语",
    description="渲染客服系统提示，验证模型以客服身份作答并含问候语。",
    tags=("instruction",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def role_persona(ctx: TestContext) -> None:
    """断言：以客服身份作答，回复含"您好"。"""
    messages = ctx.prompts.render("support-agent", {
        "company": "Acme",
        "headquarters": "上海市浦东新区",
        "question": "你好，我想咨询退货政策",
    }, version=1)
    client = scripted_or_real(ctx, rules=[
        MockRule(match="退货政策", reply="您好，感谢咨询！Acme 支持 7 天无理由退货。"),
    ])
    resp = await client.generate(messages, temperature=0.0, max_tokens=512)
    assert_contains(resp.text, "您好")


@test(
    id="llm-ins-005",
    suite="llm",
    name="多约束同时满足",
    description="要求模型同时满足词数上限、必含词与禁用词三个约束。",
    tags=("instruction",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def multi_constraints(ctx: TestContext) -> None:
    """断言：词数≤12、含 secure、不含 free 三个约束同时成立。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="12 个英文单词", reply="Acme data is secure and encrypted for all users."),
    ])
    resp = await client.generate(
        [Message.user("用不超过 12 个英文单词回答，必须包含 secure，不得包含 free：Acme 的数据安全如何？")],
        temperature=0.0, max_tokens=512,
    )
    assert_word_count(resp.text, max_words=12)
    assert_contains(resp.text, "secure")
    assert_not_contains(resp.text, "free")


@test(
    id="llm-ins-006",
    suite="llm",
    name="按指定顺序输出",
    description="要求模型先介绍成立时间再介绍总部，验证输出顺序。",
    tags=("instruction",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def output_order(ctx: TestContext) -> None:
    """断言："首先"位于"其次"之前，且包含 2015 与上海两个关键信息。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="按顺序", reply="首先，Acme 成立于 2015 年；其次，总部位于上海。"),
    ])
    resp = await client.generate(
        [Message.user("请按顺序先介绍 Acme 的成立时间，再介绍总部位置。")],
        temperature=0.0, max_tokens=512,
    )
    assert_matches(resp.text, r"首先.*其次", flags=re.DOTALL)
    assert_contains(resp.text, "2015", "上海")
    assert resp.text.find("首先") < resp.text.find("其次")
