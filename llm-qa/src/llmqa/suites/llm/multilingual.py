"""多语言测试套件（llm-ml-）：验证模型的语言适配与翻译能力。

关注点：
- 中文提问→中文回答（assert_in_language zh）；
- 英文提问→英文回答（assert_in_language en）；
- 中译英翻译任务（assert_contains 英文词 + assert_in_language en）。

语言判定采用启发式占比（CJK/Latin），无需 NLP 依赖。
"""
from __future__ import annotations

from llmqa.assertors import assert_contains, assert_in_language
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(
    id="llm-ml-001",
    suite="llm",
    name="中文提问得到中文回答",
    description="中文提问应得到中文回答，验证主要语言为中文。",
    tags=("multilingual",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def zh_answer(ctx: TestContext) -> None:
    """断言：回复中文（CJK）占比≥0.5。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="总部", reply="Acme 公司总部位于上海市浦东新区。"),
    ])
    resp = await client.generate(
        [Message.user("Acme 公司的总部在哪里？")], temperature=0.0, max_tokens=512)
    assert_in_language(resp.text, "zh")


@test(
    id="llm-ml-002",
    suite="llm",
    name="英文提问得到英文回答",
    description="英文提问应得到英文回答，验证主要语言为英文。",
    tags=("multilingual",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def en_answer(ctx: TestContext) -> None:
    """断言：回复英文（Latin）占比≥0.5。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="headquarter", reply="Acme is headquartered in Shanghai."),
    ])
    resp = await client.generate(
        [Message.user("Where is Acme headquartered?")], temperature=0.0, max_tokens=512)
    assert_in_language(resp.text, "en")


@test(
    id="llm-ml-003",
    suite="llm",
    name="中译英翻译任务",
    description="中译英任务应输出英文译文，验证译文包含英文词且为英文。",
    tags=("multilingual",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def translate_zh_to_en(ctx: TestContext) -> None:
    """断言：译文包含 hello 且主要语言为英文。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="翻译", reply="Hello"),
    ])
    resp = await client.generate(
        [Message.user("请把'你好'翻译成英文。")], temperature=0.0, max_tokens=512)
    assert_contains(resp.text, "hello")
    assert_in_language(resp.text, "en")
