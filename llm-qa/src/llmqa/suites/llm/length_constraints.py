"""长度约束测试套件（llm-len-）：验证模型对输出长度的控制。

关注点：
- 一句话作答（字符数上限，assert_char_length max）；
- 最少字符数要求（assert_char_length min）；
- 输出 token 上限（resp.usage.completion_tokens ≤ max_tokens）。

mock 分支用脚本模拟理想长度以验证测试逻辑，真实 Provider 下由真实模型遵循长度约束。
"""
from __future__ import annotations

from llmqa.assertors import assert_char_length
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(
    id="llm-len-001",
    suite="llm",
    name="一句话作答（字符上限）",
    description="要求模型用一句话作答，验证回复字符数不超过上限。",
    tags=("length",),
    severity=Severity.LOW,
    timeout=60,
)
async def one_sentence(ctx: TestContext) -> None:
    """断言：回复字符数≤30。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="一句话", reply="Acme 公司成立于 2015 年。"),
    ])
    resp = await client.generate(
        [Message.user("请用一句话回答 Acme 的成立时间。")], temperature=0.0, max_tokens=512)
    assert_char_length(resp.text, max_chars=30)  # 字符口径（len(text)），对中文逐字计数更直观


@test(
    id="llm-len-002",
    suite="llm",
    name="最少字符数要求",
    description="要求模型详细作答，验证回复字符数不少于下限。",
    tags=("length",),
    severity=Severity.LOW,
    timeout=60,
)
async def min_chars(ctx: TestContext) -> None:
    """断言：回复字符数≥50。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="详细",
                 reply="Acme 商城的退货政策为：自签收之日起 7 天内可申请无理由退货，商品需保持完好；退款将在审核通过后的 3-5 个工作日内原路退回。"),
    ])
    resp = await client.generate(
        [Message.user("请详细说明 Acme 的退货政策，至少 50 个字符。")], temperature=0.0, max_tokens=512)
    assert_char_length(resp.text, min_chars=50)  # 下限校验只保底不保质量，常与关键词断言配合使用


@test(
    id="llm-len-003",
    suite="llm",
    name="输出 token 数不超过上限",
    description="验证一次生成输出的 token 数不超过请求的 max_tokens。",
    tags=("length",),
    severity=Severity.LOW,
    timeout=60,
)
async def token_limit(ctx: TestContext) -> None:
    """断言：resp.usage.completion_tokens ≤ 请求的 max_tokens。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="50 字", reply="退货政策为 7 天无理由退货，退款 3-5 个工作日到账。"),  # 回复短于 32 token 上限，离线可稳定通过
    ])
    max_tokens = 32  # 用较小上限制造张力：mock 回复简短，真实模型需真正收敛在 32 token 内
    resp = await client.generate(
        [Message.user("请用不超过 50 字回答 Acme 的退货政策。")],
        temperature=0.0, max_tokens=max_tokens,
    )
    assert resp.usage.completion_tokens <= max_tokens, \
        f"输出 token {resp.usage.completion_tokens} 超过上限 {max_tokens}"  # usage 由 provider 回填；mock 按估算、真实 provider 以 API 返回为准
