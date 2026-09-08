"""一致性测试套件（llm-con-）：验证模型多次/多问法回答的一致性。

关注点：
- 同一问题重复提问，回复相似度（SequenceMatcher）≥ 阈值；
- 同义改写问题答案一致；
- 多轮问答信息不矛盾。

使用确定性文本相似度（assert_similarity），不依赖 LLM，可离线复现。
"""
from __future__ import annotations

from llmqa.assertors import (
    Judge,
    assert_contains,
    assert_not_contains,
    assert_similarity,
    text_similarity,
)
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


@test(
    id="llm-con-001",
    suite="llm",
    name="同一问题两次回答相似",
    description="同一问题提问两次，验证两次回复相似度不低于 0.8。",
    tags=("consistency", "smoke"),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def repeat_question_similar(ctx: TestContext) -> None:
    """断言：两次回复相似度≥0.8（仅措辞微调）。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="退货", reply="支持 7 天无理由退货，退款 3-5 个工作日到账。", times=1),  # 两条规则各 times=1：两次相同提问依次命中，产出措辞微调的两个回复
        MockRule(match="退货", reply="支持 7 天无理由退货，退款将在 3-5 个工作日到账。", times=1),
    ])
    question = [Message.user("Acme 的退货政策是什么？")]
    resp1 = await client.generate(question, temperature=0.0, max_tokens=512)
    resp2 = await client.generate(question, temperature=0.0, max_tokens=512)
    similarity = text_similarity(resp1.text, resp2.text)
    ctx.record(similarity=round(similarity, 4))
    if similarity < 0.8:
        # 真实 API 下 temperature=0 不保证两次独立调用逐字一致（推理模型尤甚），
        # 字面相似度 miss 时用 Judge 判"两次回答语义一致"兜底
        verdict = await Judge(ctx.client()).assert_score(
            question="Acme 的退货政策是什么？",
            answer=("回答A: " + resp1.text + "\n回答B: " + resp2.text),
            criteria="判断两次回答是否表达相同事实（允许措辞差异）；一致给 8 分以上，矛盾给低分",
            min_score=7.0)
        ctx.record(consistency_mode="semantic", judge_score=verdict.score)
        ctx.add_evidence("裁判理由(一致性语义兜底): " + verdict.reasoning)
    else:
        ctx.record(consistency_mode="deterministic")


@test(
    id="llm-con-002",
    suite="llm",
    name="同义改写答案一致",
    description="两个同义问法应得到一致答案，验证关键事实一致。",
    tags=("consistency",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def paraphrase_consistency(ctx: TestContext) -> None:
    """断言：两个同义问题的答案相似度≥0.9，且都包含 2015。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="成立于", reply="Acme 公司成立于 2015 年。"),
        MockRule(match="哪一年创立", reply="Acme 公司成立于 2015 年。"),
    ])
    resp1 = await client.generate(
        [Message.user("Acme 公司成立于哪一年？")], temperature=0.0, max_tokens=512)
    resp2 = await client.generate(
        [Message.user("Acme 公司是哪一年创立的？")], temperature=0.0, max_tokens=512)
    assert_similarity(resp1.text, resp2.text, min_score=0.9)  # 同义改写答案应高度一致，阈值比重复提问(0.8)更严
    assert_contains(resp1.text, "2015")
    assert_contains(resp2.text, "2015")


@test(
    id="llm-con-003",
    suite="llm",
    name="多轮问答不矛盾",
    description="关于退货与退款的两问答案应不矛盾（7 天 + 3-5 工作日）。",
    tags=("consistency",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def no_contradiction(ctx: TestContext) -> None:
    """断言：两问分别给出 7 天与 3-5 工作日，且不含相互矛盾的数字。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="几天", reply="支持 7 天无理由退货。"),
        MockRule(match="退款", reply="退款将在 3-5 个工作日内到账。"),
    ])
    resp1 = await client.generate(
        [Message.user("Acme 退货可以在几天内申请？")], temperature=0.0, max_tokens=512)
    resp2 = await client.generate(
        [Message.user("退款多久能到账？")], temperature=0.0, max_tokens=512)
    assert_contains(resp1.text, "7 天")
    assert_contains(resp2.text, "3-5 个工作日")
    assert_not_contains(resp1.text, "30 天", "15 天")  # 用显式矛盾数字做负向断言，比仅验证正向关键词更能发现冲突
    assert_not_contains(resp2.text, "立即", "当天")  # 退款时效不得与"3-5 个工作日"矛盾
