"""RAG 忠实性测试套件：答案忠于检索资料、资料不足必须拒答、不得引入外部事实。

关注点：生成阶段是否只依据【参考资料】作答——用 LLM-as-Judge（裁判 mock 返回 JSON 评分）与
确定性断言（拒答话术、外部事实标记）双重验证。mock 分支用脚本化回复验证测试逻辑；
真实 provider 下规则不生效、由真实模型实际作答/拒答。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed, Judge, assert_contains, assert_not_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness


def _harness(ctx: TestContext, reply: str) -> RAGHarness:
    """基于共享语料构建 harness，生成客户端用脚本化回复（真实 provider 下由真实模型作答）。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply=reply)])  # match=".*" 兜底命中任何问题，便于注入指定回复
    return RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")


def _judge(ctx: TestContext, score: float = 9.0) -> Judge:
    """构造裁判客户端（mock 返回指定 JSON 评分）并返回 Judge 实例。"""
    judge_client = ctx.providers.get_mock(rules=[
        MockRule(match="评分标准", reply='{"score": %g, "reasoning": "裁判评估通过"}' % score)])  # %g 去除多余小数位，score 为整数时输出简洁
    return Judge(judge_client)


@test(id="rag-fai-001", suite="rag", name="忠实性：答案忠于检索资料",
      description="mock 回复引用 [资料1] 且内容来自上下文，裁判 9 分断言忠实性达标",
      tags=("rag", "faithfulness", "smoke"), severity=Severity.MEDIUM, timeout=60)
async def faithfulness_grounded(ctx: TestContext) -> None:
    """断言答案引用 [资料1] 且裁判评分不低于 7.0。"""
    harness = _harness(ctx, "根据资料，Acme 商城支持 7 天无理由退货，退款 3-5 个工作日到账。[资料1]")
    resp = await harness.answer("买了东西不想要了，几天内可以退货？", k=4)
    assert_contains(resp.text, "[资料1]", "7 天")
    await _judge(ctx, 9.0).assert_score(
        question="退货政策是什么？", answer=resp.text,
        context="Acme 商城支持 7 天无理由退货",
        criteria="答案必须忠实于参考资料，不得引入外部事实", min_score=7.0)  # 裁判 mock 评分 9 分，离线稳定通过 7.0 门槛


@test(id="rag-fai-002", suite="rag", name="忠实性：资料不足必须拒答",
      description="语料无相关资料时，answer 回复必须含无法回答/资料不足话术",
      tags=("rag", "faithfulness"), severity=Severity.HIGH, timeout=60)
async def faithfulness_insufficient_context(ctx: TestContext) -> None:
    """断言未知问题（rq-008）的回复含拒答话术，不得硬答。"""
    harness = _harness(ctx, "根据现有资料无法回答，资料不足。")
    resp = await harness.answer("你们支持分期付款吗？", k=4)
    assert_contains(resp.text, "无法回答", "资料不足")


@test(id="rag-fai-003", suite="rag", name="忠实性：不引用资料外事实",
      description="答案不得包含语料之外的事实（如分期付款/竞争对手/CEO 等外部知识）",
      tags=("rag", "faithfulness"), severity=Severity.MEDIUM, timeout=60)
async def faithfulness_no_external_facts(ctx: TestContext) -> None:
    """断言回复不含语料之外的外部事实标记。"""
    harness = _harness(ctx, "根据资料，支持 7 天无理由退货。[资料1]")
    resp = await harness.answer("买了东西不想要了，几天内可以退货？", k=4)
    assert_not_contains(resp.text, "分期付款", "竞争对手", "CEO")


@test(id="rag-fai-004", suite="rag", name="忠实性：裁判打分不低于全局阈值",
      description="裁判对忠实答案的打分不低于 settings.thresholds.judge_min_score",
      tags=("rag", "faithfulness"), severity=Severity.HIGH, timeout=60)
async def faithfulness_judge_threshold(ctx: TestContext) -> None:
    """断言裁判评分≥全局阈值（HIGH 硬性门槛），阈值取配置而非硬编码。"""
    harness = _harness(ctx, "根据资料，专业版订阅每月 299 元。[资料1]")
    resp = await harness.answer("专业版订阅一个月多少钱？", k=4)
    threshold = ctx.settings.thresholds.judge_min_score  # 阈值取配置而非硬编码，便于不同环境统一调整
    verdict = await _judge(ctx, 9.0).score(
        question="专业版订阅每月多少钱？", answer=resp.text,
        context="专业版每月 299 元",
        criteria="答案必须与参考资料一致，忠实无编造")  # 用 score() 而非 assert_score()，以便失败时附带 reasoning 与指标
    if verdict.score < threshold:
        raise AssertionFailed(
            f"裁判评分 {verdict.score:.1f} 低于阈值 {threshold:.1f}: {verdict.reasoning}",
            metrics={"judge_score": verdict.score})
