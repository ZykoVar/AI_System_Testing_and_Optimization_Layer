"""RAG 相关性测试套件：检索结果与查询相关、答案与问题相关、空上下文不得硬答。

关注点：检索 top1 是否命中标注相关文档；生成答案是否切题（LLM-as-Judge 相关性评分）；
以及 build_context 为空（无任何检索块）时模型必须拒答而非编造。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed, Judge, assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness


@test(id="rag-rel-001", suite="rag", name="相关性：检索 top1 命中相关文档",
      description="rq-001..003 的 top1 检索文档必须位于标注的 relevant_doc_ids 中",
      tags=("rag", "relevance"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_top1_relevant(ctx: TestContext) -> None:
    """断言 rq-001/002/003 的 top1 文档命中标注相关文档。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="占位回复")])
    harness = RAGHarness(corpus, client, chunk_size=400, overlap=80, top_k=4)
    queries = {q["id"]: q for q in ctx.datasets.load("rag/queries")["queries"]}
    for qid in ("rq-001", "rq-002", "rq-003"):
        q = queries[qid]
        result = await harness.retrieve(q["query"], k=1)  # k=1 只取 top1，校验最相关文档是否命中标注
        if not result.chunks:
            raise AssertionFailed(f"{qid} 检索结果为空")
        top_doc = result.chunks[0].doc_id  # retrieve 按 BM25 分数降序返回，chunks[0] 即 top1
        if top_doc not in q["relevant_doc_ids"]:  # 用 doc 粒度（非 chunk）判断命中，容忍同文档被切成多块
            raise AssertionFailed(
                "{} top1 文档 {} 不在相关文档 {} 中".format(qid, top_doc, q["relevant_doc_ids"]),
                metrics={"top1_hit": 0})


@test(id="rag-rel-002", suite="rag", name="相关性：答案与问题相关",
      description="mock 相关回复 + 裁判 9 分，断言答案切题",
      tags=("rag", "relevance"), severity=Severity.MEDIUM, timeout=60)
async def answer_relevance(ctx: TestContext) -> None:
    """断言答案直接回答免运费门槛问题，裁判评分不低于 7.0。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="单笔订单满 199 元即可包邮。[资料1]")])
    harness = RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")
    resp = await harness.answer("多少钱可以免运费？", k=4)
    judge_client = ctx.providers.get_mock(rules=[
        MockRule(match="评分标准", reply='{"score": 9, "reasoning": "答案切题"}')])
    await Judge(judge_client).assert_score(
        question="多少钱可以免运费？", answer=resp.text,
        context="单笔订单满 199 元包邮",
        criteria="答案必须直接回答免运费门槛问题", min_score=7.0)


@test(id="rag-rel-003", suite="rag", name="相关性：空上下文不得硬答",
      description="空语料下 build_context 为空串，answer 生成必须拒答",
      tags=("rag", "relevance"), severity=Severity.HIGH, timeout=60)
async def empty_context_no_hard_answer(ctx: TestContext) -> None:
    """断言空语料检索为空、build_context 为空串，且 answer 回复含拒答话术。"""
    corpus = RAGCorpus.from_dicts([])  # 空语料构造空库，验证检索为空时的降级路径
    client = scripted_or_real(ctx, rules=[
        MockRule(match=".*", reply="抱歉，当前没有可用资料，无法回答您的问题。")])
    harness = RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")
    result = await harness.retrieve("任意问题", k=4)
    if harness.build_context(result) != "":  # 空结果集下 build_context 必须返回空串，否则会把空资料占位注入提示词
        raise AssertionFailed(f"空语料下 build_context 应为空串，实际: {harness.build_context(result)!r}")
    resp = await harness.answer("任意问题", k=4)
    assert_contains(resp.text, "无法回答")
