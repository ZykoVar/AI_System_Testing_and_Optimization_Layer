"""RAG 检索质量测试套件：BM25-lite 词法检索的召回、命中、MRR、精确率与跨文档查询。

关注点：内置纯 Python BM25-lite 检索器（无外部依赖）在 rag/corpus + rag/queries 数据集上的
召回率 / 命中率 / MRR / 精确率指标，以及跨文档查询（rq-007）能否同时召回多个相关文档。

阈值说明（实测值，默认 chunk_size=400、k=4）：mean_recall_at_k=1.0、mean_hit_at_k=0.875、
mean_mrr=0.875、mean_precision_at_k=0.25。mean_precision_at_k 因"单相关文档查询在 k=4 时
最多命中 1/4 块"的词法检索口径天然 ≤0.25，故阈值定为 0.2（初稿 0.5 不可达）；其余保留原阈值。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness, RetrievalQuery


def _build(ctx: TestContext):
    """基于共享语料 + 查询集构建 harness（k=4）与查询列表。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="占位回复")])  # 检索阶段不调用生成，占位回复仅保证 harness 构造完整
    harness = RAGHarness(corpus, client, prompt_manager=ctx.prompts,
                         prompt_id="rag/answer", chunk_size=400, overlap=80, top_k=4)
    queries = [RetrievalQuery(**q) for q in ctx.datasets.load("rag/queries")["queries"]]  # **q 展开数据集字段构造查询对象，字段需与 RetrievalQuery 签名一致
    return harness, queries


@test(id="rag-ret-001", suite="rag", name="检索：全查询集平均召回率达标",
      description="BM25-lite 全查询集 mean_recall_at_k≥0.8（实测 1.0）",
      tags=("rag", "retrieval", "smoke"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_mean_recall(ctx: TestContext) -> None:
    """断言 evaluate_retrieval(k=4) 的 mean_recall_at_k 不低于 0.8。"""
    harness, queries = _build(ctx)
    m = await harness.evaluate_retrieval(queries, k=4)
    if m.mean_recall_at_k < 0.8:
        raise AssertionFailed(
            f"平均召回率 {m.mean_recall_at_k:.2f} 低于阈值 0.8",
            metrics={"mean_recall_at_k": m.mean_recall_at_k})


@test(id="rag-ret-002", suite="rag", name="检索：全查询集平均命中率达标",
      description="BM25-lite 全查询集 mean_hit_at_k≥0.8（实测 0.875）",
      tags=("rag", "retrieval"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_mean_hit(ctx: TestContext) -> None:
    """断言 evaluate_retrieval(k=4) 的 mean_hit_at_k 不低于 0.8。"""
    harness, queries = _build(ctx)
    m = await harness.evaluate_retrieval(queries, k=4)
    if m.mean_hit_at_k < 0.8:
        raise AssertionFailed(
            f"平均命中率 {m.mean_hit_at_k:.2f} 低于阈值 0.8",
            metrics={"mean_hit_at_k": m.mean_hit_at_k})


@test(id="rag-ret-003", suite="rag", name="检索：全查询集平均 MRR 达标",
      description="BM25-lite 全查询集 mean_mrr≥0.7（实测 0.875）",
      tags=("rag", "retrieval"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_mean_mrr(ctx: TestContext) -> None:
    """断言 evaluate_retrieval(k=4) 的 mean_mrr 不低于 0.7。"""
    harness, queries = _build(ctx)
    m = await harness.evaluate_retrieval(queries, k=4)
    if m.mean_mrr < 0.7:
        raise AssertionFailed(
            f"平均 MRR {m.mean_mrr:.2f} 低于阈值 0.7",
            metrics={"mean_mrr": m.mean_mrr})


@test(id="rag-ret-004", suite="rag", name="检索：全查询集平均精确率达标",
      description="BM25-lite 全查询集 mean_precision_at_k≥0.2（实测 0.25，初稿 0.5 不可达）",
      tags=("rag", "retrieval"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_mean_precision(ctx: TestContext) -> None:
    """断言 evaluate_retrieval(k=4) 的 mean_precision_at_k 不低于 0.2。

    说明：单相关文档查询在 k=4 下精确率上限为 1/4，故 0.5 目标无法用词法检索达到。
    """
    harness, queries = _build(ctx)
    m = await harness.evaluate_retrieval(queries, k=4)
    if m.mean_precision_at_k < 0.2:  # 单相关文档查询在 k=4 下精确率上限 1/4，故阈值降至 0.2（见模块 docstring）
        raise AssertionFailed(
            f"平均精确率 {m.mean_precision_at_k:.2f} 低于阈值 0.2",
            metrics={"mean_precision_at_k": m.mean_precision_at_k})


@test(id="rag-ret-005", suite="rag", name="检索：跨文档查询召回完整",
      description="跨文档查询 rq-007 的 per_query recall==1.0（两个相关文档全部命中）",
      tags=("rag", "retrieval"), severity=Severity.MEDIUM, timeout=60)
async def retrieval_cross_doc_recall(ctx: TestContext) -> None:
    """断言 rq-007（退货+保修）能同时召回 doc-return 与 doc-warranty。"""
    harness, queries = _build(ctx)
    m = await harness.evaluate_retrieval(queries, k=4)
    pq = next(p for p in m.per_query if p["id"] == "rq-007")  # per_query 逐查询给出指标，用 next 取跨文档查询 rq-007
    if pq["recall_at_k"] != 1.0:  # 跨文档查询要求两个相关文档全部命中，必须严格等于 1.0 而非 ≥
        raise AssertionFailed(
            "rq-007 跨文档召回 {:.2f} 未达到 1.0，检索到: {}".format(
                pq["recall_at_k"], pq["retrieved_docs"]),
            metrics={"rq_007_recall": pq["recall_at_k"]})
