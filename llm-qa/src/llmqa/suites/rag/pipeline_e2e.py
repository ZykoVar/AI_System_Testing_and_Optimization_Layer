"""RAG 端到端测试套件：golden 一致性、未知问题拒答、端到端延迟与批量查询稳定性。

关注点：从"问题 → 检索 → 构建上下文 → 生成 → 引用"的完整管线是否可用，
以及 golden 答案关键词、拒答行为、单次延迟上限、批量查询无异常等端到端要求。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed, assert_contains
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness

_GOLDEN_KEYWORDS = {
    "rq-001": ["7 天", "无理由退货"],
    "rq-002": ["299"],
    "rq-003": ["199"],
}


def _harness(ctx: TestContext, reply: str) -> RAGHarness:
    """基于共享语料构建 harness，生成客户端用脚本化回复。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply=reply)])
    return RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")


@test(id="rag-e2e-001", suite="rag", name="端到端：golden 答案关键词一致",
      description="rq-001..003 mock 回复=golden_answer，断言回复含关键信息",
      tags=("rag", "e2e", "smoke"), severity=Severity.MEDIUM, timeout=60)
async def golden_consistency(ctx: TestContext) -> None:
    """断言 rq-001..003 的回复包含 golden_answer 的关键词。"""
    queries = {q["id"]: q for q in ctx.datasets.load("rag/queries")["queries"]}
    for qid in ("rq-001", "rq-002", "rq-003"):
        q = queries[qid]
        harness = _harness(ctx, q["golden_answer"])
        resp = await harness.answer(q["query"], k=4)
        assert_contains(resp.text, *_GOLDEN_KEYWORDS[qid])


@test(id="rag-e2e-002", suite="rag", name="端到端：未知问题拒答",
      description="rq-008 mock 回复为无法回答，断言拒答话术",
      tags=("rag", "e2e"), severity=Severity.HIGH, timeout=60)
async def unknown_question_refusal(ctx: TestContext) -> None:
    """断言未知问题（语料无资料）的回复含拒答话术。"""
    q = next(q for q in ctx.datasets.load("rag/queries")["queries"] if q["id"] == "rq-008")
    harness = _harness(ctx, "根据现有资料无法回答。")
    resp = await harness.answer(q["query"], k=4)
    assert_contains(resp.text, "无法回答")


@test(id="rag-e2e-003", suite="rag", name="端到端：单次延迟达标",
      description="answer 端到端延迟低于 settings.thresholds.p95_latency_ms",
      tags=("rag", "e2e"), severity=Severity.MEDIUM, timeout=60)
async def e2e_latency(ctx: TestContext) -> None:
    """断言单次端到端生成延迟低于 P95 阈值。"""
    harness = _harness(ctx, "根据资料，专业版每月 299 元。[资料1]")
    resp = await harness.answer("专业版订阅一个月多少钱？", k=4)
    limit = ctx.settings.thresholds.p95_latency_ms
    if resp.latency_ms >= limit:
        raise AssertionFailed(
            "端到端延迟 {:.1f}ms ≥ 阈值 {:.1f}ms".format(resp.latency_ms, limit),
            metrics={"latency_ms": resp.latency_ms})


@test(id="rag-e2e-004", suite="rag", name="端到端：批量查询无异常",
      description="连续 5 个查询循环 await 均正常返回非空回复",
      tags=("rag", "e2e"), severity=Severity.LOW, timeout=60)
async def batch_queries_no_error(ctx: TestContext) -> None:
    """断言批量 5 个查询均正常完成且回复非空。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply="根据资料作答。[资料1]")])
    harness = RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")
    for q in ctx.datasets.load("rag/queries")["queries"][:5]:
        resp = await harness.answer(q["query"], k=4)
        if not resp.text:
            raise AssertionFailed("查询 {} 返回空回复".format(q["id"]))
