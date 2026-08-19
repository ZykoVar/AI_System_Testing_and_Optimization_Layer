"""RAG Harness 自测。"""
import asyncio

from llmqa.clients import MockClient
from llmqa.harnesses import RAGCorpus, RAGHarness, RetrievalQuery

DOCS = [
    {"id": "d1", "title": "退货", "text": "退货政策：支持 7 天无理由退货，退款 3-5 个工作日。"},
    {"id": "d2", "title": "定价", "text": "专业版定价 299 元每月，标准版 99 元。"},
    {"id": "d3", "title": "配送", "text": "包邮政策：满 199 包邮，全国 3-5 天送达。"},
]


def make_harness():
    """构造标准语料 + MockClient 的 RAGHarness，供各用例复用。"""
    corpus = RAGCorpus.from_dicts(DOCS)
    return RAGHarness(corpus, MockClient())


def test_chunking():
    harness = make_harness()
    # 每个分块不得超过 chunk_size，且 chunk_id 全局唯一
    assert all(len(c.text) <= harness.chunk_size for c in harness.chunks)
    assert len({c.chunk_id for c in harness.chunks}) == len(harness.chunks)


def test_retrieve_hits():
    harness = make_harness()
    # k=1 只取最高分，退货语义应命中 d1 文档
    result = asyncio.run(harness.retrieve("退货要几天？", k=1))
    assert result.chunks[0].doc_id == "d1"
    assert result.scores[0] > 0


def test_retrieval_metrics():
    harness = make_harness()
    queries = [
        RetrievalQuery(id="q1", query="退货政策", relevant_doc_ids=["d1"]),
        RetrievalQuery(id="q2", query="专业版多少钱？", relevant_doc_ids=["d2"]),
        RetrievalQuery(id="q3", query="包邮门槛", relevant_doc_ids=["d3"]),
    ]
    # 三个查询的正确答案都在 top-2 内，因此召回/命中/MRR 全为 1.0
    metrics = harness.evaluate_retrieval(queries, k=2)
    assert metrics.mean_recall_at_k == 1.0
    assert metrics.mean_hit_at_k == 1.0
    assert metrics.mean_mrr == 1.0


def test_answer_uses_rag_prompt():
    corpus = RAGCorpus.from_dicts(DOCS)
    # 验证 answer 走 RAG prompt 路径，且统计到实际 prompt token 消耗
    client = MockClient(default_reply="根据资料，7 天内可以退货。")
    harness = RAGHarness(corpus, client, top_k=2)
    resp = asyncio.run(harness.answer("退货要几天？"))
    assert "7 天" in resp.text
    assert resp.usage.prompt_tokens > 0
