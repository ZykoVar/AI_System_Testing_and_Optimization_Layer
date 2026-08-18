"""RAG 测试 Harness：可复现的检索 + 生成管线。

- 内置纯 Python BM25-lite 词法检索（中英文混排分词），无外部依赖，结果确定可复现；
- 检索质量指标：recall@k / hit@k / MRR / precision@k；
- 生成质量交给 assertors.judge（忠实性/相关性）或引用断言。
"""
from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from llmqa.clients.base import LLMClient, LLMResponse, Message

DEFAULT_RAG_PROMPT = (
    "你是一名基于检索资料作答的助手。只能依据【参考资料】回答；"
    "若资料不足，必须明确说明无法回答，不得编造。\n\n"
    "【参考资料】\n{context}\n\n【问题】\n{question}"
)


class RAGDocument(BaseModel):
    id: str
    title: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    text: str = ""
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    """检索评估样例：query 及标注的相关文档/块。"""
    id: str
    query: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    golden_answer: str = ""


class RetrievalMetrics(BaseModel):
    queries: int = 0
    mean_recall_at_k: float = 0.0
    mean_hit_at_k: float = 0.0
    mean_mrr: float = 0.0
    mean_precision_at_k: float = 0.0
    per_query: list[dict[str, Any]] = Field(default_factory=list)


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按字。"""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class RAGCorpus:
    def __init__(self, documents: list[RAGDocument]):
        self.documents = documents

    @classmethod
    def from_dicts(cls, docs: list[dict]) -> "RAGCorpus":
        return cls([RAGDocument(**d) for d in docs])

    def chunk(self, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            text = doc.text.strip()
            if not text:
                continue
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(Chunk(
                    chunk_id="{}#{}".format(doc.id, idx), doc_id=doc.id,
                    title=doc.title, text=text[start:end], index=idx,
                    metadata=dict(doc.metadata)))
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
                idx += 1
        return chunks


class RAGHarness:
    """检索 + 生成双阶段管线，评估口径统一。"""

    def __init__(self, corpus: RAGCorpus | list[dict], client: LLMClient, *,
                 chunk_size: int = 400, overlap: int = 80, top_k: int = 4,
                 prompt_manager=None, prompt_id: str = "rag/answer"):
        if isinstance(corpus, list):
            corpus = RAGCorpus.from_dicts(corpus)
        self.corpus = corpus
        self.client = client
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.prompt_manager = prompt_manager
        self.prompt_id = prompt_id
        self.chunks: list[Chunk] = corpus.chunk(chunk_size, overlap)
        self._df: dict[str, int] = {}
        self._index()

    # ---------- BM25-lite 索引 ----------
    def _index(self) -> None:
        self._doc_tokens: list[list[str]] = [tokenize(c.text) for c in self.chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_tokens))
        self._df = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        n = len(self._doc_tokens)
        k1, b = 1.5, 0.75
        scores = [0.0] * n
        for term in set(query_tokens):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for i, tokens in enumerate(self._doc_tokens):
                tf = tokens.count(term)
                if tf:
                    denom = tf + k1 * (1 - b + b * self._doc_len[i] / self._avgdl)
                    scores[i] += idf * tf * (k1 + 1) / denom
        return scores

    async def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        scores = self._bm25_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: (k or self.top_k)]
        return RetrievalResult(
            query=query,
            chunks=[self.chunks[i] for i in order if scores[i] > 0],
            scores=[round(scores[i], 4) for i in order if scores[i] > 0],
        )

    def build_context(self, result: RetrievalResult) -> str:
        parts = []
        for i, c in enumerate(result.chunks, 1):
            parts.append("[资料{} 来自文档《{}》]\n{}".format(i, c.title or c.doc_id, c.text))
        return "\n\n".join(parts)

    async def answer(self, question: str, *, k: int | None = None,
                     temperature: float = 0.0, max_tokens: int = 512) -> LLMResponse:
        result = await self.retrieve(question, k)
        context = self.build_context(result)
        if self.prompt_manager is not None:
            messages = self.prompt_manager.render(
                self.prompt_id, {"context": context, "question": question})
        else:
            messages = [Message.system(
                DEFAULT_RAG_PROMPT.format(context=context, question=question))]
        return await self.client.generate(messages, temperature=temperature, max_tokens=max_tokens)

    # ---------- 检索质量评估 ----------
    def evaluate_retrieval(self, queries: list[RetrievalQuery],
                           k: int | None = None) -> RetrievalMetrics:
        per_query: list[dict[str, Any]] = []
        recalls, hits, mrrs, precs = [], [], [], []
        for q in queries:
            # 同步评估（纯计算，不需要事件循环）
            scores = self._bm25_scores(tokenize(q.query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[: (k or self.top_k)]
            retrieved_docs = [self.chunks[i].doc_id for i in order if scores[i] > 0]
            retrieved_chunks = [self.chunks[i].chunk_id for i in order if scores[i] > 0]
            rel_docs = set(q.relevant_doc_ids)
            rel_chunks = set(q.relevant_chunk_ids) or {
                c.chunk_id for c in self.chunks if c.doc_id in rel_docs}
            hit_docs = [d for d in retrieved_docs if d in rel_docs]
            recall = len(set(hit_docs)) / len(rel_docs) if rel_docs else 1.0
            hit = 1.0 if hit_docs else 0.0
            mrr = 0.0
            for rank, d in enumerate(retrieved_docs, 1):
                if d in rel_docs:
                    mrr = 1.0 / rank
                    break
            rel_retrieved_chunks = [c for c in retrieved_chunks if c in rel_chunks]
            prec = len(rel_retrieved_chunks) / max(1, len(retrieved_chunks))
            recalls.append(recall); hits.append(hit); mrrs.append(mrr); precs.append(prec)
            per_query.append({
                "id": q.id, "query": q.query, "retrieved_docs": retrieved_docs,
                "recall_at_k": round(recall, 4), "hit_at_k": hit,
                "mrr": round(mrr, 4), "precision_at_k": round(prec, 4),
            })
        n = len(queries)
        return RetrievalMetrics(
            queries=n,
            mean_recall_at_k=round(sum(recalls) / n, 4) if n else 0.0,
            mean_hit_at_k=round(sum(hits) / n, 4) if n else 0.0,
            mean_mrr=round(sum(mrrs) / n, 4) if n else 0.0,
            mean_precision_at_k=round(sum(precs) / n, 4) if n else 0.0,
            per_query=per_query,
        )
