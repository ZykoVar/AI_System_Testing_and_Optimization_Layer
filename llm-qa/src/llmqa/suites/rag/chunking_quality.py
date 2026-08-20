"""RAG 分块质量测试套件：块长度、相邻块重叠、文档内容覆盖与 chunk_id 唯一性。

关注点：RAGCorpus.chunk(chunk_size, overlap) 的切片逻辑是否正确——块不超过 chunk_size、
overlap>0 时相邻块首尾相交、每篇文档内容被完整覆盖、块 ID 全库唯一。
分块为纯字符切片（无 tokenizer），故重叠校验按字符口径。
"""
from __future__ import annotations

from llmqa.assertors import AssertionFailed
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness


def _corpus_harness(ctx: TestContext, chunk_size: int, overlap: int) -> RAGHarness:
    """基于共享语料构建指定分块参数的 harness。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    return RAGHarness(corpus, ctx.client(), chunk_size=chunk_size, overlap=overlap)


@test(id="rag-chk-001", suite="rag", name="分块：所有块长度不超过 chunk_size",
      description="共享语料分块后每块长度≤chunk_size",
      tags=("rag", "chunking"), severity=Severity.LOW, timeout=60)
async def chunk_length_bounded(ctx: TestContext) -> None:
    """断言每个块的字符长度不超过 chunk_size=400。"""
    harness = _corpus_harness(ctx, chunk_size=400, overlap=80)
    over = [c.chunk_id for c in harness.chunks if len(c.text) > harness.chunk_size]  # 收集所有越界块一次性报告，而非遇到首个即抛
    if over:
        raise AssertionFailed(
            f"存在超出 chunk_size 的块: {over}",
            metrics={"over_length_blocks": len(over)})


@test(id="rag-chk-002", suite="rag", name="分块：相邻块存在重叠文本",
      description="长文档分块后相邻块首尾字符相交（overlap>0）",
      tags=("rag", "chunking"), severity=Severity.MEDIUM, timeout=60)
async def adjacent_chunks_overlap(ctx: TestContext) -> None:
    """用合成长文档校验 overlap>0 时相邻块首尾字符一致。"""
    long_text = "这是一段用于分块重叠校验的示例文本。" * 40
    corpus = RAGCorpus.from_dicts([{"id": "doc-long", "title": "长文档", "text": long_text}])
    chunk_size, overlap = 100, 20
    harness = RAGHarness(corpus, ctx.client(), chunk_size=chunk_size, overlap=overlap)
    if len(harness.chunks) < 2:  # 单块无法验证重叠，提前失败并给出明确原因
        raise AssertionFailed("长文档未产生多块，无法校验重叠", metrics={"chunks": len(harness.chunks)})
    bad = []
    for i in range(len(harness.chunks) - 1):
        tail = harness.chunks[i].text[-overlap:]  # 字符切片分块，重叠按字符口径校验（前块尾 vs 后块头）
        head = harness.chunks[i + 1].text[:overlap]
        if tail != head:
            bad.append((harness.chunks[i].chunk_id, harness.chunks[i + 1].chunk_id))
    if bad:
        raise AssertionFailed(f"相邻块重叠文本不一致: {bad}",
                              metrics={"bad_pairs": len(bad)})


@test(id="rag-chk-003", suite="rag", name="分块：文档内容完整覆盖",
      description="每篇文档的全文（含关键句）都能被其分块内容重构覆盖",
      tags=("rag", "chunking"), severity=Severity.MEDIUM, timeout=60)
async def document_content_covered(ctx: TestContext) -> None:
    """断言每篇文档全文是其按序分块拼接后的子串（即内容无丢失）。"""
    harness = _corpus_harness(ctx, chunk_size=400, overlap=80)
    missing = []
    for doc in harness.corpus.documents:
        joined = "".join(c.text for c in harness.chunks if c.doc_id == doc.id)  # 仅拼接属于该文档的块，保持文档内顺序
        if doc.text.strip() not in joined:  # 依赖语料文档短于 chunk_size（单块）；若被切分则 overlap 会破坏拼接串的子串关系
            missing.append(doc.id)
    if missing:
        raise AssertionFailed(f"文档内容未被完整覆盖: {missing}",
                              metrics={"missing_docs": len(missing)})


@test(id="rag-chk-004", suite="rag", name="分块：chunk_id 全库唯一",
      description="共享语料分块后所有 chunk_id 互不重复",
      tags=("rag", "chunking"), severity=Severity.MEDIUM, timeout=60)
async def chunk_ids_unique(ctx: TestContext) -> None:
    """用较小 chunk_size 使部分文档产生多块，再校验 chunk_id 无重复。"""
    harness = _corpus_harness(ctx, chunk_size=100, overlap=20)
    ids = [c.chunk_id for c in harness.chunks]
    dup = sorted({i for i in ids if ids.count(i) > 1})  # set 去重 + 排序，稳定输出重复 chunk_id 列表
    if dup:
        raise AssertionFailed(f"chunk_id 重复: {dup}",
                              metrics={"duplicates": len(dup)})
