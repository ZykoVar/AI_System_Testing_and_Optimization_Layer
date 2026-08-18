"""RAG 引用质量测试套件：引用格式、引用编号范围与引用内容确实来自检索结果。

关注点：生成答案是否按约定输出 [资料N] 引用、引用编号不越界、被引用的块内容确实出现在答案中，
保证答案可追溯到检索结果（可验证、可审计）。
"""
from __future__ import annotations

import re

from llmqa.assertors import AssertionFailed, assert_contains, assert_matches
from llmqa.clients import MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test
from llmqa.harnesses import RAGCorpus, RAGHarness

_CITE_RE = re.compile(r"\[资料(\d+)\]")


def _citations(text: str) -> list[int]:
    """从回复文本中解析出所有 [资料N] 的编号 N。"""
    return [int(n) for n in _CITE_RE.findall(text)]


def _harness(ctx: TestContext, reply: str) -> RAGHarness:
    """基于共享语料构建 harness，生成客户端用脚本化回复。"""
    corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])
    client = scripted_or_real(ctx, rules=[MockRule(match=".*", reply=reply)])
    return RAGHarness(corpus, client, prompt_manager=ctx.prompts, prompt_id="rag/answer")


@test(id="rag-cit-001", suite="rag", name="引用：回复包含 [资料N] 格式",
      description="answer 回复包含 [资料N] 引用格式",
      tags=("rag", "citation"), severity=Severity.LOW, timeout=60)
async def citation_format_present(ctx: TestContext) -> None:
    """断言回复匹配 [资料N] 正则格式。"""
    harness = _harness(ctx, "根据资料，7 天内可无理由退货。[资料1]")
    resp = await harness.answer("买了东西不想要了，几天内可以退货？", k=4)
    assert_matches(resp.text, r"\[资料\d+\]")


@test(id="rag-cit-002", suite="rag", name="引用：编号不超过实际检索块数",
      description="回复中所有 [资料N] 编号均在 1..k 范围内",
      tags=("rag", "citation"), severity=Severity.MEDIUM, timeout=60)
async def citation_numbers_in_bounds(ctx: TestContext) -> None:
    """断言所有引用编号不超过实际检索到的块数。"""
    harness = _harness(ctx, "支持 7 天无理由退货 [资料1]；用户数据加密存储 [资料2]。")
    question = "买了东西不想要了，几天内可以退货？"
    result = await harness.retrieve(question, k=4)
    resp = await harness.answer(question, k=4)
    nums = _citations(resp.text)
    if not nums:
        raise AssertionFailed("回复未包含任何 [资料N] 引用")
    out_of_bounds = [n for n in nums if n < 1 or n > len(result.chunks)]
    if out_of_bounds:
        raise AssertionFailed(
            "引用编号越界（检索块数 {}）: {}".format(len(result.chunks), out_of_bounds),
            metrics={"cited": nums})


@test(id="rag-cit-003", suite="rag", name="引用：被引资料确实在检索结果中",
      description="解析 [资料N] 编号并对照 result.chunks，被引块内容出现在答案中",
      tags=("rag", "citation"), severity=Severity.MEDIUM, timeout=60)
async def cited_material_exists(ctx: TestContext) -> None:
    """断言每个被引用的块真实存在于检索结果，且其标题出现在答案中。"""
    harness = _harness(ctx, "根据《退货政策》，支持 7 天无理由退货。[资料1]")
    question = "退货政策是什么？"
    result = await harness.retrieve(question, k=4)
    resp = await harness.answer(question, k=4)
    nums = _citations(resp.text)
    if not nums:
        raise AssertionFailed("回复未包含 [资料N] 引用")
    for n in nums:
        if n < 1 or n > len(result.chunks):
            raise AssertionFailed("引用编号 {} 越界".format(n))
        chunk = result.chunks[n - 1]
        assert_contains(resp.text, chunk.title)
