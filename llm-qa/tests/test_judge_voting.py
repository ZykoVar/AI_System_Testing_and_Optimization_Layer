"""Judge 增强自测：托管模板接入 + 多次投票聚合。"""
import asyncio
from pathlib import Path

import pytest

from llmqa.assertors import Judge
from llmqa.clients import MockClient, MockRule
from llmqa.prompts import PromptManager

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def manager() -> PromptManager:
    return PromptManager(ROOT / "prompts").load()


def test_single_pass_backward_compat():
    client = MockClient(rules=[MockRule(
        match="评分标准", reply='{"score": 9, "reasoning": "准确"}')])
    verdict = run(Judge(client).score(
        question="1+1=?", answer="2", criteria="答案必须正确"))
    assert verdict.score == 9
    assert verdict.passes == 1
    assert verdict.scores == [9]
    assert verdict.agreement == 1.0


def test_managed_template_via_prompt_manager(manager):
    # 走版本化托管模板（judge/correctness.yaml），模板 user 消息含"【评分标准】"
    client = MockClient(rules=[MockRule(
        match="评分标准", reply='{"score": 8, "reasoning": "基本正确"}')])
    judge = Judge(client, prompt_manager=manager)
    verdict = run(judge.score(question="总部在哪？", answer="上海", context="",
                              criteria="必须指出上海"))
    assert verdict.score == 8


def test_multi_pass_voting_median():
    # 三次投票分数 6/8/9：中位数 8，一致度 1-(9-6)/10=0.7
    rules = [
        MockRule(match="评分标准", reply='{"score": 6, "reasoning": "偏低"}', times=1),
        MockRule(match="评分标准", reply='{"score": 9, "reasoning": "偏高"}', times=1),
        MockRule(match="评分标准", reply='{"score": 8, "reasoning": "适中"}'),
    ]
    verdict = run(Judge(MockClient(rules=rules)).score(
        question="q", answer="a", criteria="c", passes=3))
    assert verdict.score == 8
    assert verdict.passes == 3
    assert verdict.scores == [6.0, 8.0, 9.0]
    assert verdict.agreement == pytest.approx(0.7, abs=1e-6)
    assert verdict.reasoning == "适中"


def test_assert_score_failure_carries_evidence():
    # 低分失败时，裁判理由与投票明细作为证据随断言传播
    from llmqa.assertors import AssertionFailed
    client = MockClient(rules=[MockRule(
        match="评分标准", reply='{"score": 3, "reasoning": "答案错误"}')])
    with pytest.raises(AssertionFailed) as exc_info:
        run(Judge(client).assert_score(
            question="q", answer="a", criteria="c", min_score=7.0))
    evidence = "\n".join(exc_info.value.evidence)
    assert "答案错误" in evidence
    assert exc_info.value.metrics["judge_score"] == 3


def test_assert_score_with_voting():
    client = MockClient(rules=[MockRule(
        match="评分标准", reply='{"score": 9, "reasoning": "好"}')])
    verdict = run(Judge(client, passes=3).assert_score(
        question="q", answer="a", criteria="c", min_score=7.0))
    assert verdict.score == 9
    assert verdict.passes == 3
