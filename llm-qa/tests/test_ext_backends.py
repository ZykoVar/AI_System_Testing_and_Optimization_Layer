"""可插拔成熟工具后端（llmqa.ext）自测。

本机未安装 litellm/ragas，因此实测：
1. 依赖缺失时构造抛 LLMError 且携带安装指引（离线 CI 不被破坏）；
2. JudgeBackend 协议注入：外部后端被正确委托，投票/证据语义仍由 Judge 持有；
3. RAGHarness 外部检索后端（已在 test_rag_harness 覆盖）。
"""
import asyncio

import pytest

from llmqa.assertors import Judge, JudgeVerdict
from llmqa.clients import LLMError, MockClient, MockRule
from llmqa.config import ProviderConfig


def run(coro):
    return asyncio.run(coro)


def test_litellm_backend_requires_install():
    from llmqa.ext.litellm_backend import LiteLLMClient
    with pytest.raises(LLMError) as exc_info:
        LiteLLMClient("litellm", "openai/gpt-4o-mini")
    assert "pip install" in str(exc_info.value)


def test_factory_litellm_kind_requires_install():
    from llmqa.clients.factory import build_client
    provider = ProviderConfig(name="litellm", kind="litellm",
                              model="openai/gpt-4o-mini")
    with pytest.raises(LLMError):
        build_client(provider)


def test_ragas_backend_requires_install():
    from llmqa.ext.ragas_backend import RagasJudgeBackend
    with pytest.raises(LLMError) as exc_info:
        RagasJudgeBackend(MockClient())
    assert "pip install" in str(exc_info.value)


class FakeEngineBackend:
    """模拟外部评测引擎：协议注入后 Judge 应委托它并保留投票语义。"""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    async def score(self, *, question, answer, context, criteria, scale,
                    temperature=0.0):
        self.calls += 1
        s = self.scores.pop(0)
        return JudgeVerdict(score=s, reasoning="engine-{}".format(s))


def test_judge_delegates_to_external_backend():
    backend = FakeEngineBackend([8.0])
    judge = Judge(MockClient(), backend=backend)
    verdict = run(judge.score(question="q", answer="a", criteria="c"))
    assert verdict.score == 8.0
    assert verdict.reasoning == "engine-8.0"
    assert backend.calls == 1


def test_judge_voting_still_owned_by_judge_with_external_backend():
    # 投票聚合（中位数/一致度）不因后端替换而丢失
    backend = FakeEngineBackend([6.0, 9.0, 8.0])
    judge = Judge(MockClient(), backend=backend, passes=3)
    verdict = run(judge.score(question="q", answer="a", criteria="c"))
    assert verdict.score == 8.0
    assert verdict.passes == 3
    assert verdict.scores == [6.0, 8.0, 9.0]
    assert verdict.agreement == pytest.approx(0.7, abs=1e-6)
    assert backend.calls == 3


def test_judge_backend_evidence_propagation():
    # 外部后端低分 → AssertionFailed 仍携带证据（reasoning 随证据传播）
    from llmqa.assertors import AssertionFailed
    backend = FakeEngineBackend([3.0])
    judge = Judge(MockClient(), backend=backend)
    with pytest.raises(AssertionFailed) as exc_info:
        run(judge.assert_score(question="q", answer="a", criteria="c", min_score=7.0))
    assert "engine-3.0" in "\n".join(exc_info.value.evidence)
