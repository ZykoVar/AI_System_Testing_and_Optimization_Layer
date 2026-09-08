"""Ragas 接入骨架（integration scaffold / adapter contract）——不是 production-ready backend。

定位说明（刻意区分）：
- 本模块证明"llmqa 可以把成熟评测引擎插进来"，并提供 JudgeBackend 契约绑定；
- score() 的 ragas.evaluate 调用需要按**你锁定的 ragas 版本**补全（Ragas API
  跨版本有差异，绑定细节见下方参考实现），完成后 Ragas 指标即可作为裁判后端；
- 判定语义（投票/证据/min_score 门禁）由本项目 Judge 持有，不依赖 Ragas。

激活：pip install -e ".[ext]"；未安装 ragas 时构造抛 LLMError（附安装指引）。
"""
from __future__ import annotations

from llmqa.assertors.judge import JudgeVerdict
from llmqa.clients.base import LLMError


def _import_ragas():
    try:
        import ragas  # noqa: F401
        return ragas
    except ImportError as e:  # pragma: no cover —— 依赖缺失路径由单测覆盖
        raise LLMError(
            "ragas",
            '未安装 ragas，无法使用 RagasJudgeBackend。安装：pip install -e ".[ext]"',
        ) from e


class RagasJudgeBackend:
    """Ragas 指标作为 Judge 后端的契约实现（scaffold）。

    参考实现（ragas 0.2.x，接入时按你锁定的版本核对 API）::

        from ragas import evaluate, SingleTurnSample, metrics
        from ragas.llms import LangchainLLMWrapper

        class RagasJudgeBackend:
            async def score(self, *, question, answer, context, criteria,
                            scale, temperature=0.0):
                sample = SingleTurnSample(
                    user_input=question, response=answer,
                    retrieved_contexts=[context] if context else [])
                result = evaluate(
                    [sample],
                    metrics=[metrics.Faithfulness()],
                    llm=LangchainLLMWrapper(self.client),  # 或 ragas 对应版本包装器
                )
                raw = result.to_pandas().iloc[0]["faithfulness"]  # 0-1
                return JudgeVerdict(score=raw * scale,
                                    reasoning="Ragas faithfulness={:.2f}".format(raw))
    """

    def __init__(self, client, *, metric: str = "faithfulness", scale: int = 10):
        self._ragas = _import_ragas()
        self.client = client
        self.metric = metric
        self.scale = scale

    async def score(self, *, question: str, answer: str, context: str,
                    criteria: str, scale: int, temperature: float = 0.0) -> JudgeVerdict:
        # 骨架终止点：绑定 ragas.evaluate 后返回 JudgeVerdict 即完成接入
        # （参考实现见类 docstring；Ragas API 跨版本差异，此处不替用户锁定）
        raise NotImplementedError(
            "RagasJudgeBackend 是接入骨架（adapter contract）："
            "请按类 docstring 的参考实现补全 ragas.evaluate 绑定（约 10 行），"
            "完成后 Ragas 指标即以 JudgeBackend 身份参与投票/证据/门禁。")
