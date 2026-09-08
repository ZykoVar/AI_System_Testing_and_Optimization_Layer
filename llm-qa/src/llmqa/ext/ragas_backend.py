"""Ragas 裁判后端：把 Ragas 评估指标（faithfulness 等）接到 JudgeBackend 协议。

激活：pip install -e ".[ext]"（ragas 依赖真实 LLM/嵌入模型）。
未安装 ragas 时构造抛 LLMError（附安装指引），不影响离线 CI。

注意：本适配器是"协议绑定骨架"——Ragas 版本间 API 有差异，
生产接入时按所用 ragas 版本补全 evaluate 调用细节；判定语义
（投票/证据/min_score 门禁）仍由本项目 Judge 持有。
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
    """Ragas 指标作为 Judge 后端：score() 返回 JudgeVerdict（0-1 指标 × scale 归一）。

    用法::

        judge = Judge(client, backend=RagasJudgeBackend(client, metric="faithfulness"),
                      scale=10)
        await judge.assert_score(question=..., answer=..., context=..., criteria=...)
    """

    def __init__(self, client, *, metric: str = "faithfulness", scale: int = 10):
        self._ragas = _import_ragas()
        self.client = client
        self.metric = metric
        self.scale = scale

    async def score(self, *, question: str, answer: str, context: str,
                    criteria: str, scale: int, temperature: float = 0.0) -> JudgeVerdict:
        ragas = self._ragas
        # Ragas 单轮评估样本：问题 + 回答 + 检索上下文
        sample = ragas.SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[context] if context else [],
        )
        metric = ragas.metrics.Faithfulness()
        # 按所用 ragas 版本补全 evaluate 绑定（LLM 包装器/嵌入配置因版本而异）：
        # result = await ragas.evaluate(Dataset([sample]), metrics=[metric], llm=<wrapper>)
        # score = result.to_pandas().iloc[0]["faithfulness"]
        # 归一化到 Judge 口径后返回：
        raise NotImplementedError(
            "RagasJudgeBackend 是协议绑定骨架：请按当前 ragas 版本补全 evaluate 调用，"
            "并将 0-1 指标分 × scale 归一为 JudgeVerdict（见模块 docstring）。")
