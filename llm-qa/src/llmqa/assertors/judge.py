"""LLM-as-Judge：用独立裁判模型对答案做软性质量评分。

适用场景：正确性、忠实性、相关性等无法用确定性断言覆盖的维度。
生产建议：
1. 裁判模型与被测模型解耦（用更强的模型当裁判）；
2. 裁判 Prompt 走版本化托管模板（prompts/judge/correctness.yaml，传 prompt_manager 启用）；
3. 关键判定开启多次投票（passes>1），降低单次打分的随机噪声；
4. 成熟评测引擎（Ragas/Braintrust/DeepEval 指标）可通过 JudgeBackend 协议接入
   （见 llmqa/ext/），投票/证据/门禁等判定语义仍归 Judge 持有。
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field

from llmqa.assertors.base import AssertionFailed
from llmqa.clients.base import LLMClient, Message

DEFAULT_JUDGE_PROMPT = """你是一名严格、公正的评测裁判。请依据【评分标准】为【待评答案】打分。

要求：
1. 只依据评分标准与参考资料判断，不要被答案的语气、长度影响；
2. 满分为 {scale} 分，分数为 0-{scale} 的整数或一位小数；
3. 只输出一行 JSON：{{"score": <分数>, "reasoning": "<中文理由，不超过100字>"}}"""

# 多次投票时的温度序列：0 保证有基准分，渐增引入多样性
_PASS_TEMPERATURES = (0.0, 0.4, 0.8)


class JudgeVerdict(BaseModel):
    """裁判对单个答案的评分结论（多次投票时为聚合结果）。"""
    score: float                  # 0~scale 的评分；投票时为中位数
    reasoning: str = ""           # 中文评分理由（对应中位数的这一次）
    passes: int = 1               # 实际投票次数
    scores: list[float] = Field(default_factory=list)   # 每次投票的原始分（升序）
    agreement: float | None = None  # 各次打分一致度：1 - (max-min)/scale，1 表示完全一致


class JudgeBackend(Protocol):
    """可插拔裁判后端协议。

    成熟评测引擎（Ragas/Braintrust 指标等）实现 score() 后即可接入：
    投票聚合、证据传播、min_score 门禁由 Judge 统一持有。
    实现要求：返回 JudgeVerdict（reasoning 会随失败证据传播）。
    """

    async def score(self, *, question: str, answer: str, context: str,
                    criteria: str, scale: int, temperature: float = 0.0) -> JudgeVerdict:
        ...


class _BuiltinJudgeBackend:
    """内置后端：独立 LLM 打分（默认实现，离线可用）。"""

    def __init__(self, client: LLMClient, prompt: str, prompt_manager, prompt_id: str):
        self.client = client
        self.prompt = prompt
        self.prompt_manager = prompt_manager
        self.prompt_id = prompt_id

    def build_messages(self, *, question: str, answer: str, context: str,
                       criteria: str, scale: int) -> list[Message]:
        if self.prompt_manager is not None:
            # 托管模板：judge/correctness.yaml，含 question/context/criteria/answer/scale 变量
            return self.prompt_manager.render(
                self.prompt_id,
                {"question": question, "context": context or "",
                 "criteria": criteria, "answer": answer, "scale": scale})
        system = self.prompt.format(scale=scale)
        user_parts = ["【问题】", question]
        if context:
            user_parts += ["\n【参考资料】", context]
        user_parts += ["\n【评分标准】", criteria, "\n【待评答案】", answer]
        return [Message.system(system), Message.user("\n".join(user_parts))]

    async def score(self, *, question: str, answer: str, context: str,
                    criteria: str, scale: int, temperature: float = 0.0) -> JudgeVerdict:
        resp = await self.client.generate(
            self.build_messages(question=question, answer=answer, context=context,
                                criteria=criteria, scale=scale),
            temperature=temperature, max_tokens=300)
        return _parse_score(resp.text, scale)


class Judge:
    """基于独立 LLM 的裁判评分器；后端可插拔（默认内置，见 llmqa.ext）。"""

    def __init__(self, client: LLMClient, *, scale: int = 10,
                 prompt: str | None = None, passes: int = 1,
                 prompt_manager=None, prompt_id: str = "judge/correctness",
                 backend: JudgeBackend | None = None):
        self.client = client
        self.scale = scale
        # 自定义 Prompt 必须保留 {scale} 占位符供 format 填充
        self.prompt = prompt or DEFAULT_JUDGE_PROMPT
        self.passes = max(1, passes)
        self.prompt_manager = prompt_manager
        self.prompt_id = prompt_id
        # 后端可插拔：默认内置 LLM 打分；外部引擎实现 JudgeBackend 协议后接入
        self.backend = backend or _BuiltinJudgeBackend(
            client, self.prompt, prompt_manager, prompt_id)

    async def score(self, *, question: str, answer: str, context: str = "",
                    criteria: str, passes: int | None = None) -> JudgeVerdict:
        n = passes if passes is not None else self.passes
        if n <= 1:
            verdict = await self.backend.score(
                question=question, answer=answer, context=context,
                criteria=criteria, scale=self.scale, temperature=0.0)
            verdict.passes = 1
            verdict.scores = [verdict.score]
            verdict.agreement = 1.0
            return verdict
        # 多次投票：温度渐增引入多样性，取中位数抑制离群打分
        verdicts: list[JudgeVerdict] = []
        for i in range(n):
            temperature = _PASS_TEMPERATURES[i % len(_PASS_TEMPERATURES)]
            verdicts.append(await self.backend.score(
                question=question, answer=answer, context=context,
                criteria=criteria, scale=self.scale, temperature=temperature))
        ordered = sorted(v.score for v in verdicts)
        median = ordered[len(ordered) // 2]
        # 取分数等于中位数的第一次投票的理由，保证 reasoning 与 score 对应
        chosen = next(v for v in verdicts if v.score == median)
        return JudgeVerdict(
            score=median, reasoning=chosen.reasoning, passes=n,
            scores=ordered,
            agreement=round(1 - (ordered[-1] - ordered[0]) / self.scale, 3))

    async def assert_score(self, *, question: str, answer: str, context: str = "",
                           criteria: str, min_score: float | None = None,
                           passes: int | None = None,
                           message: str | None = None) -> JudgeVerdict:
        verdict = await self.score(question=question, answer=answer, context=context,
                                   criteria=criteria, passes=passes)
        threshold = min_score if min_score is not None else 7.0  # 满分 10 时的默认及格线
        if verdict.score < threshold:
            # 证据随失败一起传播：裁判理由、各次投票分与一致度写入 evidence，
            # 排障时无需重跑即可看到"为什么低分"
            evidence = ["裁判理由: " + verdict.reasoning]
            if verdict.passes > 1:
                evidence.append("各次投票分: " + str(verdict.scores))
                evidence.append("一致度: " + str(verdict.agreement))
            raise AssertionFailed(
                message or f"裁判评分 {verdict.score:.1f} 低于阈值 {threshold:.1f}：{verdict.reasoning}",
                metrics={"judge_score": verdict.score,
                         "judge_passes": verdict.passes,
                         "judge_agreement": verdict.agreement},
                evidence=evidence)
        return verdict


def _parse_score(text: str, scale: int) -> JudgeVerdict:
    """解析裁判输出：先剥围栏取 JSON，失败则回退到正则抓取 score 字段。"""
    fence = chr(96) * 3
    cleaned = text.strip()
    cleaned = re.sub("^" + fence + "(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(fence + "$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            score = float(obj.get("score", -1))
            if 0 <= score <= scale:  # 越界分数视为非法，回退到正则兜底
                return JudgeVerdict(score=score, reasoning=str(obj.get("reasoning", "")))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # 兜底：模型可能输出"score: 8"这类非 JSON 文本，直接抓数字字段
    m = re.search(r"(?i)score[\"']?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return JudgeVerdict(score=float(m.group(1)), reasoning=text[:200])
    raise AssertionFailed("裁判输出无法解析为 JSON 评分", evidence=[text[:500]])
