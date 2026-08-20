"""LLM-as-Judge：用独立裁判模型对答案做软性质量评分。

适用场景：正确性、忠实性、相关性等无法用确定性断言覆盖的维度。
生产建议：
1. 裁判模型与被测模型解耦（用更强的模型当裁判）；
2. 裁判 Prompt 走版本化托管模板（prompts/judge/correctness.yaml，传 prompt_manager 启用）；
3. 关键判定开启多次投票（passes>1），降低单次打分的随机噪声。
"""
from __future__ import annotations

import json
import re

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


class Judge:
    """基于独立 LLM 的裁判评分器。"""

    def __init__(self, client: LLMClient, *, scale: int = 10,
                 prompt: str | None = None, passes: int = 1,
                 prompt_manager=None, prompt_id: str = "judge/correctness"):
        self.client = client
        self.scale = scale
        # 自定义 Prompt 必须保留 {scale} 占位符供 format 填充
        self.prompt = prompt or DEFAULT_JUDGE_PROMPT
        self.passes = max(1, passes)
        # 传入 PromptManager 时走版本化托管模板（企业推荐），否则用内置 Prompt
        self.prompt_manager = prompt_manager
        self.prompt_id = prompt_id

    def _build_messages(self, *, question: str, answer: str, context: str,
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

    async def score(self, *, question: str, answer: str, context: str = "",
                    criteria: str, passes: int | None = None) -> JudgeVerdict:
        n = passes if passes is not None else self.passes
        if n <= 1:
            resp = await self.client.generate(
                self._build_messages(question=question, answer=answer, context=context,
                                     criteria=criteria, scale=self.scale),
                temperature=0.0, max_tokens=300)
            verdict = self._parse(resp.text)
            verdict.passes = 1
            verdict.scores = [verdict.score]
            verdict.agreement = 1.0
            return verdict
        # 多次投票：温度渐增引入多样性，取中位数抑制离群打分
        verdicts: list[JudgeVerdict] = []
        for i in range(n):
            temperature = _PASS_TEMPERATURES[i % len(_PASS_TEMPERATURES)]
            resp = await self.client.generate(
                self._build_messages(question=question, answer=answer, context=context,
                                     criteria=criteria, scale=self.scale),
                temperature=temperature, max_tokens=300)
            verdicts.append(self._parse(resp.text))
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
            raise AssertionFailed(
                message or f"裁判评分 {verdict.score:.1f} 低于阈值 {threshold:.1f}：{verdict.reasoning}",
                metrics={"judge_score": verdict.score})
        return verdict

    def _parse(self, text: str) -> JudgeVerdict:
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
                if 0 <= score <= self.scale:  # 越界分数视为非法，回退到正则兜底
                    return JudgeVerdict(score=score, reasoning=str(obj.get("reasoning", "")))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # 兜底：模型可能输出"score: 8"这类非 JSON 文本，直接抓数字字段
        m = re.search(r"(?i)score[\"']?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            return JudgeVerdict(score=float(m.group(1)), reasoning=text[:200])
        raise AssertionFailed("裁判输出无法解析为 JSON 评分", evidence=[text[:500]])
