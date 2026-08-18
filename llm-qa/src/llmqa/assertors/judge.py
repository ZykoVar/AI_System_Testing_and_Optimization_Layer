"""LLM-as-Judge：用独立裁判模型对答案做软性质量评分。

适用场景：正确性、忠实性、相关性等无法用确定性断言覆盖的维度。
生产建议：裁判模型与被测模型解耦（用更强的模型当裁判），并固定裁判 Prompt 版本。
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from llmqa.assertors.base import AssertionFailed
from llmqa.clients.base import LLMClient, Message

DEFAULT_JUDGE_PROMPT = """你是一名严格、公正的评测裁判。请依据【评分标准】为【待评答案】打分。

要求：
1. 只依据评分标准与参考资料判断，不要被答案的语气、长度影响；
2. 满分为 {scale} 分，分数为 0-{scale} 的整数或一位小数；
3. 只输出一行 JSON：{{"score": <分数>, "reasoning": "<中文理由，不超过100字>"}}"""


class JudgeVerdict(BaseModel):
    score: float
    reasoning: str = ""


class Judge:
    """基于独立 LLM 的裁判评分器。"""

    def __init__(self, client: LLMClient, *, scale: int = 10,
                 prompt: str | None = None):
        self.client = client
        self.scale = scale
        self.prompt = prompt or DEFAULT_JUDGE_PROMPT

    async def score(self, *, question: str, answer: str, context: str = "",
                    criteria: str) -> JudgeVerdict:
        system = self.prompt.format(scale=self.scale)
        user_parts = ["【问题】", question]
        if context:
            user_parts += ["\n【参考资料】", context]
        user_parts += ["\n【评分标准】", criteria, "\n【待评答案】", answer]
        resp = await self.client.generate(
            [Message.system(system), Message.user("\n".join(user_parts))],
            temperature=0.0, max_tokens=300)
        return self._parse(resp.text)

    async def assert_score(self, *, question: str, answer: str, context: str = "",
                           criteria: str, min_score: float | None = None,
                           message: str | None = None) -> JudgeVerdict:
        verdict = await self.score(question=question, answer=answer,
                                   context=context, criteria=criteria)
        threshold = min_score if min_score is not None else 7.0
        if verdict.score < threshold:
            raise AssertionFailed(
                message or "裁判评分 {:.1f} 低于阈值 {:.1f}：{}".format(
                    verdict.score, threshold, verdict.reasoning),
                metrics={"judge_score": verdict.score})
        return verdict

    def _parse(self, text: str) -> JudgeVerdict:
        fence = chr(96) * 3
        cleaned = text.strip()
        cleaned = re.sub("^" + fence + "(?:json)?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(fence + "$", "", cleaned, flags=re.MULTILINE).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end + 1])
                score = float(obj.get("score", -1))
                if 0 <= score <= self.scale:
                    return JudgeVerdict(score=score, reasoning=str(obj.get("reasoning", "")))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        m = re.search(r"(?i)score[\"']?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            return JudgeVerdict(score=float(m.group(1)), reasoning=text[:200])
        raise AssertionFailed("裁判输出无法解析为 JSON 评分", evidence=[text[:500]])
