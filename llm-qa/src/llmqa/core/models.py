"""测试核心数据模型：严重级别、判定、用例上下文与结果。"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """失败严重级别，用于过滤（--severity HIGH 只跑 HIGH+）与报告排序。"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """数值等级，用于排序与"不低于某级别"的过滤比较。"""
        return _SEVERITY_RANK[self]

    def __ge__(self, other: Severity) -> bool:
        # 重载比较运算符，使 self.severity >= min_severity 这类过滤写法可读且直观。
        return self.rank >= other.rank


# 级别到数值的映射：数值越大越严重，是排序与阈值过滤的统一基准。
_SEVERITY_RANK = {
    Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
    Severity.HIGH: 3, Severity.CRITICAL: 4,
}


class Verdict(str, Enum):
    PASS = "PASS"       # 断言全部通过
    FAIL = "FAIL"       # 断言失败（被测对象不满足要求）
    ERROR = "ERROR"     # 基础设施错误（网络/超时/未配置 Provider 等）
    SKIP = "SKIP"       # 环境不满足，显式跳过


class TestOutcome(BaseModel):
    """单个用例的最终结果。"""
    case_id: str
    name: str
    suite: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    verdict: Verdict
    duration_ms: float = 0.0
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)   # 数值指标（延迟、评分、命中率等）
    evidence: list[str] = Field(default_factory=list)       # 证据文本/引用
    traceback: str | None = None  # 仅在 ERROR（基础设施异常）时填充，便于报告定位
    retries_used: int = 0  # 本次判定前因 ERROR 实际重试的次数（0=一次通过），用于 flaky 可见性
    skip_reason: str = ""  # SKIP 语义分类：intentional | budget | fail_fast（""=非 SKIP）
                           # 回归对比时 budget/fail_fast 视为中性，避免误报回归


class TestContext(BaseModel):
    """注入到每个测试用例的执行上下文。

    注意：LLM 调用必须通过 ctx.client() 获取（统一走连接池），
    以保证 mock/真实 Provider 可无缝切换、用量可被统一统计。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str             # 本次运行的唯一标识，写入报告与输出目录
    settings: Any           # 全局配置对象（Provider、超时、报告等运行时参数）
    providers: Any          # ClientPool
    prompts: Any            # PromptManager
    datasets: Any           # DatasetManager
    record_metrics: dict = Field(default_factory=dict)    # 用例主动记录的判定数据（PASS 也落盘）
    record_evidence: list = Field(default_factory=list)   # 用例主动记录的证据（PASS 也落盘）

    def record(self, **metrics) -> None:
        """记录判定数据（judge 分数/相似度/延迟分位等）——PASS 用例的信号来源。

        用法：verdict = await judge.assert_score(...); ctx.record(judge_score=verdict.score)
        """
        self.record_metrics.update(metrics)

    def add_evidence(self, text: str) -> None:
        """记录证据文本（裁判理由/关键片段等），PASS 用例同样可见。"""
        self.record_evidence.append(text)

    def client(self, name: str | None = None):
        """按名称从连接池获取客户端；不传名则取默认 Provider。"""
        return self.providers.get(name)
