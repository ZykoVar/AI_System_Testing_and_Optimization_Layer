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
        return _SEVERITY_RANK[self]

    def __ge__(self, other: "Severity") -> bool:
        return self.rank >= other.rank


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
    traceback: str | None = None


class TestContext(BaseModel):
    """注入到每个测试用例的执行上下文。

    注意：LLM 调用必须通过 ctx.client() 获取（统一走连接池），
    以保证 mock/真实 Provider 可无缝切换、用量可被统一统计。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    settings: Any
    providers: Any          # ClientPool
    prompts: Any            # PromptManager
    datasets: Any           # DatasetManager

    def client(self, name: str | None = None):
        return self.providers.get(name)
