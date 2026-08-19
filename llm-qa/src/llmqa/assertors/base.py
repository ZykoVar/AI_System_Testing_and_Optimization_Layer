"""断言基础：AssertionFailed 异常。"""
from __future__ import annotations

from typing import Any


class AssertionFailed(AssertionError):
    """用例断言失败信号。运行器捕获后标记为 FAIL。

    metrics: 附带数值指标（如实际相似度、裁判打分），会写入报告。
    继承 AssertionError 而非普通 Exception，使 except AssertionError 也能捕获。
    """

    def __init__(self, message: str, *, metrics: dict[str, Any] | None = None,
                 evidence: list[str] | None = None):
        super().__init__(message)
        self.metrics = metrics or {}    # 数值指标，会被写入报告的 metrics 字段
        self.evidence = evidence or []  # 证据文本/引用，用于报告定位问题
