"""通用统计工具：分位数与延迟统计。"""
from __future__ import annotations

from typing import Sequence


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """nearest-rank 分位数；输入必须已升序排序。"""
    if not sorted_values:
        return 0.0  # 空序列约定返回 0，调用方无需特判空输入。
    n = len(sorted_values)
    # nearest-rank 公式：位置 = round(p/100 * n + 0.5)，再夹到 [1, n] 防止越界。
    rank = max(1, min(n, int(round(p / 100.0 * n + 0.5))))
    return float(sorted_values[rank - 1])


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """延迟（毫秒）统计：count/min/max/mean/p50/p90/p95/p99。"""
    if not values:
        return {"count": 0}  # 无样本时只返回 count=0，其余字段缺省以表示"无数据"。
    ordered = sorted(values)  # 分位数要求升序，此处排序一次供后续多次取值复用。
    return {
        # 统一以浮点输出，单位毫秒（ms），便于 JSON 序列化与图表直接消费。
        "count": float(len(ordered)),
        "min_ms": min(ordered),
        "max_ms": max(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(ordered, 50),
        "p90_ms": percentile(ordered, 90),
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
    }
