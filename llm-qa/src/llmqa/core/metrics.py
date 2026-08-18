"""通用统计工具：分位数与延迟统计。"""
from __future__ import annotations

from typing import Sequence


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """nearest-rank 分位数；输入必须已升序排序。"""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, int(round(p / 100.0 * n + 0.5))))
    return float(sorted_values[rank - 1])


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """延迟（毫秒）统计：count/min/max/mean/p50/p90/p95/p99。"""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "min_ms": min(ordered),
        "max_ms": max(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(ordered, 50),
        "p90_ms": percentile(ordered, 90),
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
    }
