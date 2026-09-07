"""指标漂移判定策略：把 compare 从"字面 diff"升级为"评价引擎"。

策略文件：config/metrics_policy.yaml
- direction: higher_better / lower_better / neutral（neutral 仅记录）
- tolerance: 绝对值，或相对百分比（字符串以 % 结尾，按基线值计算）

未声明策略的指标回退到"任何变化都记录为 drift（不计回归）"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_policies(root: Path) -> dict[str, dict]:
    """加载 metrics_policy.yaml；文件缺失时返回空策略（全部按 drift 处理）。"""
    path = root / "config" / "metrics_policy.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("metrics") or {}


def _tolerance_value(policy: dict, baseline: float) -> float:
    """解析容差：数字为绝对值；'20%' 类字符串按基线值的百分比计算。"""
    tol = policy.get("tolerance", 0)
    if isinstance(tol, str) and tol.endswith("%"):
        return abs(baseline) * float(tol[:-1]) / 100.0
    return float(tol)


def classify_metric_delta(name: str, a: float, b: float,
                          policies: dict[str, dict] | None) -> str | None:
    """判定指标变化的方向，返回 None 表示容差内不显著。

    返回：regression | improvement | drift（drift=仅记录，不计回归/改善）。
    """
    policy = (policies or {}).get(name)
    delta = b - a
    if policy is None:
        return "drift"
    tolerance = _tolerance_value(policy, a)
    if abs(delta) <= tolerance:
        return None   # 容差内：不显著
    direction = policy.get("direction", "neutral")
    if direction == "higher_better":
        return "improvement" if delta > 0 else "regression"
    if direction == "lower_better":
        return "regression" if delta > 0 else "improvement"
    return "drift"
