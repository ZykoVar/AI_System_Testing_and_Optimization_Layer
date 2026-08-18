"""测试数据集加载器：按名称加载 datasets/ 下的 YAML / JSON / CSV。"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import yaml


class DatasetNotFound(RuntimeError):
    pass


class DatasetManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        for ext in (".yaml", ".yml", ".json", ".csv"):
            candidate = self.root / (name + ext)
            if candidate.exists():
                return candidate
        raise DatasetNotFound("数据集不存在: {}（查找于 {}）".format(name, self.root))

    def list(self) -> list[str]:
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.suffix in (".yaml", ".yml", ".json", ".csv"):
                rel = p.relative_to(self.root).as_posix()
                out.append(rel[: -len(p.suffix)])
        return out

    def load(self, name: str) -> Any:
        path = self.path(name)
        if path.suffix == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(text)
        return json.loads(text)
