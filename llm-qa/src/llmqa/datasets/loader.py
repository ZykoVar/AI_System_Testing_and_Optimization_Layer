"""测试数据集加载器：按名称加载 datasets/ 下的 YAML / JSON / CSV。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class DatasetNotFound(RuntimeError):
    """按名称查找数据集但未找到任何受支持扩展名的文件。"""


class DatasetManager:
    """按名称加载 datasets/ 目录下的 YAML / JSON / CSV 文件。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._used: dict[str, str] = {}   # 运行溯源：name → 内容指纹（sha256）
        self._hash_cache: dict[Path, str] = {}

    def path(self, name: str) -> Path:
        """按固定扩展名顺序把 name 解析到实际文件，找不到即抛 DatasetNotFound。"""
        # 扩展名即优先级：同名文件存在多种扩展时，YAML 优先于 JSON/CSV。
        for ext in (".yaml", ".yml", ".json", ".csv"):
            candidate = self.root / (name + ext)
            if candidate.exists():
                return candidate
        raise DatasetNotFound(f"数据集不存在: {name}（查找于 {self.root}）")

    def list(self) -> list[str]:
        """列出全部数据集名（相对路径去掉扩展名），可直接用作 load 的入参。"""
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.suffix in (".yaml", ".yml", ".json", ".csv"):
                rel = p.relative_to(self.root).as_posix()
                # 去掉扩展名，保证 list() 输出的名字能被 load()/path() 直接解析。
                out.append(rel[: -len(p.suffix)])
        return out

    def used_datasets(self) -> list[dict]:
        """本次运行实际加载过的数据集及其内容指纹（按名排序，运行溯源用）。"""
        return [{"name": name, "content_hash": self._used[name]}
                for name in sorted(self._used)]

    def _content_hash(self, path: Path) -> str:
        """数据集文件字节级 sha256：version 之外的防篡改指纹。"""
        if path not in self._hash_cache:
            self._hash_cache[path] = hashlib.sha256(
                path.read_bytes()).hexdigest()[:16]
        return self._hash_cache[path]

    def load(self, name: str) -> Any:
        """按扩展名解析并返回数据：CSV → 字典列表，YAML/JSON → 反序列化对象。"""
        path = self.path(name)
        self._used[name] = self._content_hash(path)   # 记录使用与指纹（运行溯源用）
        if path.suffix == ".csv":
            # utf-8-sig 兼容 Excel 导出的带 BOM 头；newline="" 交给 csv 模块统一处理换行。
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(text)
        return json.loads(text)
