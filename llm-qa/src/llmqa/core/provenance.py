"""运行溯源（Provenance）：让每次运行可回答"用什么代码、什么数据集、
什么 Prompt、什么模型、什么用例代码，跑出了什么结果"。

组成（全部随 report.json 的 provenance 块落盘）：
- git 提交与工作区状态（实验代码基线）；
- Prompt 使用清单：id + version + content_hash（version 是命名，hash 防篡改）；
- 数据集清单：name + content_hash；
- 模型：provider/kind/model（重建实验条件的接入信息）；
- 用例身份：case_id → 源码指纹（Stable Test ID 的内容维度）。
"""
from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field


class PromptUsage(BaseModel):
    """一次运行中实际渲染使用的 Prompt 版本与内容指纹。"""
    id: str
    version: int
    content_hash: str = ""


class DatasetUsage(BaseModel):
    """一次运行中实际加载的数据集及其内容指纹。"""
    name: str
    content_hash: str = ""


class ModelUsage(BaseModel):
    """被测模型接入信息（重建实验条件的必要参数）。"""
    provider: str
    kind: str = ""
    model: str = ""


class RunProvenance(BaseModel):
    """一次运行的溯源信息。git/子进程失败时宽容降级为空值，不影响测试执行。"""
    git_commit: str = ""                 # 运行时代码的 git 提交（非 git 环境为空）
    git_dirty: bool | None = None        # 已跟踪文件是否有未提交修改（非 git 环境为 None）
    python_version: str = ""
    platform: str = ""
    timezone: str = ""
    model: ModelUsage | None = None
    prompts_used: list[PromptUsage] = Field(default_factory=list)
    datasets_used: list[DatasetUsage] = Field(default_factory=list)
    test_identity: dict[str, str] = Field(default_factory=dict)  # case_id → 源码指纹


def git_info(root: Path) -> tuple[str, bool | None]:
    """读取 git 提交与工作区脏状态；非 git 环境返回 ("", None)。"""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if commit.returncode != 0:
            return "", None
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10,
        )
        # 仅统计已跟踪文件的修改，忽略 .idea 等未跟踪噪声
        return commit.stdout.strip(), bool(dirty.stdout.strip())
    except Exception:  # noqa: BLE001 —— git 不可用时溯源降级，不阻断测试
        return "", None


def collect_provenance(root: Path, prompts=None, datasets=None, *,
                       settings=None, provider_name: str | None = None,
                       cases=None) -> RunProvenance:
    """聚合当前运行的溯源信息。"""
    commit, dirty = git_info(root)
    tz = dt.datetime.now().astimezone().tzname() or ""
    provenance = RunProvenance(
        git_commit=commit,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        timezone=tz,
    )
    if settings is not None:
        try:
            cfg = settings.provider(provider_name or settings.default_provider)
            provenance.model = ModelUsage(
                provider=cfg.name, kind=cfg.kind, model=cfg.model)
        except Exception:  # noqa: BLE001 —— 配置异常不阻断溯源
            pass
    if prompts is not None and hasattr(prompts, "used_prompts"):
        provenance.prompts_used = [PromptUsage(**u) for u in prompts.used_prompts()]
    if datasets is not None and hasattr(datasets, "used_datasets"):
        provenance.datasets_used = [DatasetUsage(**u) for u in datasets.used_datasets()]
    if cases is not None:
        provenance.test_identity = {c.id: c.source_hash for c in cases}
    return provenance


def attach_provenance(report, root: Path, prompts=None, datasets=None, *,
                      settings=None, provider_name: str | None = None,
                      cases=None) -> None:
    """把溯源信息挂到 TestReport（CLI/abtest/demo 在运行结束后调用）。"""
    report.provenance = collect_provenance(
        root, prompts, datasets, settings=settings,
        provider_name=provider_name, cases=cases)
