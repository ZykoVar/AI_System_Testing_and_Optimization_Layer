"""Prompt 管理：版本化模板库、渲染校验、版本 diff 与状态流转。

模板文件约定（prompts/<路径>.yaml）：
    id: support-agent/v1          # 可选，默认取相对路径
    name: 客服助手                 # 展示名
    version: 1                    # 版本号
    status: active                # draft | active | deprecated
    tags: [customer-service]
    messages:                     # 消息模板列表
      - role: system
        content: |
          你是 {{company}} 的客服助手。
    variables:                    # 变量声明（未声明变量不得出现在模板中）
      company: {type: string, required: true, description: 公司名}
    changelog:
      - {version: 1, date: '2025-01-10', note: 初始版本}

模板使用 {{ var }} 占位符；渲染时校验必填/未声明变量。
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmqa.clients.base import Message

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")


class PromptError(RuntimeError):
    pass


class PromptNotFound(PromptError):
    pass


class PromptRenderError(PromptError):
    pass


class VariableSpec(BaseModel):
    type: str = "string"
    required: bool = True
    description: str = ""
    default: Any = None


class PromptMessage(BaseModel):
    role: str = "system"
    content: str = ""


class PromptTemplate(BaseModel):
    """单版本 Prompt 模板。"""
    id: str
    name: str = ""
    version: int = 1
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    messages: list[PromptMessage] = Field(default_factory=list)
    variables: dict[str, VariableSpec] = Field(default_factory=dict)
    changelog: list[dict[str, Any]] = Field(default_factory=list)
    scan_ignore: list[str] = Field(default_factory=list)   # 豁免的扫描规则名（如护栏文本合法提及"系统提示词"）
    path: Path | None = None

    @property
    def used_variables(self) -> set[str]:
        used: set[str] = set()
        for m in self.messages:
            used.update(_VAR_RE.findall(m.content))
        return used

    def full_text(self) -> str:
        return "\n\n".join("[{}]\n{}".format(m.role, m.content) for m in self.messages)


class PromptManager:
    """加载、查询、渲染、diff 与状态管理的 Prompt 仓库。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._templates: dict[str, dict[int, PromptTemplate]] = {}

    # ---------- 加载与查询 ----------
    def load(self) -> "PromptManager":
        self._templates.clear()
        for path in sorted(self.root.rglob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            rel = path.relative_to(self.root).as_posix()[:-5]  # 去 .yaml
            template = PromptTemplate(
                id=data.get("id", rel),
                name=data.get("name", ""),
                version=int(data.get("version", 1)),
                status=data.get("status", "active"),
                tags=data.get("tags", []),
                messages=[PromptMessage(**m) for m in data.get("messages", [])],
                variables={k: VariableSpec(**v) if isinstance(v, dict) else VariableSpec()
                           for k, v in (data.get("variables") or {}).items()},
                changelog=data.get("changelog", []),
                scan_ignore=data.get("scan_ignore", []),
                path=path,
            )
            self._templates.setdefault(template.id, {})[template.version] = template
        return self

    def list(self, *, name: str | None = None, status: str | None = None) -> list[PromptTemplate]:
        out: list[PromptTemplate] = []
        for versions in self._templates.values():
            for t in versions.values():
                if name and name not in t.id:
                    continue
                if status and t.status != status:
                    continue
                out.append(t)
        return sorted(out, key=lambda t: (t.id, t.version))

    def get(self, prompt_id: str, version: int | None = None) -> PromptTemplate:
        versions = self._templates.get(prompt_id)
        if not versions:
            raise PromptNotFound("Prompt 不存在: {}".format(prompt_id))
        if version is None:
            version = max(versions)
        if version not in versions:
            raise PromptNotFound("Prompt {} 无版本 v{}（现有: {}）".format(
                prompt_id, version, sorted(versions)))
        return versions[version]

    # ---------- 渲染 ----------
    def render(self, prompt_id: str, variables: dict[str, Any] | None = None,
               version: int | None = None, *, strict: bool = True) -> list[Message]:
        """渲染为客户端 Message 列表；校验必填变量与未声明变量。"""
        template = self.get(prompt_id, version)
        variables = dict(variables or {})
        # 1. 必填校验
        missing = [k for k, spec in template.variables.items()
                   if spec.required and k not in variables and spec.default is None]
        if missing:
            raise PromptRenderError("Prompt {} 缺少必填变量: {}".format(prompt_id, missing))
        # 2. 模板中使用了未声明变量 → 严格模式下报错
        undeclared = template.used_variables - set(template.variables)
        if strict and undeclared:
            raise PromptRenderError("Prompt {} 模板使用了未声明变量: {}".format(prompt_id, sorted(undeclared)))
        # 3. 应用默认值
        for k, spec in template.variables.items():
            if k not in variables and spec.default is not None:
                variables[k] = spec.default
        rendered: list[Message] = []
        for m in template.messages:
            content = _VAR_RE.sub(
                lambda match: _render_value(variables, match.group(1), prompt_id), m.content)
            rendered.append(Message(role=m.role, content=content))
        return rendered

    # ---------- 校验与 diff ----------
    def validate(self) -> list[str]:
        """全库校验，返回问题列表。"""
        problems: list[str] = []
        for t in self.list():
            used, declared = t.used_variables, set(t.variables)
            if used - declared:
                problems.append("{}: 未声明变量 {}".format(t.id, sorted(used - declared)))
            for k, spec in t.variables.items():
                if k not in used and not spec.description:
                    problems.append("{}: 声明了但未使用的变量 {}".format(t.id, k))
            if not t.messages:
                problems.append("{}: 无消息模板".format(t.id))
        return problems

    def diff(self, prompt_id: str, v1: int, v2: int) -> str:
        a = self.get(prompt_id, v1)
        b = self.get(prompt_id, v2)
        return "".join(difflib.unified_diff(
            a.full_text().splitlines(keepends=True),
            b.full_text().splitlines(keepends=True),
            fromfile="{} v{}".format(prompt_id, v1),
            tofile="{} v{}".format(prompt_id, v2)))

    # ---------- 状态流转 ----------
    def promote(self, prompt_id: str, status: str, version: int | None = None) -> PromptTemplate:
        """状态流转（draft → active → deprecated），直接写回 YAML 文件。"""
        allowed = {"draft", "active", "deprecated"}
        if status not in allowed:
            raise PromptError("非法状态 {}，允许: {}".format(status, sorted(allowed)))
        template = self.get(prompt_id, version)
        if template.path is None:
            raise PromptError("{} 无源文件，无法流转".format(prompt_id))
        data = yaml.safe_load(template.path.read_text(encoding="utf-8")) or {}
        data["status"] = status
        template.path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                 encoding="utf-8")
        template.status = status
        return template


def _render_value(variables: dict[str, Any], key: str, prompt_id: str) -> str:
    if key not in variables:
        raise PromptRenderError("Prompt {} 渲染时缺少变量: {}".format(prompt_id, key))
    value = variables[key]
    if value is None:
        raise PromptRenderError("Prompt {} 变量 {} 为 None".format(prompt_id, key))
    return str(value)
