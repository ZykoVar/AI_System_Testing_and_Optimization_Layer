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

# 模板占位符语法 {{ var }}：变量名支持点号路径（如 user.name），允许两侧空白。
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")


class PromptError(RuntimeError):
    """Prompt 子系统的统一异常基类，便于调用方一次性捕获。"""


class PromptNotFound(PromptError):
    """请求的 Prompt id 或版本不存在。"""


class PromptRenderError(PromptError):
    """渲染阶段失败：缺少必填变量、使用了未声明变量或变量值为 None。"""


class VariableSpec(BaseModel):
    """单个模板变量的声明：类型、是否必填、说明与默认值。"""

    type: str = "string"
    required: bool = True
    description: str = ""
    default: Any = None    # 默认值；None 表示无默认（必填变量不得缺省）


class PromptMessage(BaseModel):
    """一条消息模板：content 可含 {{ var }} 占位符，渲染时被替换。"""

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
    path: Path | None = None    # 源 YAML 路径；内存构造时为 None（无法回写 promote）

    @property
    def used_variables(self) -> set[str]:
        """模板消息中实际引用的变量集合（从占位符提取，用于未声明校验）。"""
        used: set[str] = set()
        for m in self.messages:
            used.update(_VAR_RE.findall(m.content))
        return used

    def full_text(self) -> str:
        """把全部消息拼成纯文本（[role] 前缀 + content），作为 diff 的输入。"""
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in self.messages)


class PromptManager:
    """加载、查询、渲染、diff 与状态管理的 Prompt 仓库。"""

    def __init__(self, root: str | Path):
        """root 为 prompts 目录；_templates 按 id→version 二级索引，_pins 记录全局钉住版本。"""
        self.root = Path(root)
        self._templates: dict[str, dict[int, PromptTemplate]] = {}
        self._pins: dict[str, int] = {}

    # ---------- 全局版本钉住（A/B 测试用） ----------
    def pin(self, prompt_id: str, version: int) -> None:
        """钉住某 Prompt 的默认渲染版本；render 未显式指定 version 时生效。"""
        self._pins[prompt_id] = version

    def unpin(self, prompt_id: str | None = None) -> None:
        """解除钉住；prompt_id 为 None 时清空全部。"""
        if prompt_id is None:
            self._pins.clear()
        else:
            self._pins.pop(prompt_id, None)

    # ---------- 加载与查询 ----------
    def load(self) -> PromptManager:
        """扫描 root 下全部 .yaml 重建索引；同一 id 可并存多个 version（二级键为版本号）。"""
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
            # 以 id 为一级键、version 为二级键，同一 id 的多个版本自然聚拢。
            self._templates.setdefault(template.id, {})[template.version] = template
        return self

    def list(self, *, name: str | None = None, status: str | None = None) -> list[PromptTemplate]:
        """列出过滤后的模板：name 做 id 子串匹配，status 精确匹配，结果按 (id, version) 排序。"""
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
        """按 id（可选 version）取模板；version 为 None 时取最新 active，无 active 回退最大版本号。"""
        versions = self._templates.get(prompt_id)
        if not versions:
            raise PromptNotFound(f"Prompt 不存在: {prompt_id}")
        if version is None:
            # 默认语义：取最新 active 版本；无 active 时回退到最大版本号
            active = [v for v, t in versions.items() if t.status == "active"]
            version = max(active) if active else max(versions)
        if version not in versions:
            raise PromptNotFound(f"Prompt {prompt_id} 无版本 v{version}（现有: {sorted(versions)}）")
        return versions[version]

    # ---------- 渲染 ----------
    def render(self, prompt_id: str, variables: dict[str, Any] | None = None,
               version: int | None = None, *, strict: bool = True) -> list[Message]:
        """渲染为客户端 Message 列表；校验必填变量与未声明变量。

        版本解析优先级：显式 version 参数 > pin() 全局钉住 > 最新 active。
        """
        if version is None:
            version = self._pins.get(prompt_id)
        template = self.get(prompt_id, version)
        variables = dict(variables or {})   # 拷贝一份，避免渲染过程污染调用方的字典
        # 1. 必填校验
        missing = [k for k, spec in template.variables.items()
                   if spec.required and k not in variables and spec.default is None]
        if missing:
            raise PromptRenderError(f"Prompt {prompt_id} 缺少必填变量: {missing}")
        # 2. 模板中使用了未声明变量 → 严格模式下报错
        undeclared = template.used_variables - set(template.variables)
        if strict and undeclared:
            raise PromptRenderError(f"Prompt {prompt_id} 模板使用了未声明变量: {sorted(undeclared)}")
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
                problems.append(f"{t.id}: 未声明变量 {sorted(used - declared)}")
            for k, spec in t.variables.items():
                if k not in used and not spec.description:
                    problems.append(f"{t.id}: 声明了但未使用的变量 {k}")
            if not t.messages:
                problems.append(f"{t.id}: 无消息模板")
        return problems

    def diff(self, prompt_id: str, v1: int, v2: int) -> str:
        """返回 v1→v2 的统一 diff（基于 full_text 展开，面向人工审阅而非机器解析）。"""
        a = self.get(prompt_id, v1)
        b = self.get(prompt_id, v2)
        return "".join(difflib.unified_diff(
            a.full_text().splitlines(keepends=True),
            b.full_text().splitlines(keepends=True),
            fromfile=f"{prompt_id} v{v1}",
            tofile=f"{prompt_id} v{v2}"))

    # ---------- 状态流转 ----------
    def promote(self, prompt_id: str, status: str, version: int | None = None) -> PromptTemplate:
        """状态流转（draft → active → deprecated），直接写回 YAML 文件。"""
        allowed = {"draft", "active", "deprecated"}
        if status not in allowed:
            raise PromptError(f"非法状态 {status}，允许: {sorted(allowed)}")
        template = self.get(prompt_id, version)
        if template.path is None:
            raise PromptError(f"{prompt_id} 无源文件，无法流转")
        data = yaml.safe_load(template.path.read_text(encoding="utf-8")) or {}
        data["status"] = status
        # 重新读盘只改 status 再整体写回：避免用内存模型序列化而丢失 YAML 中的其他字段与手工排版。
        template.path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                 encoding="utf-8")
        template.status = status
        return template


def _render_value(variables: dict[str, Any], key: str, prompt_id: str) -> str:
    """取变量值并转字符串；缺失或为 None 抛 PromptRenderError，不允许静默输出空串。"""
    if key not in variables:
        raise PromptRenderError(f"Prompt {prompt_id} 渲染时缺少变量: {key}")
    value = variables[key]
    if value is None:
        raise PromptRenderError(f"Prompt {prompt_id} 变量 {key} 为 None")
    return str(value)
