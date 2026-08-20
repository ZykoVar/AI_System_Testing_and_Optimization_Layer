"""配置模型与加载器。

加载约定：
- 仓库根通过向上查找包含 config/settings.yaml 的目录确定（fallback 到 cwd）。
- settings.yaml / providers.yaml 中的字符串支持 ${ENV_VAR:默认值} 环境变量展开。
- 密钥一律通过 api_key_env 从环境变量读取，不落盘。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# 匹配 ${ENV_VAR} 或 ${ENV_VAR:默认值}；默认值可为空、可包含除 } 外的任意字符。
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归展开字符串中的 ${ENV_VAR} / ${ENV_VAR:默认值}。"""
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            # 未定义且无默认值时回退为空串，避免展开后残留 ${VAR} 占位符。
            return os.environ.get(var, default if default is not None else "")
        return _ENV_PATTERN.sub(_repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _read_yaml(path: Path) -> dict:
    """读取 YAML 配置为字典；文件不存在返回空字典，供调用方使用默认值。"""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        # 空文件 safe_load 返回 None，统一归一为 {}。
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置格式错误（期望顶层为字典）: {path}")
    return data


def repo_root(start: str | Path | None = None) -> Path:
    """定位仓库根目录：
    1. 优先从 start（默认 cwd）向上查找 config/settings.yaml；
    2. 找不到时回退到安装包所在仓库（支持在任意目录运行 CLI，如从仓库上级目录启动）。
    """
    cur = Path(start or os.getcwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "config" / "settings.yaml").exists():
            return candidate
    package_root = Path(__file__).resolve().parents[2]   # src/llmqa/config.py → 仓库根
    if (package_root / "config" / "settings.yaml").exists():
        return package_root
    # 最终兜底：都找不到时退回 cwd，让后续报错信息指向当前目录。
    return cur


class Pricing(BaseModel):
    """每百万 token 价格（USD），用于成本估算与预算测试。"""
    prompt_per_1m_usd: float = 0.0
    completion_per_1m_usd: float = 0.0


class ProviderConfig(BaseModel):
    """单个 LLM Provider 的接入配置。"""
    name: str
    kind: str = "openai_compat"          # openai_compat | anthropic | mock
    base_url: str | None = None          # 自托管端点；anthropic/mock 可省略
    api_key_env: str | None = None       # 存密钥的环境变量名，密钥本身不落盘
    model: str = "default"
    timeout_seconds: float = 60.0        # 单次 HTTP 请求超时（秒）
    max_retries: int = 2                 # 网络错误重试次数（不含 4xx/5xx 业务错误）
    extra_headers: dict[str, str] = Field(default_factory=dict)  # 追加请求头（如鉴权、租户标识）
    pricing: Pricing = Field(default_factory=Pricing)

    @property
    def api_key(self) -> str | None:
        """从环境变量读取密钥；访问时实时读取，避免把密钥写入配置对象。"""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def resolve_model(self, override: str | None = None) -> str:
        return override or self.model


class Thresholds(BaseModel):
    """默认判定阈值（config/settings.yaml 可覆盖）。"""
    judge_min_score: float = 7.0          # 评判模型给出的最低合格分
    similarity_min: float = 0.75          # 检索/回答相似度下限
    p95_latency_ms: float = 2000.0        # 总时延 P95 上限
    p99_latency_ms: float = 5000.0        # 总时延 P99 上限
    ttft_p95_ms: float = 1500.0           # 首 token 时延（TTFT）P95 上限
    cost_per_request_usd: float = 0.01    # 单请求成本预算上限
    max_tokens_per_request: int = 4096    # 单请求最大输出 token
    max_agent_iterations: int = 8         # Agent 最大迭代轮次


class Settings(BaseModel):
    """全局测试设置（settings.yaml + providers.yaml 合并结果）。"""
    default_provider: str = "mock"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    concurrency: int = 8
    timeout_per_test: float = 90.0
    retries_on_error: int = 1
    fail_fast: bool = False
    report_dir: Path = Path("reports")
    artifact_dir: Path = Path("artifacts")
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> Settings:
        """加载并合并 settings.yaml 与 providers.yaml，返回完整设置。"""
        if config_dir is None:
            config_dir = repo_root() / "config"
        config_dir = Path(config_dir)
        raw = _expand_env(_read_yaml(config_dir / "settings.yaml"))
        providers_raw = _expand_env(_read_yaml(config_dir / "providers.yaml"))
        for name, cfg in providers_raw.items():
            cfg = dict(cfg)
            # providers.yaml 以名称为键，键名即 Provider 名，注入到配置对象供后续引用。
            cfg["name"] = name
            raw.setdefault("providers", {})[name] = cfg
        return cls(**raw)

    def provider(self, name: str) -> ProviderConfig:
        """按名称返回 Provider 配置；不存在时抛出 KeyError 并列出可用名称。"""
        if name not in self.providers:
            raise KeyError(f"未配置 Provider: {name}（providers.yaml 中现有: {sorted(self.providers)}）")
        return self.providers[name]
