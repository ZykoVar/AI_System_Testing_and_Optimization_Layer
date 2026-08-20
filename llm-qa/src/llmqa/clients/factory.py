"""客户端工厂与连接池：按 Provider 配置构建并缓存客户端。"""
from __future__ import annotations

from llmqa.clients.base import LLMClient, LLMError
from llmqa.clients.mock import MockClient, MockRule
from llmqa.config import ProviderConfig, Settings


def build_client(provider: ProviderConfig) -> LLMClient:
    """按 Provider 配置的 kind 构建对应客户端；未知 kind 抛 LLMError。"""
    kind = provider.kind
    if kind == "mock":
        return MockClient(name=provider.name, model=provider.resolve_model())
    if kind == "openai_compat":
        # 延迟导入：只在真正需要时引入，减少无用依赖与潜在循环导入风险。
        from llmqa.clients.openai_compat import OpenAICompatClient
        if not provider.base_url:
            raise LLMError(provider.name, "openai_compat 需要配置 base_url")
        return OpenAICompatClient(
            provider.name, provider.resolve_model(), provider.base_url,
            api_key=provider.api_key, timeout_seconds=provider.timeout_seconds,
            max_retries=provider.max_retries, extra_headers=provider.extra_headers,
            pricing=provider.pricing,
        )
    if kind == "anthropic":
        from llmqa.clients.anthropic import AnthropicClient  # 同上，延迟导入
        return AnthropicClient(
            provider.name, provider.resolve_model(), api_key=provider.api_key,
            base_url=provider.base_url or "https://api.anthropic.com",
            timeout_seconds=provider.timeout_seconds, max_retries=provider.max_retries,
        )
    raise LLMError(provider.name, f"未知 Provider kind: {kind}")


def scripted_or_real(ctx, rules: list[MockRule | dict], *,
                      mock_name: str = "scripted-mock",
                      default_reply: str = "脚本化默认回复。"):
    """测试用例取客户端的最佳实践：
    - 默认 provider 为 mock 时：返回带规则脚本的 MockClient（离线可复现，验证测试逻辑）；
    - 否则：返回真实 Provider 客户端（同一用例直接变成真实模型测试）。
    需要 TestContext，避免与 ClientPool 的循环依赖，故定义为延迟取用。"""
    cfg = ctx.settings.provider(ctx.settings.default_provider)
    if cfg.kind == "mock":
        client = ctx.providers.get_mock(name=mock_name, default_reply=default_reply)
        for rule in rules:
            client.add_rule(rule)
        return client
    return ctx.client()


class ClientPool:
    """按名称缓存客户端，测试用例通过 ctx.client("openai") 复用同一连接。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, LLMClient] = {}

    def get(self, name: str | None = None) -> LLMClient:
        """按名称返回缓存的客户端；未指定名称时使用默认 Provider。"""
        name = name or self.settings.default_provider
        if name not in self._cache:
            # 缓存连接池：同一 Provider 复用同一 httpx 客户端，避免反复建连。
            self._cache[name] = build_client(self.settings.provider(name))
        return self._cache[name]

    def get_mock(self, *, name: str = "mock-test", default_reply: str = "",
                 rules: list[MockRule | dict] | None = None) -> MockClient:
        """为测试用例构造带脚本的临时 MockClient（不进入池缓存）。"""
        client = MockClient(name=name, default_reply=default_reply or "测试 Mock 回复。")
        for rule in rules or []:
            client.add_rule(rule)
        return client
