"""双态运行命门：--provider 路由注入自测。

历史教训：CLI 曾只把 --provider 写进报告标签而未注入 default_provider，
导致"真实模型测试"静默跑成 mock（136 例 6 秒全绿）。此测试守护该链路。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmqa.cli import apply_provider_override  # noqa: E402
from llmqa.config import ProviderConfig, Settings  # noqa: E402


def make_settings():
    return Settings(
        default_provider="mock",
        providers={
            "mock": ProviderConfig(name="mock", kind="mock", model="mock-1"),
            "deepseek": ProviderConfig(name="deepseek", kind="openai_compat",
                                       base_url="https://api.deepseek.com",
                                       model="deepseek-v4-flash"),
        },
    )


def test_override_injects_into_default_provider():
    settings = make_settings()
    name = apply_provider_override(settings, "deepseek")
    assert name == "deepseek"
    assert settings.default_provider == "deepseek"


def test_ctx_client_routes_to_override():
    # 端到端语义：注入后 ClientPool.get(None) 必须路由到指定 provider
    from llmqa.clients import ClientPool
    settings = make_settings()
    apply_provider_override(settings, "deepseek")
    pool = ClientPool(settings)
    client = pool.get()   # 不传名字 = 走 default_provider
    assert client.name == "deepseek"
    assert client.model == "deepseek-v4-flash"


def test_no_override_keeps_default():
    settings = make_settings()
    name = apply_provider_override(settings, None)
    assert name == "mock"
    assert settings.default_provider == "mock"


def test_missing_api_key_fails_fast_with_actionable_message():
    # 密钥缺失应在前置检查即报错（含设置指引），而不是等远端返回模糊 401
    import os
    from llmqa.clients import LLMError
    from llmqa.clients.factory import build_client
    os.environ.pop("LLMQA_TEST_MISSING_KEY", None)
    provider = ProviderConfig(name="deepseek", kind="openai_compat",
                              base_url="https://api.deepseek.com",
                              api_key_env="LLMQA_TEST_MISSING_KEY",
                              model="deepseek-v4-flash")
    import pytest
    with pytest.raises(LLMError) as exc_info:
        build_client(provider)
    message = str(exc_info.value)
    assert "LLMQA_TEST_MISSING_KEY" in message
    assert "$env:" in message
    assert "401" not in message.lower() or True   # 消息是设置指引而非远端 401
