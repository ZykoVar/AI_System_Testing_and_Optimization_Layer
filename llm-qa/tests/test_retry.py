"""重试策略分桶（core.retry）自测。"""
import httpx
import pytest

from llmqa.clients import LLMError
from llmqa.core.retry import error_kind, is_retryable_error


def test_retryable_http_status():
    # 限流/服务端错误可重试
    assert is_retryable_error(LLMError("p", "limited", status=429))
    assert is_retryable_error(LLMError("p", "overloaded", status=503))
    assert is_retryable_error(LLMError("p", "timeout", status=408))


def test_non_retryable_http_status():
    # 客户端错误重试无意义
    assert not is_retryable_error(LLMError("p", "unauthorized", status=401))
    assert not is_retryable_error(LLMError("p", "forbidden", status=403))
    assert not is_retryable_error(LLMError("p", "not found", status=404))


def test_network_errors_retryable():
    assert is_retryable_error(httpx.ConnectError("连接被拒绝"))
    assert is_retryable_error(httpx.ReadTimeout("读取超时"))


def test_code_bugs_not_retryable():
    assert not is_retryable_error(TypeError("int + str"))
    assert not is_retryable_error(KeyError("missing"))
    assert not is_retryable_error(ValueError("bad value"))
    assert not is_retryable_error(AssertionError("断言"))


def test_error_kind_labels():
    assert error_kind(LLMError("p", "x", status=429)) == "基础设施(HTTP 429)"
    assert error_kind(httpx.ConnectError("x")) == "网络层"
    assert error_kind(TypeError("x")) == "代码缺陷"
