"""重试策略分桶：区分"基础设施故障"（值得重试）与"代码缺陷"（重试无意义）。

策略（is_retryable_error）：
- LLMError：HTTP 408/425/429 与 5xx → 重试；400/401/403/404 → 不重试；
- httpx 网络层异常（连接失败/读取超时等）→ 重试；
- 其他一切（KeyError/TypeError/ValueError/AssertionError）→ 不重试，直接判 ERROR。
"""
from __future__ import annotations

from llmqa.clients.base import LLMError

# 值得重试的 HTTP 状态：请求超时/过早发送/限流
_RETRYABLE_STATUS = {408, 425, 429}


def _is_httpx_error(exc: BaseException) -> bool:
    """httpx 异常模块名为 'httpx' 或 'httpx._exceptions'（版本差异），两者都算网络层。"""
    module = type(exc).__module__ or ""
    return module == "httpx" or module.startswith("httpx.")


def is_retryable_error(exc: BaseException) -> bool:
    """判定异常是否属于"基础设施故障"，值得按退避策略重试。"""
    if isinstance(exc, LLMError):
        return exc.status in _RETRYABLE_STATUS or (
            exc.status is not None and exc.status >= 500)
    if _is_httpx_error(exc):
        # httpx 的连接/超时/协议异常都是网络层问题，重试有意义
        return True
    return False


def error_kind(exc: BaseException) -> str:
    """异常分类标签，写入 ERROR 消息便于排障。"""
    if isinstance(exc, LLMError):
        return "基础设施(HTTP {})".format(exc.status or "?")
    if _is_httpx_error(exc):
        return "网络层"
    return "代码缺陷"
