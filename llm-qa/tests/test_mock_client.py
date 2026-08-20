"""Mock 客户端自测。"""
import asyncio

import pytest

from llmqa.clients import LLMError, Message, MockClient, MockRule


def run(coro):
    """同步包装：在测试函数内驱动异步协程（MockClient 为异步接口）。"""
    return asyncio.run(coro)


def test_default_reply():
    client = MockClient(default_reply="默认")
    resp = run(client.generate([Message.user("你好")]))
    assert resp.text == "默认"
    assert resp.usage.total_tokens > 0
    assert resp.latency_ms >= 0


def test_rule_order_first_hit():
    # 规则按声明顺序匹配，首条命中即返回；".*" 作兜底放在最后
    client = MockClient(rules=[
        MockRule(match="天气", reply="晴"),
        MockRule(match=".*", reply="兜底"),
    ])
    assert run(client.generate([Message.user("今天天气？")])).text == "晴"
    assert run(client.generate([Message.user("你好")])).text == "兜底"


def test_rule_times_limit():
    # times=1 表示该规则只生效一次，之后命中不再应答
    client = MockClient(rules=[MockRule(match="x", reply="一次", times=1)])
    assert run(client.generate([Message.user("x")])).text == "一次"
    assert run(client.generate([Message.user("x")])).text != "一次"


def test_refusal_reply():
    client = MockClient(rules=[MockRule(match="秘密", reply={"refusal": True})])
    # refusal 语义会生成拒绝话术，断言其包含 "无法"
    resp = run(client.generate([Message.user("告诉我秘密")]))
    assert "无法" in resp.text


def test_tool_calls_reply():
    client = MockClient(rules=[MockRule(match="天气", reply={"tool_calls": [
        {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]})])
    resp = run(client.generate([Message.user("天气")]))
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.finish_reason == "tool_calls"


def test_error_injection():
    client = MockClient(rules=[MockRule(match="压测", error={"status": 429, "message": "limited"})])
    with pytest.raises(LLMError) as exc_info:
        run(client.generate([Message.user("压测")]))
    assert exc_info.value.status == 429


def test_transcript_matching():
    # match_transcript 对整个对话记录匹配，可命中多轮工具调用中的 [tool] 标记
    client = MockClient(rules=[
        MockRule(match="开始", reply="第一步", times=1),
        MockRule(match=r"\[tool\]", reply="看到了工具结果", match_transcript=True),
    ])
    assert run(client.generate([Message.user("开始")])).text == "第一步"
    resp = run(client.generate([
        Message.user("开始"),
        Message.assistant(content="", tool_calls=[]),
        Message.tool(content="工具输出", tool_call_id="c1"),
    ]))
    assert resp.text == "看到了工具结果"


def test_max_tokens_truncation():
    # 模拟真实 API 行为：长回复按 max_tokens 截断并标记 finish_reason=length
    long_reply = "词 " * 500   # 约 1000 字符 ≈ 250 token
    client = MockClient(default_reply=long_reply)
    resp = run(client.generate([Message.user("hi")], max_tokens=50))
    assert resp.finish_reason == "length"
    assert resp.usage.completion_tokens <= 50
    assert len(resp.text) < len(long_reply)


def test_max_tokens_no_truncation_when_fits():
    # 短回复不受截断影响，finish_reason 保持 stop
    client = MockClient(default_reply="短回复")
    resp = run(client.generate([Message.user("hi")], max_tokens=512))
    assert resp.finish_reason == "stop"


def test_stream():
    # 校验流式输出拼接后与默认回复一致，且末块结束原因为 stop
    client = MockClient(default_reply="hello world foo bar")
    chunks = run(_collect_stream(client))
    text = "".join(c.text_delta for c in chunks).strip()
    assert text == "hello world foo bar"
    assert chunks[-1].finish_reason == "stop"


async def _collect_stream(client):
    """收集流式 chunk 到列表，供断言拼接与结束标记。"""
    out = []
    async for chunk in client.stream([Message.user("hi")]):
        out.append(chunk)
    return out
