"""格式合规测试套件（llm-fmt-）：验证模型对输出格式硬约束的遵循能力。

关注点：
- JSON 输出与 JSON Schema 校验（字段类型、必填项、取值范围）；
- 列表 / 纯数字 / 纯文本等特定输出格式；
- 代码围栏包裹的 JSON 可被宽松解析（assert_json_valid 支持围栏）；
- 禁止 Markdown 标记（#、**、围栏）等负向格式约束。

断言均为确定性断言（不依赖 LLM），mock 分支离线可复现、用于验证测试逻辑；
真实 Provider 下由真实模型遵循格式约束。
"""
from __future__ import annotations

from llmqa.assertors import (
    assert_contains,
    assert_json_schema,
    assert_json_valid,
    assert_matches,
    assert_not_contains,
)
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test

# 代码围栏字面量（避免在源码中直接写反引号）
_FENCE = chr(96) * 3  # 三反引号围栏的共享字面量，避免各用例重复拼写


@test(
    id="llm-fmt-001",
    suite="llm",
    name="JSON 输出且通过 Schema 校验",
    description="要求模型输出 JSON，并校验字段类型、必填项与取值范围。",
    tags=("format", "smoke"),
    severity=Severity.LOW,
    timeout=60,
)
async def json_output_schema(ctx: TestContext) -> None:
    """断言：回复为合法 JSON，且满足 name:string、price:integer>=0、in_stock:boolean 的 Schema。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="专业版订阅", reply='{"name": "Acme 专业版", "price": 299, "in_stock": true}'),
    ])
    resp = await client.generate(
        [Message.user("请以 JSON 格式输出 Acme 专业版订阅信息，字段为 name、price、in_stock。")],
        temperature=0.0, max_tokens=512,
    )
    schema = {
        "type": "object",
        "required": ["name", "price"],  # in_stock 刻意不列为必填：同时校验必填缺失与可选字段类型
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "integer", "minimum": 0},
            "in_stock": {"type": "boolean"},
        },
    }
    assert_json_schema(resp.text, schema)  # 宽松解析（剥围栏/取首个对象）后按 Schema 校验，并返回解析后的对象


@test(
    id="llm-fmt-002",
    suite="llm",
    name="按列表格式输出",
    description="要求模型用列表逐条输出，验证每行以列表标记开头且条目完整。",
    tags=("format",),
    severity=Severity.LOW,
    timeout=60,
)
async def list_format(ctx: TestContext) -> None:
    """断言：回复每行以 -/*/• 开头，并包含三条服务特点的关键词。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="列表", reply="- 支持 7 天无理由退货\n- 退款 3-5 个工作日到账\n- 工作日 9:00-18:00 客服在线"),
    ])
    resp = await client.generate(
        [Message.user("请用列表逐条列出 Acme 的三项服务特点。")],
        temperature=0.0, max_tokens=512,
    )
    assert_matches(resp.text, r"(?m)^\s*[-*•]\s+")  # (?m) 多行模式逐行校验列表标记；\s* 容忍缩进
    assert_contains(resp.text, "7 天无理由退货", "3-5 个工作日", "9:00-18:00")


@test(
    id="llm-fmt-003",
    suite="llm",
    name="禁止 Markdown 标记",
    description="要求模型用纯文本回答，验证回复不含 #、**、代码围栏等 Markdown 标记。",
    tags=("format",),
    severity=Severity.LOW,
    timeout=60,
)
async def no_markdown(ctx: TestContext) -> None:
    """断言：回复为纯文本，不含任何 Markdown 标记。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="纯文本", reply="退货政策为 7 天无理由退货，退款 3-5 个工作日到账。"),
    ])
    resp = await client.generate(
        [Message.user("请用纯文本回答 Acme 的退货政策，不要使用任何 Markdown 标记。")],
        temperature=0.0, max_tokens=512,
    )
    assert_not_contains(resp.text, "#", "**", _FENCE)  # 任一 Markdown 标记出现即违规（负向硬约束）


@test(
    id="llm-fmt-004",
    suite="llm",
    name="纯数字格式输出",
    description="要求模型只输出一个数字，验证回复为纯数字且无其他文字。",
    tags=("format",),
    severity=Severity.LOW,
    timeout=60,
)
async def pure_number(ctx: TestContext) -> None:
    """断言：回复去除首尾空白后仅由数字组成。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="数字", reply="42"),
    ])
    resp = await client.generate(
        [Message.user("只输出一个数字，不要任何解释：6 乘以 7 等于多少？")],
        temperature=0.0, max_tokens=512,
    )
    assert_matches(resp.text, r"^\s*\d+\s*$")  # 全量锚定匹配，杜绝数字前后夹带文字或解释


@test(
    id="llm-fmt-005",
    suite="llm",
    name="JSON 中文字段完整",
    description="要求模型在 JSON 中用中文完整填写注释字段，验证字段类型与内容完整。",
    tags=("format",),
    severity=Severity.LOW,
    timeout=60,
)
async def json_chinese_comment(ctx: TestContext) -> None:
    """断言：JSON 含 status/comment 两个字符串字段，且中文 comment 内容完整无截断。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="订单状态",
                 reply='{"status": "成功", "comment": "您的订单已处理完成，预计 3-5 个工作日送达"}'),
    ])
    resp = await client.generate(
        [Message.user("请用 JSON 输出订单状态，comment 字段用中文完整说明处理结果。")],
        temperature=0.0, max_tokens=512,
    )
    schema = {
        "type": "object",
        "required": ["status", "comment"],
        "properties": {
            "status": {"type": "string"},
            "comment": {"type": "string", "minLength": 5},
        },
    }
    obj = assert_json_schema(resp.text, schema)  # 复用校验返回的对象直接取字段，避免二次 JSON 解析
    assert_contains(obj["comment"], "3-5 个工作日送达")  # minLength=5 只保长度，这里再校验内容完整无截断


@test(
    id="llm-fmt-006",
    suite="llm",
    name="代码围栏包裹的 JSON 可解析",
    description="要求模型用 Markdown 代码围栏包裹 JSON，验证宽松解析器能提取并解析。",
    tags=("format",),
    severity=Severity.LOW,
    timeout=60,
)
async def fenced_json(ctx: TestContext) -> None:
    """断言：围栏包裹的 JSON 可被 assert_json_valid 解析为期望对象。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="围栏",
                 reply=_FENCE + "json\n" + '{"answer": "上海", "confidence": 0.95}' + "\n" + _FENCE),
    ])
    resp = await client.generate(
        [Message.user("请用 Markdown 代码围栏包裹 JSON，输出 Acme 总部所在城市。")],
        temperature=0.0, max_tokens=512,
    )
    obj = assert_json_valid(resp.text)  # parse_json 自动剥除围栏并截取首个 {...}，故围栏包裹也能解析
    assert obj["answer"] == "上海"
    assert obj["confidence"] == 0.95
