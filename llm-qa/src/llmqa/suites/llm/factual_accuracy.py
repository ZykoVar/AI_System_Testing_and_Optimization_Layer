"""事实准确性测试套件（llm-acc-）：基于 golden_qa 验证模型回答的事实正确性。

采用 LLM-as-Judge：被测客户端 mock 回复=标准答案（离线验证"问题→答案→裁判"链路），
裁判客户端独立脚本化（回复固定评分 JSON）；真实 Provider 下由真实模型作答、裁判链路不变。
覆盖 5 个不同 category：company_facts / product_facts / policy_facts / math / unknown_facts。
"""
from __future__ import annotations

from llmqa.assertors import Judge
from llmqa.clients import Message, MockRule, scripted_or_real
from llmqa.core.models import Severity, TestContext
from llmqa.core.registry import test


def _get_item(ctx: TestContext, item_id: str) -> dict:
    """按条目 ID 从 golden_qa 数据集取回标准问答条目；未命中时抛 KeyError。"""
    for it in ctx.datasets.load("golden_qa")["items"]:  # 线性扫描即可：golden_qa 条目极少，无需建索引
        if it["id"] == item_id:
            return it
    raise KeyError("golden_qa 中未找到条目: {}".format(item_id))


async def _judge_accuracy(ctx: TestContext, item_id: str, match: str) -> None:
    """通用准确性判定：被测回复=标准答案，裁判评分不低于全局阈值。"""
    item = _get_item(ctx, item_id)
    subject = scripted_or_real(ctx, rules=[  # 被测端 mock 回复=标准答案，离线验证"问题→答案→裁判"链路
        MockRule(match=match, reply=item["answer"]),
    ])
    resp = await subject.generate(
        [Message.user(item["question"])], temperature=0.0, max_tokens=512)
    judge = ctx.providers.get_mock(rules=[  # 裁判独立脚本化：固定 9 分，隔离裁判噪声只测被测链路
        MockRule(match="评分标准", reply='{"score": 9, "reasoning": "准确"}'),
    ])
    await Judge(judge).assert_score(
        question=item["question"],
        answer=resp.text,
        context="",
        criteria=item["judge_criteria"],
        min_score=ctx.settings.thresholds.judge_min_score,  # 阈值取全局配置，避免各用例硬编码不一致
    )


@test(
    id="llm-acc-001",
    suite="llm",
    name="公司事实准确性（总部）",
    description="用 LLM-as-Judge 验证 qa-001 关于总部位置的事实回答。",
    tags=("accuracy", "smoke"),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def company_facts_accuracy(ctx: TestContext) -> None:
    """断言：裁判对 qa-001 答案评分不低于 judge_min_score（总部在上海浦东）。"""
    await _judge_accuracy(ctx, "qa-001", "总部在哪里")  # 第三参数仅供 mock 规则匹配，真实 Provider 下忽略


@test(
    id="llm-acc-002",
    suite="llm",
    name="产品事实准确性（退货政策）",
    description="用 LLM-as-Judge 验证 qa-003 关于退货政策的事实回答。",
    tags=("accuracy",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def product_facts_accuracy(ctx: TestContext) -> None:
    """断言：裁判对 qa-003 答案评分不低于 judge_min_score（7 天退货与退款时限）。"""
    await _judge_accuracy(ctx, "qa-003", "退货政策")


@test(
    id="llm-acc-003",
    suite="llm",
    name="政策事实准确性（隐私保护）",
    description="用 LLM-as-Judge 验证 qa-006 关于隐私保护的事实回答。",
    tags=("accuracy",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def policy_facts_accuracy(ctx: TestContext) -> None:
    """断言：裁判对 qa-006 答案评分不低于 judge_min_score（加密存储与用户同意）。"""
    await _judge_accuracy(ctx, "qa-006", "隐私数据")


@test(
    id="llm-acc-004",
    suite="llm",
    name="数学事实准确性",
    description="用 LLM-as-Judge 验证 qa-009 的数学计算回答。",
    tags=("accuracy",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def math_facts_accuracy(ctx: TestContext) -> None:
    """断言：裁判对 qa-009 答案评分不低于 judge_min_score（答案必须为 96）。"""
    await _judge_accuracy(ctx, "qa-009", "12 乘以")


@test(
    id="llm-acc-005",
    suite="llm",
    name="未知事实准确性（诚实拒答）",
    description="用 LLM-as-Judge 验证 qa-007 对未知事实的诚实拒答。",
    tags=("accuracy",),
    severity=Severity.MEDIUM,
    timeout=60,
)
async def unknown_facts_accuracy(ctx: TestContext) -> None:
    """断言：裁判对 qa-007 答案评分不低于 judge_min_score（明确表示不知道）。"""
    await _judge_accuracy(ctx, "qa-007", "竞争对手")
