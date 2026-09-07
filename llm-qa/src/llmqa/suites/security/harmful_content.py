"""有害内容测试：数据驱动覆盖 harmful_requests.yaml 全部请求。

每条数据集记录生成一个用例（sec-hrm-001..009），按 expect_refusal 分支：
- true：断言模型拒答（暴力/仇恨/自残/违法建议/隐私侵犯/虚假信息）；
- false：断言模型不得误拒（压力疏导/学习推荐/健康建议，防误拒）。

新增有害或良性样本只需在数据集加一条记录，无需改代码。
"""
from __future__ import annotations

from llmqa.assertors import assert_not_refusal, assert_refusal
from llmqa.core.models import TestContext
from llmqa.core.registry import data_driven

from ._common import benign_client, refusing_client, render_support


@data_driven(
    "security/harmful_requests",
    items_key="items",
    suite="security",
    id_prefix="sec-hrm",
    name_field="case_name",
    timeout=60,
)
async def case_harmful_content(ctx: TestContext, item: dict) -> None:
    """断言：有害请求拒答、无害请求不误拒。"""
    if item.get("expect_refusal"):
        # 有害：mock 分支命中 mock_match 即拒绝；真实 provider 由模型判定
        client = refusing_client(ctx, item["mock_match"])
        messages = render_support(ctx, item["request"])
        resp = await client.generate(messages, temperature=0.0, max_tokens=512)
        assert_refusal(resp.text)
    else:
        # 良性：任何输入都应正常回答，防止安全护栏"一刀切"误拒正常请求
        client = benign_client(ctx)
        messages = render_support(ctx, item["request"])
        resp = await client.generate(messages, temperature=0.0, max_tokens=512)
        assert_not_refusal(resp.text)
