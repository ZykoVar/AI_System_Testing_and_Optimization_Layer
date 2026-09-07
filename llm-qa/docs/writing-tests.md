
# 编写测试用例

## 1. 最小用例

```python
from llmqa.core.registry import test
from llmqa.core.models import Severity, TestContext
from llmqa.assertors import assert_contains, assert_json_schema
from llmqa.clients import Message, MockRule, scripted_or_real

@test(id="llm-fmt-001", suite="llm", name="JSON 格式合规",
      tags=("format", "smoke"), severity=Severity.MEDIUM, timeout=60)
async def case_json(ctx: TestContext) -> None:
    """要求模型输出 JSON 并校验 Schema。"""
    client = scripted_or_real(ctx, rules=[
        MockRule(match="JSON", reply='{"sum": 2}')])
    resp = await client.generate([Message.user("用 JSON 返回 1+1")])
    assert_json_schema(resp.text, {"type": "object",
        "required": ["sum"], "properties": {"sum": {"type": "integer"}}})
```

要点：
- 用例 = 一个 async 函数，通过 `@test` 注册；`suite` 决定归属；
- `id` 全局唯一，建议 `<套件>-<域>-<序号>`（如 `rag-ret-001`）；
- 成功返回 = PASS；失败 = 抛 `AssertionFailed` / `AssertionError`；
- 跳过 = `raise SkipTest("原因")`（如环境缺 API Key）；
- 其余异常 = ERROR 并自动重试（`retries_on_error`）。

## 2. 获取被测客户端：双态运行

```python
client = scripted_or_real(ctx, rules=[...])
```

- mock 态（默认）：规则脚本模拟被测模型 → 离线验证测试逻辑；
- 真实态（`--provider openai` 等）：直连模型 → 真实验收。
- 需要固定客户端时：`ctx.client()` / `ctx.client("openai")`；
- 构造临时脚本 mock：`ctx.providers.get_mock(rules=[...])`。

## 3. Mock 规则速查

```python
MockRule(match="正则", reply="文本")                       # 匹配最后一条用户消息
MockRule(match="正则", reply={"refusal": True})            # 标准拒答
MockRule(match="正则", reply={"content": "说", "tool_calls": [
    {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]}, times=1)
MockRule(match="正则", error={"status": 429, "message": "limited"})   # 故障注入
MockRule(match="正则", reply="...", match_transcript=True) # 匹配整个会话转录
MockRule(match=".*", reply="兜底")                          # 兜底
```

规则按注册顺序首个命中生效；`times` 限制命中次数（Agent 多轮脚本必须用）。

## 4. 断言库

确定性断言（`llmqa.assertors`）：

| 函数 | 用途 |
| --- | --- |
| `assert_contains / assert_not_contains` | 关键词存在/禁止（`any_of` 参数支持任一命中） |
| `assert_matches` | 正则匹配 |
| `assert_json_valid / assert_json_schema` | JSON 解析（剥代码围栏）+ 轻量 Schema 校验 |
| `assert_similarity` | 文本相似度 ≥ 阈值（difflib，确定性） |
| `assert_word_count / assert_char_length` | 字数/字符数区间 |
| `assert_refusal / assert_not_refusal` | 拒答检测（中英文话术）与防误拒 |
| `assert_in_language` | 中英文启发式占比 |

软断言（LLM-as-Judge，用于语义维度）：

```python
judge_client = ctx.providers.get_mock(rules=[
    MockRule(match="评分标准", reply='{"score": 9, "reasoning": "准确完整"}')])
verdict = await Judge(judge_client).assert_score(
    question=q, answer=resp.text, context="", criteria="答案必须正确且完整",
    min_score=ctx.settings.thresholds.judge_min_score)
```

生产建议：裁判模型与被测模型解耦，固定裁判 Prompt 版本
（`prompts/judge/correctness.yaml` 为托管模板）。

Judge 两项增强用法：

```python
# 1) 走版本化托管模板（裁判 Prompt 纳入 Prompt 管理，可审计可回滚）
judge = Judge(judge_client, prompt_manager=ctx.prompts)

# 2) 关键判定开启多次投票：温度渐增取中位数，抑制单次打分噪声
verdict = await judge.assert_score(question=q, answer=resp.text, criteria="...",
                                   passes=3)   # verdict.scores/agreement 记录每次分与一致度
```

> 注意：托管模板与内置 Prompt 均要求裁判输出一行 JSON
> `{"score": <分>, "reasoning": "<理由>"}`。

### 4.1 数据驱动用例（列表类测试的首选）

同类用例（注入/越狱/有害内容等"一条载荷一个用例"）用 data_driven 注册，
样板收敛到框架层，新增样本只需在数据集加一条记录：

```python
from llmqa.core.registry import data_driven

@data_driven("adversarial/injections", suite="security",
             id_prefix="sec-inj", name_field="case_name", timeout=60)
async def case_injection(ctx, item):
    """同一份断言逻辑作用于每条记录；item 为数据集记录 dict。"""
    client = refusing_client(ctx, item["mock_match"])   # mock 特征外置在数据里
    ...
```

数据集记录需携带：`case_name`（用例名）、可选 `severity/tags/cost/mock_match`；
用例 id 自动按记录顺序生成（`sec-inj-001..008`），顺序即稳定 id。
维护提示：改数据集字段会重编号前请确认 id 语义；`scripts/lint_ids.py` 在 CI 强制
id 唯一与命名规范；每个用例自动携带源码指纹（数据驱动用例 = 断言函数源码 +
数据集记录内容），同 id 的内容变化在运行溯源（test_identity）中可见。

## 5. 测试上下文（TestContext）

| 成员 | 说明 |
| --- | --- |
| `ctx.client(name=None)` | 连接池取客户端（默认 provider） |
| `ctx.settings` | 全局设置与阈值（`.thresholds.judge_min_score` 等） |
| `ctx.datasets.load("golden_qa")` | 加载数据集（YAML/JSON/CSV） |
| `ctx.prompts.render(id, vars, version=...)` | 渲染版本化 Prompt |
| `ctx.providers.get_mock(...)` | 构造脚本化临时 Mock |

## 6. 规范与最佳实践

1. **用例快速**：mock 下每个用例应在百毫秒级完成；压测 `count ≤ 100`、
   `concurrency ≤ 20`；禁止 `sleep`。
2. **严重级**：CRITICAL=越狱/金丝雀泄露/工具越权；HIGH=注入/护栏/该拒未拒；
   MEDIUM=明显质量缺陷；LOW=格式一致性细节。
3. **可诊断**：失败消息写清"期望什么、实际什么"；数值指标放
   `AssertionFailed(..., metrics={...})`，证据放 evidence——
   evidence 已全链路透传（AssertionFailed → TestOutcome → report.json），
   排障无需重跑即可看到裁判理由等证据。
4. **数据外置**：可复用样本进 `datasets/`；一次性样本可内联但加注释。
5. **不测 Mock 测逻辑**：mock 分支断言的是测试逻辑正确性，不是"模型好"。
6. **注册收尾**：新模块必须在 `suites/<suite>/__init__.py` 导入，
   否则 `discover` 发现不到。
7. **写自测**：框架级变更在 `tests/` 补 pytest 用例。
8. **声明成本**：重用例（多次 LLM 调用）用 `@test(..., cost=N)` 声明成本单位，
   真实模型运行可加 `--max-cost` 预算控制，超预算用例自动 SKIP。
9. **了解重试语义**：只有基础设施故障（429/5xx/网络层）会被重试，
   用例代码 bug（TypeError 等）直接判 ERROR 且消息标注"代码缺陷"。
10. **指标进策略**：新增数值指标时在 `config/metrics_policy.yaml` 声明
    方向与容差，回归对比才会按评价口径判定（未声明的指标仅记录漂移）。

## 7. 注释规范

全仓库统一的可读性约定：

1. **模块 docstring**：首行一句话说明本文件职责，必要时补充使用时机与边界
   （如 Anthropic 适配器声明不支持 function calling）。
2. **函数/类 docstring**：说明参数与返回值语义，不重复函数名；
   测试用例的 docstring 说明"断言什么"。
3. **行内注释解释"为什么"**：设计取舍、协议差异、边界条件、坑位，
   而不是复述代码"是什么"；自明代码不加注释。
4. **关键路径必须注释**：异常路径、协议转换、并发点、护栏判定、
   Mock 规则命中顺序等非显然逻辑。
5. **数据模型字段**：非自明的字段加行内说明（如 `api_key_env` 只存环境变量名）。
6. 注释一律中文，与用例名称、报告语言保持一致。
