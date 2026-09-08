
# 架构设计

## 0. 战略定位（先于分层）

llm_learn 是 **AI System Testing & Optimization Layer**：
外部平台负责"能力"（模型网关/指标/轨迹/监控），本项目负责"工程体系"
（统一测试模型、行为规定、回归、优化闭环、质量门禁）。

```text
                         llm_learn
                             │
                ┌────────────┴────────────┐
                │                         │
          LLM Evaluation             Agent Testing
                │                         │
        ┌───────┴───────┐         ┌───────┴───────┐
        │               │         │               │
   External Eval     Native     External Trace  Native
    Providers         Eval      (LangSmith/      Harness
   (Ragas/Judge)                 Langfuse/...)
        │               │         │               │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                └──────────┬──────────────┘
                           ▼
               Unified Outcome（TestOutcome / AgentTrajectory）
                           │
                     Regression（compare + 指标策略 + 基线）
                           │
                       Quality Gate
```

核心资产：**统一测试模型（TestOutcome）+ 轨迹统一模型（AgentTrajectory）+
行为断言 DSL + 回归语义**；接入点：`llmqa/ext/` 的四个 Adapter 协议
（LLM 客户端 / Judge 后端 / 检索后端 / 轨迹归一）。

## 1. 分层视图

llm-qa 采用四层架构，上层依赖下层稳定接口，下层不感知上层：

```text
┌─────────────────────────────────────────────────────────────┐
│ L4 测试套件层：llm / rag / agent / security / performance   │
│    - @test 注册的 async 用例，通过 ctx 访问一切资源           │
├─────────────────────────────────────────────────────────────┤
│ L3 测试支撑层：harnesses + assertors + load                  │
│    - RAGHarness：分块、BM25-lite 索引、检索评估、RAG 生成     │
│    - AgentHarness：工具调用环、循环检测、预算/白名单护栏       │
│    - Judge：LLM-as-Judge 软断言（正确性/忠实性/相关性）        │
│    - assertors：确定性断言（文本/JSON Schema/拒答/相似度）     │
│    - run_load：并发压测、延迟分位、吞吐/token 速率统计         │
├─────────────────────────────────────────────────────────────┤
│ L2 执行引擎层：registry + runner + reporter                  │
│    - @test 装饰器注册 → discover 包发现 → 过滤（suite/tag/    │
│      severity）→ TestRunner（并发/超时/重试/fail-fast）       │
│    - Verdict：PASS / FAIL(被测缺陷) / ERROR(基础设施) / SKIP  │
│    - Reporter：控制台进度 + JSON/MD/HTML/JUnit 四格式报告     │
├─────────────────────────────────────────────────────────────┤
│ L1 资产与接入层：config + clients + prompts + datasets       │
│    - Settings：Pydantic 模型，YAML + 环境变量展开             │
│    - LLMClient：统一接口 + 三种实现（Mock/OpenAI兼容/         │
│      Anthropic）+ ClientPool 连接池                          │
│    - PromptManager：版本化模板、渲染校验、diff、状态流转       │
│    - PromptScanner：发布前注入扫描                            │
│    - DatasetManager：YAML/JSON/CSV 数据集                    │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心数据流

```text
CLI (llmqa run)
  │ 1. Settings.load(config/) → PromptManager(prompts/) → DatasetManager(datasets/)
  │ 2. discover(DEFAULT_PACKAGES) 导入五大套件包 → @test 注册到全局注册表
  │ 3. 过滤：--suite / --tag / --exclude-tag / --severity / fail-fast
  ▼
TestRunner.run_sync
  │ 4. 每用例独立 TestContext（含连接池/提示库/数据集）
  │ 5. asyncio.Semaphore 并发调度，asyncio.wait_for 超时
  │ 6. 捕获：AssertionFailed→FAIL；AssertionError→FAIL；
  │    TimeoutError/其他异常→ERROR（按 retries_on_error 重试）
  ▼
Reporter.finalize
  │ 7. reports/<run_id>/report.{json,md,html} + junit.xml
  │ 8. 退出码：0=全过/跳过；1=存在 FAIL 或 ERROR（--soft 可放宽）
  ▼
CI（GitHub Actions）
    9. 上传报告 artifacts，按严重级门禁判定 job 成败
```

## 3. 关键契约

### 3.1 测试用例契约

```python
from llmqa.core.registry import test
from llmqa.core.models import Severity, TestContext

@test(id="sec-inj-001", suite="security", name="直接注入防御",
      tags=("injection", "smoke"), severity=Severity.CRITICAL)
async def case(ctx: TestContext) -> None:
    resp = await ctx.client().generate([...])
    assert_refusal(resp.text)
```

用例函数签名固定为 `async def case(ctx: TestContext) -> None`；
成功返回即 PASS；抛 `AssertionFailed`/`AssertionError` 记 FAIL；
抛 `SkipTest` 记 SKIP；其他异常记 ERROR（自动重试）。

### 3.2 LLM 客户端契约

所有 Provider 实现 `LLMClient`：

```python
async def generate(messages, *, temperature=0.0, max_tokens=512,
                   stop=None, tools=None, tool_choice=None) -> LLMResponse
async def stream(messages, ...) -> AsyncIterator[StreamChunk]
```

`LLMResponse` 统一携带：文本、工具调用、token 用量、延迟（毫秒）、
估算成本、finish_reason、原始响应（`raw`，仅含可序列化数据）。

### 3.3 Mock 脚本契约

MockClient 按"规则列表"模拟被测模型，规则按注册顺序首个命中生效：

```python
MockRule(match="正则", reply="文本")                    # 匹配最后一条用户消息
MockRule(match="正则", reply={"refusal": True})         # 标准拒答
MockRule(match="正则", reply={"content": "", "tool_calls": [...]}, times=1)
MockRule(match="正则", error={"status": 429, ...})      # 故障注入
MockRule(match="正则", reply="...", match_transcript=True)  # 匹配整个会话转录
```

用例通过 `scripted_or_real(ctx, rules=[...])` 取客户端：
默认 provider 为 mock 时用脚本（离线验证测试逻辑），
否则直连真实模型（同一用例即真实测试），实现"一次编写，两态运行"。

## 4. 判定与严重级模型

| Verdict | 语义 | 是否重试 | 影响退出码 |
| --- | --- | --- | --- |
| PASS | 断言全部通过 | - | 否 |
| FAIL | 被测对象不满足要求（缺陷） | 否 | 是 |
| ERROR | 基础设施故障 / 代码缺陷 | 仅基础设施类（见重试分桶） | 是 |
| SKIP | 按 skip_reason 分类（见下） | - | 否 |

**重试分桶**（core/retry.py）：LLMError(429/408/425/5xx) 与 httpx 网络层异常
→ 重试；其余一切（KeyError/TypeError/AssertionError）→ 代码缺陷，不重试，
立即判 ERROR 并在消息中标注分类。

**SKIP 语义分类**（skip_reason）：intentional（有意跳过）｜budget（预算耗尽）｜
fail_fast（前置高危失败）。回归对比时 budget/fail_fast 视为**中性**（未执行≠变差），
intentional 视为覆盖丢失计入回归。

严重级五档：INFO < LOW < MEDIUM < HIGH < CRITICAL。
报告按严重级倒序排列失败；`--fail-fast` 在 HIGH 及以上失败时停止调度新用例；
CI 门禁按严重级分层（见 docs/ci-integration.md）。

## 4.5 运行溯源（Provenance）

每次运行的报告携带：git commit、Prompt 版本+内容哈希（sha256）、数据集指纹、
模型接入信息、用例源码指纹（case_id → source_hash）。
version 是命名约定，content_hash 才是防篡改依据；配合命名基线
（reports/baselines/<name>.json）与指标策略（config/metrics_policy.yaml），
回归对比回答的不只是"变没变"，而是"在哪次代码/数据/Prompt 变更上变的"。

## 5. 扩展点

- **新 Provider（薄化策略）**：新增单家适配器不再是默认路径——统一走 LiteLLM（kind: litellm，100+ 模型归一化）；openai_compat/anthropic 保留为离线/轻依赖后备。核心命题不是支持多少模型，而是把任何模型/评测引擎的结果纳入统一 QA 生命周期。
- **成熟工具后端**（`llmqa/ext/`，全部懒加载，不装则离线 CI 不受影响；
  `pip install -e ".[ext]"` 激活）：
  - LiteLLM 统一网关：`providers.yaml` 设 `kind: litellm`（100+ 模型/重试/成本）；
  - Ragas 指标：`JudgeBackend` 契约 + 接入骨架（contract，绑定按版本补全；
    投票/证据/门禁仍归 Judge）——不宣称 production-ready；
  - 外部检索后端：实现 `Retriever` 协议传入 `RAGHarness(retriever=...)`
    （生产接 Chroma/Qdrant/LlamaIndex，离线 CI 保留 BM25-lite）。
- **新断言**：在 `assertors/` 添加函数或 Judge 维度。
- **新套件**：新建包 `suites/<name>/`，在 `__init__.py` 导入模块，
  把包名加入 `DEFAULT_PACKAGES`。
- **新数据集**：`datasets/` 下任意 YAML/JSON/CSV，用例内 `ctx.datasets.load()`。
- **新 Prompt**：`prompts/` 下 YAML，CLI `llmqa prompts validate/scan` 纳入门禁。

## 6. 项目边界（不做什么）

```text
不继续卷：
  ❌ 更多 evaluator / 更多 LLM provider / 更多 RAG metric / 更多向量数据库
  ❌ 更复杂的 tracing / 自建 SaaS dashboard / 大型分布式压测（交给 k6/locust）

继续卷（项目壁垒所在）：
  ✅ Test identity → Baseline → Provenance → Compatibility
  ✅ Regression semantics → Metric policy → Severity policy
  ✅ CI gate → Release decision
      （Evaluation → Normalize → Compare → Policy → Decision）
```

成熟工具当"引擎"，本项目当"判定层"；对成熟工具的声明以"可用适配 vs
接入骨架"如实区分（见 llmqa/ext/）。
