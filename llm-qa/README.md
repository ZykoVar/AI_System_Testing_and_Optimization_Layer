
# llm-qa —— 企业级 LLM/Agent 质量保障测试框架

把 LLM/Agent 的全部测试流程沉淀为可复现、可回归、可上 CI 的代码资产：
**Prompt 管理 · LLM 质量测试 · RAG 专项测试 · Agent 测试 · 安全红队测试 · 性能测试**。

- 无厂商锁定：统一 `LLMClient` 抽象，同一批用例可在 Mock / OpenAI 兼容（OpenAI、DeepSeek、vLLM、Ollama）/ Anthropic 之间一键切换
- 离线可跑：内置确定性 Mock Provider 与脚本化规则，CI 冒烟零成本、零密钥
- 企业级工程：版本化 Prompt 库、LLM-as-Judge、多格式报告（JSON/Markdown/HTML/JUnit）、失败分级与门禁退出码
- 安全第一：内置提示注入/越狱/金丝雀泄露/数据外泄红队用例与 Prompt 发布前扫描
- **可重建实验**：每次运行携带完整溯源（git commit / Prompt 版本+内容哈希 / 数据集指纹 / 模型 / 用例源码指纹），任何结果都可追溯到"用什么代码、什么数据、什么 Prompt 跑出来的"
- **回归平台**：命名基线（production/security/...）、指标漂移判定策略（方向+容差）、重试分桶与 SKIP 语义分类——compare 是评价引擎而非字面 diff

## 功能矩阵

| 子系统 | 能力 |
| --- | --- |
| Prompt 管理 | 版本化模板库、变量声明与渲染校验、版本 diff、状态流转（draft/active/deprecated）、发布前注入扫描 |
| LLM 质量 | 格式合规（JSON Schema）、指令遵循、事实准确性（Golden QA + Judge）、幻觉/拒答、一致性、多语言、语气安全、长度控制 |
| RAG 专项 | 检索质量（recall@k/hit@k/MRR/precision@k）、分块质量、忠实性、相关性、引用质量、端到端管线 |
| Agent | 工具选择、参数 Schema、多步规划、会话记忆、循环检测、预算护栏、工具白名单 |
| 安全红队 | 直接/间接注入、越狱、提示词窃取、PII 金丝雀、有害内容拒答与防误拒、数据外泄、混淆绕过 |
| 性能 | 延迟分位（P50-P99）、TTFT、吞吐、并发扩展、成本与 token 效率、长上下文、限流行为 |
| 回归平台 | 运行溯源（commit/内容哈希）、命名基线、report compare（指标策略+SKIP 语义）、严重级门禁 |

## 快速开始

```powershell
# 1. 安装（Python 3.10+）
cd llm-qa
pip install -e ".[dev]"

# 2. 运行内置演示（全部基于 Mock，无需任何 API Key）
llmqa demo

# 3. 查看全部内置用例
llmqa run --list

# 4. 跑安全红队套件（默认 Mock 脚本模式，离线验证测试逻辑）
llmqa run --suite security

# 5. 接入真实模型（OpenAI 兼容协议）
$env:OPENAI_API_KEY = "sk-..."
llmqa run --provider openai --suite llm --tag smoke

# 6. 回归基线（发布流程：全绿运行登记为基线，之后每次对比）
llmqa report baseline-set <run_id> --name production
llmqa report compare --baseline --baseline-name production
```

运行后会生成四格式报告：`reports/<run_id>/report.{json,md,html}` 与 `junit.xml`；
每份 `report.json` 都携带溯源块（git commit、Prompt 版本+内容哈希、
数据集指纹、模型、用例源码指纹）。

## 架构总览

```text
┌────────────────────────── 测试套件层 ──────────────────────────┐
│  llm（质量） rag（专项） agent（行为） security（红队） performance │
├────────────────────────── 测试支撑层 ──────────────────────────┤
│  RAGHarness(BM25+生成)  AgentHarness(工具环+护栏)  run_load(压测) │
│  Judge(LLM-as-Judge)  断言库(文本/JSON Schema/拒答/相似度)       │
├────────────────────────── 执行引擎层 ──────────────────────────┤
│  @test 注册表 → TestRunner(并发/超时/重试/fail-fast) → Reporter  │
│  (控制台 + JSON/Markdown/HTML/JUnit，失败按严重级排序)           │
├────────────────────────── 资产层 ──────────────────────────────┤
│  PromptManager(版本化/diff/注入扫描)  DatasetManager(数据集)     │
│  统一 LLMClient 抽象：Mock(脚本规则) / OpenAI兼容 / Anthropic    │
└────────────────────────────────────────────────────────────────┘
```

## 目录结构

```text
llm-qa/
├── config/                 # settings.yaml（阈值/并发/重试）+ providers.yaml（模型接入）
├── prompts/                # 版本化 Prompt 库（客服助手、RAG、裁判、安全分类器）
├── datasets/               # 测试数据集（golden QA、对抗样本、PII 金丝雀、RAG 语料、压测载荷）
├── src/llmqa/
│   ├── config.py           # Pydantic 配置模型，支持环境变量展开
│   ├── clients/            # LLMClient 抽象 + mock/openai_compat/anthropic + 连接池
│   ├── core/               # 用例模型、注册表、运行器、报告器、指标、负载生成器
│   ├── assertors/          # 确定性断言 + JSON Schema + LLM-as-Judge
│   ├── prompts/            # PromptManager + PromptScanner（注入扫描）
│   ├── datasets/           # 数据集加载器（YAML/JSON/CSV）
│   ├── harnesses/          # RAGHarness / AgentHarness
│   ├── suites/             # 五大内置测试套件（llm/rag/agent/security/performance）
│   ├── demo.py             # 内置演示套件
│   └── cli.py              # llmqa 命令行
├── tests/                  # 框架自身单元测试（pytest）
├── examples/               # 自定义套件示例
├── scripts/                # 门禁与工具脚本：gate.py（严重级门禁）、lint_ids.py、
│                           # gen_catalog.py、ast_diff.py + PowerShell 便捷脚本
├── docs/                   # 完整中文文档（含回归测试与运行溯源专题）
└── .github/workflows/      # CI 集成（分级门禁）
```

## 文档导航

**基础**：[架构设计](docs/architecture.md) · [快速上手](docs/getting-started.md) · [编写测试用例](docs/writing-tests.md) · [Prompt 管理](docs/prompt-management.md)

**测试域**：[RAG 专项](docs/rag-testing.md) · [Agent](docs/agent-testing.md) · [安全红队](docs/security-testing.md) · [性能](docs/performance-testing.md)

**回归平台**：[回归测试](docs/regression-testing.md)（命名基线 / compare 判定引擎 / 指标策略） · [运行溯源](docs/provenance.md)（provenance / 内容哈希 / 实验可重建） · [CI 集成与门禁](docs/ci-integration.md)

**目录**：[测试目录](docs/test-catalog.md)

## 设计原则

1. **可复现优先**：一切测试以确定性 Mock 先行验证测试逻辑，再对真实模型验收；随机性通过 temperature=0 与固定数据集控制。
2. **契约驱动**：用例只依赖 `LLMClient` / `TestContext` 等稳定接口，供应商变化不影响用例。
3. **测试即资产**：Prompt、数据集、阈值全部外置为版本化 YAML，可评审、可审计、可回滚。
4. **失败可行动**：FAIL（被测对象缺陷）/ ERROR（基础设施问题）分离、严重级分级、证据与指标随报告输出。
5. **安全默认开启**：安全套件纳入默认发现范围；Prompt 发布前必须通过扫描（`llmqa prompts scan`）。
