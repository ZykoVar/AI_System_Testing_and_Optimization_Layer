# 项目

## llm-qa —— AI System Testing & Optimization Layer

定位：**不做"最强的 Eval 产品"，做把 Eval/Observability 产品、Agent Runtime 与 CI 串起来的测试与优化层**——外部平台负责能力（LiteLLM 网关 / Ragas 指标 / LangSmith·Langfuse 轨迹），本项目负责工程体系（统一测试模型、Agent 行为断言 DSL、回归语义、质量门禁）。

```text
llm_learn/                     # 仓库根（GitHub: Simon214hz/llm_learn）
├── llm-qa/                    # 测试框架本体：五大套件 + 回归平台（详见 llm-qa/README.md）
├── docs/                      # 仓库级文档与架构图（diagrams/）
└── tools/                     # 辅助工具（archify/ 架构图生成）

能力清单（对应 llm-qa/ 内部实现，并非仓库目录名）：
- Prompt 管理       版本化模板库、渲染校验、diff、发布前注入扫描、A/B 测试
- LLM 评测          格式合规、指令遵循、事实准确性（Golden QA + Judge）、幻觉/一致性
- RAG 专项          检索质量（recall/hit/MRR/precision）、分块、忠实性、引用、端到端
- Agent 测试        工具选择、参数 Schema、多步规划、会话记忆、循环与预算护栏
                    + AgentTrajectory 轨迹统一模型 + 行为断言 DSL（平台无关）
- 安全红队          注入/越狱/提示词窃取/PII 金丝雀/有害内容/数据外泄（41 例）
- 性能测试          延迟分位、TTFT、吞吐、并发扩展、成本、限流韧性
- 回归平台          命名基线、Baseline 兼容性判定、compare 判定引擎、退出码门禁
- 基线管理          基线即测试资产：commit/哈希/创建人随基线存档，团队共享
- 指标策略          指标漂移策略（方向+容差）：judge ±0.3、延迟 ±20% 等
- 运行溯源          每次运行记录 commit/内容哈希/模型/用例指纹（可追溯的实验条件）
- 成熟工具后端      LiteLLM 网关（可用适配）、Langfuse 轨迹（完整映射）、Ragas（接入骨架）
- CI 质量门禁       四层门禁 + L3 双门禁闭环（严重级 gate.py + 回归 compare）
```

**141 个内置用例 · 127 个框架自测 · 双态运行（mock 离线 / 真实模型一键切换）**
**协作定位：成熟工具当引擎（LiteLLM/Ragas/Langfuse/向量库），回归语义（基线/容差/门禁/溯源）归本项目**

## 快速开始

```powershell
cd llm-qa
pip install -e ".[dev]"
llmqa demo                      # Mock 演示（无需 API Key）
llmqa run --list                # 141 个内置用例
llmqa run                       # 全量回归（mock 态）
llmqa report baseline-set <run_id> --name production   # 登记基线
llmqa report compare --baseline --baseline-name production  # 回归对比
```

## 文档

- 入口：[llm-qa/README.md](./llm-qa/README.md)
- 专题：[llm-qa/docs/](./llm-qa/docs/)（架构 / 快速上手 / 写用例 / 回归测试 / 运行溯源 / CI 等 12 篇）
