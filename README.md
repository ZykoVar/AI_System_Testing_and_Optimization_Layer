# llm_learn

LLM/Agent 学习与工程实践工作区。

## 项目

- [llm-qa](./llm-qa) —— 企业级 LLM/Agent 质量保障测试框架（Prompt 管理 / LLM 质量 / RAG 专项 / Agent / 安全红队 / 性能测试）

## llm-qa 快速开始

```powershell
cd llm-qa
pip install -e ".[dev]"
llmqa demo          # Mock 演示（无需 API Key）
llmqa run --list    # 136 个内置用例
llmqa run           # 全量回归（mock 态）
```

详见 [llm-qa/README.md](./llm-qa/README.md) 与 [llm-qa/docs/](llm-qa/docs/)。
