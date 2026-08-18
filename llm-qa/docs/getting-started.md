
# 快速上手

## 1. 环境要求与安装

- Python 3.10+
- 依赖：pydantic ≥ 2.5、PyYAML、httpx（`pip install -e .` 自动安装）

```powershell
cd D:\pycharm\llm_learn\llm-qa
pip install -e ".[dev]"     # dev 含 pytest
llmqa --help
```

## 2. 三分钟跑通演示

内置演示套件覆盖五个子系统的最小闭环（JSON 校验、注入防御、RAG 检索、
Agent 工具调用、延迟压测），全部基于 Mock，不需要任何 API Key：

```powershell
llmqa demo
```

输出示例（结尾）：

```text
[PASS] LOW      demo-003  演示：RAG 检索命中 (1ms)
[PASS] HIGH     demo-002  演示：直接注入防御 (1ms)
运行 5 个用例 | 通过 5 | 失败 0 | 错误 0 | 跳过 0 | 通过率 100.0%
报告: json → reports/<run_id>/report.json, ... html, junit
```

## 3. 运行内置套件

```powershell
# 列出全部内置用例
llmqa run --list

# 按套件
llmqa run --suite llm
llmqa run --suite security
llmqa run --suite rag --suite agent

# 按标签 / 严重级 / 组合过滤
llmqa run --tag smoke
llmqa run --suite security --severity HIGH      # 只跑 HIGH 及以上
llmqa run --suite llm --exclude-tag judge       # 排除慢的裁判用例

# 并发与超时
llmqa run --suite performance --concurrency 16 --timeout 120
```

## 4. 接入真实模型

### 4.1 OpenAI 兼容协议（OpenAI / DeepSeek / vLLM / Ollama）

```powershell
$env:OPENAI_API_KEY = "sk-..."
llmqa run --provider openai --suite llm --tag smoke
```

本地 vLLM/Ollama 无需密钥：

```powershell
llmqa run --provider local_vllm --suite rag
```

### 4.2 Anthropic

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
llmqa run --provider anthropic --suite llm
```

> 注意：Anthropic 适配器暂不支持 function calling，Agent 套件请用
> OpenAI 兼容 Provider 或 Mock。

### 4.3 新增 Provider

编辑 `config/providers.yaml` 增加配置块即可（`kind` 支持
`mock / openai_compat / anthropic`）；自定义协议请实现
`LLMClient` 并在 `clients/factory.py` 注册。

## 5. 双态运行：Mock 与真实模型的一致性

内置用例遵循 `scripted_or_real(ctx, rules=[...])` 模式：

- **默认 provider 为 mock**（CI、离线开发）：用例使用规则脚本模拟被测模型行为，
  验证的是测试逻辑与断言本身是否成立，结果确定可复现；
- **指定真实 provider**：同一用例直连真实模型，脚本不再生效，用例立即成为
  真实验收测试。

因此推荐工作流：

```text
1. 离线（mock）开发与调试用例，保证 100% 通过 —— 测试逻辑正确；
2. CI 常驻 mock 冒烟 —— 防框架/用例回归；
3. 发布窗口切换真实模型跑全套 —— 模型质量验收；
4. 将真实运行报告存档对比 —— 版本间质量漂移追踪。
```

## 6. 报告解读

- `reports/<run_id>/report.json`：机器可读完整数据（含 metrics/evidence）
- `reports/<run_id>/report.md`：评审用 Markdown 摘要
- `reports/<run_id>/report.html`：浏览器查看的彩色报告
- `reports/<run_id>/junit.xml`：CI 系统（Jenkins/GitLab/GitHub）直接消费

退出码：0 = 全部通过或仅跳过；1 = 存在 FAIL 或 ERROR。CI 中直接以退出码
作为门禁，或加 `--soft` 只做记录不拦截。

## 7. 常见问题

- **用例全部 SKIP？** 检查 `config/settings.yaml` 的 `default_provider`，
  以及对应 Provider 的 `api_key_env` 环境变量是否已设置。
- **Windows 控制台乱码？** 报告器会自动切换 UTF-8；终端本身不支持时可加
  `--no-color` 并重定向输出到文件。
- **压测想更真实？** 在 `MockRule(latency_ms=...)` 中调大模拟延迟，
  或直接 `--provider openai` 压真实端点（注意成本与限流，先小并发）。
