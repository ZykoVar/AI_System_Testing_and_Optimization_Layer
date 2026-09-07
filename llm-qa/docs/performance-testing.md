
# 性能测试

## 1. 测试什么

LLM 服务性能三要素：**延迟（体验）、吞吐（容量）、成本（预算）**。

| 维度 | 指标 | 用例 |
| --- | --- | --- |
| 端到端延迟 | P50/P90/P95/P99（毫秒） | perf-lat-001/002 |
| 首 token 延迟 | TTFT（流式） | perf-lat-003 |
| 吞吐 | RPS、token/s、错误率 | perf-thr-* |
| 扩展性 | 阶梯并发下延迟退化 | perf-sca-* |
| 成本 | 单请求成本、批量预算 | perf-cos-* |
| token 效率 | 输出上限、冗长度 | perf-tok-* |
| 长上下文 | 长载荷成功率与延迟 | perf-lc-* |
| 限流行为 | 429 传播、错误率、重试 | perf-rat-* |

## 2. 负载生成器

```python
from llmqa.core.load import run_load, run_ramp
from llmqa.clients import Message

payloads = ctx.datasets.load("performance/payloads")   # short/medium/long

stats = await run_load(
    lambda i: client.generate([Message.user(payloads["medium"])]),
    concurrency=8, count=50)

stats.latency["p95_ms"]        # 分位延迟
stats.errors / stats.error_rate
stats.requests_per_second / stats.tokens_per_second
stats.total_tokens / stats.mean_cost_usd
stats.error_messages           # 前 5 条错误

ramp = await run_ramp(request_fn, levels=[1, 2, 5, 10, 20], per_level=20)
```

特性：
- asyncio 并发，`per_request_timeout` 防挂起；
- 异常不中断压测，计入 `errors` 与 `error_messages`；
- 阈值外置在 `config/settings.yaml → thresholds`：
  `p95_latency_ms / p99_latency_ms / ttft_p95_ms / cost_per_request_usd`。

## 3. 阈值设定建议

| 场景 | 建议基准 |
| --- | --- |
| 实时对话（聊天） | P95 ≤ 2000ms，TTFT P95 ≤ 1500ms |
| 批量/离线任务 | P95 ≤ 10s，以吞吐优先 |
| 成本敏感 | 单请求成本 ≤ 预算值（按 pricing 估算） |
| 发布门禁 | P99 ≤ 2×P95 阈值，error_rate == 0 |

阈值的意义在**趋势与回归**：同一基线（模型版本/硬件/并发）下连续运行，
P95 漂移超过 20% 即为退化信号。

## 4. Mock 态与真实态的差异

- **mock 态**：默认延迟 2ms，用例验证统计管线（分位计算、错误率、token 统计）
  与断言逻辑的正确性；
- **真实态**：`llmqa run --suite performance --provider openai` 产生真实数字。
  注意：
  1. 成本：50 请求 × 500 token ≈ 少量费用，但请先小规模试探；
  2. 限流：商用 API 有 RPM/TPM 限制，`--concurrency` 别超过配额；
  3. 网络抖动：P99 容易受出口网络影响，对比时尽量同网络环境。

## 5. 长上下文与输出长度

`datasets/performance/payloads.yaml` 内置 short/medium/long 三档载荷；
扩展真实业务载荷（法律文书、代码库问答等）以获得业务相关数字。

输出长度影响延迟与成本，建议追加断言：
`resp.usage.completion_tokens ≤ 期望上限`。

## 6. 限流与韧性

```python
# 模拟 429：验证错误传播契约
try:
    await client.generate([...])
except LLMError as e:
    assert e.status == 429       # 状态码透传

# 部分失败：run_load 统计错误率（0.4 ≤ error_rate ≤ 0.6 之类）
# 恢复性：先限流后成功，验证上层重试语义
```

真实 Provider 下关注：退避重试是否生效、错误是否被正确分类
（限流 vs 鉴权 vs 超时），避免把 429 误当质量缺陷。
框架级重试分桶：429/5xx/网络层才重试，代码缺陷立即判 ERROR（见 docs/architecture.md）。

## 7. 报告与趋势

性能用例通过 `metrics` 把完整分位数据写进 `report.json`；
CI 可解析该文件生成趋势图（见 docs/ci-integration.md 的 artifact 用法）。

跨运行对比遵循**指标策略**（`config/metrics_policy.yaml`）：
p95 等延迟指标按相对基线 20% 容差判定——容差内不算回归，
越界且方向劣化才计回归；RPS/token 吞吐越高越好，错误率越低越好。
