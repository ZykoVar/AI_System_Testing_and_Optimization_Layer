# 回归测试（Regression Testing）

llm-qa 的回归能力是一等公民，不是 CI 的附属品：
运行 → 基线 → 对比 → 判定 → 门禁，每一步都有独立命令与语义。

## 1. 概念链

```text
Prompt（版本+哈希） → Test（id+源码指纹） → Run（report.json + provenance）
  → Baseline（命名基线） → Compare（判定引擎） → Regression Gate（退出码）
```

## 2. 命名基线（Baseline）

基线是**测试资产**：把一次全绿运行"冻结"为后续对比的锚点，
支持多基线并存（production / staging / security / performance）。

```powershell
llmqa report baseline-set <run_id> --name production
llmqa report baseline-set <run_id> --name security
llmqa report baseline-list          # 全部基线 + commit + 创建人/时间
llmqa report baseline-show --name production
llmqa report compare --baseline --baseline-name production
```

基线文件 `reports/baselines/<name>.json` 为富元数据：
git commit、Prompt 版本+内容哈希、数据集指纹、创建人与时间——
可随仓库提交实现"团队共享基线"。

推荐工作流：发布窗口全绿 → 登记基线 → 后续每次运行
`compare --baseline` → 退出码 1 拦截回归。

## 3. 对比命令

```powershell
llmqa report list                                # 历史运行一览
llmqa report compare <runA> <runB>               # 任意两次运行（A=旧基线）
llmqa report compare --last                      # 最近两次
llmqa report compare --baseline --baseline-name production
```

产出 `reports/compare/*.{md,json}`；退出码 1 = 存在回归，0 = 无回归。

## 4. 判定引擎语义

### 4.1 判定排序

```text
PASS(3) > SKIP(2) > FAIL(1) > ERROR(0)
A→B 分数下降 = regression；上升 = improvement
```

### 4.2 SKIP 分类与中性语义

`skip_reason` 区分跳过原因：

| skip_reason | 含义 | 对比语义 |
| --- | --- | --- |
| budget | 成本预算耗尽 | **中性**（未执行 ≠ 变差） |
| fail_fast | 前置高危失败 | **中性** |
| intentional | 有意跳过（如缺 API Key） | 计回归（覆盖丢失） |

典型收益：预算受限运行 vs 全量基线 → 133 个跳过计中性而非假回归。

### 4.3 指标策略（Metric Policy）

字面 diff 会误报：judge 分数 8.1→8.0 是噪声，8.1→7.1 才是回归。
`config/metrics_policy.yaml` 声明方向与容差：

```yaml
metrics:
  judge_score: {direction: higher_better, tolerance: 0.3}
  p95_ms: {direction: lower_better, tolerance: 20%}   # 相对基线百分比
  mean_recall_at_k: {direction: higher_better, tolerance: 0.05}
```

判定：容差内 → 不显著；越界且方向劣化 → regression；方向改善 → improvement；
未声明策略的指标 → 仅记录 drift。新增数值指标时应同步在此登记。

### 4.4 结果方向总表

| direction | 含义 |
| --- | --- |
| regression | 判定劣化，或指标按策略显著劣化 |
| improvement | 反向变好 |
| metric_drift | 仅记录的指标变化（未声明策略/中性方向） |
| message_change | 同为失败但失败消息变化 |
| neutral | budget/fail_fast 跳过 |
| unchanged | 完全一致 |

## 5. 重试分桶（判定之前的净化）

只有基础设施故障会重试：LLMError(429/408/425/5xx) 与 httpx 网络层异常；
代码缺陷（TypeError/KeyError 等）不重试，立即判 ERROR 并在消息标注分类。
`outcome.retries_used` 记录重试次数，flaky 可见。

## 6. 与 CI 的关系

- 严重级门禁（`scripts/gate.py`）与回归门禁（`compare` 退出码）**串联**；
- 流水线编排与排期见 [ci-integration.md](ci-integration.md)；
- 溯源信息（commit/哈希）让回归可定位到具体变更，见 [provenance.md](provenance.md)。
