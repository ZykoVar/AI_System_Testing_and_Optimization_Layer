
# CI 集成与门禁

## 1. 分层门禁策略

| 层级 | 触发 | 内容 | 门禁 |
| --- | --- | --- | --- |
| L0 静态 | 每次 push/PR | `python -m compileall`、`llmqa prompts validate/scan`、pytest 自测 | 0 失败 |
| L1 冒烟 | 每次 push/PR | `llmqa run --tag smoke`（mock，全套件冒烟子集） | 0 FAIL/ERROR |
| L2 全量 mock | 每次 merge 到 main | 全部套件（mock 脚本态） | 0 FAIL/ERROR |
| L3 真实模型 | 定时（nightly）/ 发布前 | `--provider openai` 等，需 Secrets | 按严重级：CRITICAL/HIGH 失败即拦截；MEDIUM 告警 |

## 2. GitHub Actions 示例（仓库内置）

`.github/workflows/llm-qa.yml` 提供四个 job：

```yaml
jobs:
  static:      # L0：compileall + prompts 门禁 + pytest
  smoke:       # L1：mock 冒烟（--tag smoke）
  full-mock:   # L2：全量 mock 回归（main 分支）
  nightly:     # L3：真实模型（schedule 触发，需 OPENAI_API_KEY secret）
```

要点：
- L0/L1/L2 零成本、零密钥，任何 PR 都会跑；
- L3 只在 `schedule`（每夜）或 `workflow_dispatch`（手动发布前）运行；
- 报告 artifacts：`reports/` 整个目录上传，保留 30 天；
- 严重级门禁：CI 中解析 `report.json`，对 CRITICAL/HIGH 失败
  直接 fail job；MEDIUM 及以下仅记录（用 `--soft` 实现）。

## 3. 本地脚本（Windows）

```powershell
scripts/run_demo.ps1       # llmqa demo
scripts/run_smoke.ps1      # 冒烟：全部套件 smoke 标签
scripts/run_all_mock.ps1   # 全量 mock 回归
scripts/run_real.ps1       # 真实模型（需先设置 API Key）
scripts/ci_gate.ps1        # 解析报告 JSON，按严重级判定门禁
```

（Makefile 提供同名目标，供 Linux/macOS 或 WSL 使用。）

## 4. 门禁判定逻辑（ci_gate.ps1 / Makefile gate）

```text
1. 读取 reports/最新/report.json
2. 统计 verdict==FAIL 或 ERROR 的用例
3. CRITICAL/HIGH 失败数 > 0 → 退出 1（拦截）
4. MEDIUM 失败 → 退出 2（告警，CI 可配置为不拦截）
5. 其余 → 退出 0
```

补充：运行间回归门禁可用 `llmqa report compare --last`（退出码 1 = 存在回归），
与严重级门禁串联：任一拦截 → 发布受阻。

### 4.1 回归基线（推荐发布流程）

```powershell
# 发布窗口全绿后，把该次运行登记为基线
llmqa report baseline-set <run_id>
# 后续每次运行与基线对比（退出码 1 = 存在回归）
llmqa report compare --baseline
```

基线文件存于 `reports/.baseline.json`，可随仓库提交实现"团队共享基线"。

### 4.2 运行溯源（provenance）

每次运行的 `report.json` 都携带溯源块：git commit（及工作区是否脏）、
Python 版本、时区、本次实际使用的 Prompt 版本清单（prompts_used）与
数据集清单（datasets_used）。compare 输出会展示两次运行的 commit 差异——
回答"这次回归发生在哪次代码变更上"。

## 5. 与其他系统集成

- **JUnit 消费者**（Jenkins/GitLab/Azure DevOps）：直接归档 `junit.xml`；
- **趋势看板**：从 `report.json` 提取 `metrics`（延迟分位、judge 分数、
  检索指标）入库画趋势；
- **告警**：CRITICAL 失败触发即时通知；性能 P95 相对基线漂移 >20% 触发性能告警；
- **Prompt 变更门禁**：PR 中 `prompts/` 有改动时强制跑 L1 + 安全套件。

## 6. 推荐排期

```text
每次 PR：L0 + L1（3-5 分钟）
每次 merge：L2（10 分钟级，mock 全量）
每夜：L3 真实模型全量（成本可控时）或 --tag smoke 采样
每次发布：L3 全量 + Prompt scan + 安全套件，报告人工评审后归档
```
