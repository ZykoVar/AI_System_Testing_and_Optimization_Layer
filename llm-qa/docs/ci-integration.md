# CI 集成与门禁

> 回归基线、判定语义、运行溯源已拆分为独立文档：
> [regression-testing.md](regression-testing.md) 与 [provenance.md](provenance.md)。
> 本文只讲流水线编排与门禁接入。

## 1. 分层门禁策略

| 层级 | 触发 | 内容 | 门禁 |
| --- | --- | --- | --- |
| L0 静态 | 每次 push/PR | `python -m compileall`、`llmqa prompts validate/scan`、`python scripts/lint_ids.py`、pytest 自测 | 0 失败 |
| L1 冒烟 | 每次 push/PR | `llmqa run --tag smoke`（mock，全套件冒烟子集） | 0 FAIL/ERROR |
| L2 全量 mock | 每次 merge 到 main | 全部套件（mock 脚本态） | 0 FAIL/ERROR |
| L3 真实模型 | 定时（nightly）/ 发布前 | `--provider openai` 等，需 Secrets | 按严重级：CRITICAL/HIGH 失败即拦截；MEDIUM 告警 |

## 2. GitHub Actions 示例（仓库内置）

`.github/workflows/llm-qa.yml` 提供四个 job：

```yaml
jobs:
  static:      # L0：compileall + prompts 门禁 + id lint + pytest
  smoke:       # L1：mock 冒烟（--tag smoke）
  full-mock:   # L2：全量 mock 回归（main 分支）
  nightly:     # L3：真实模型（schedule 触发，需 OPENAI_API_KEY secret）
```

L3 内部是**双门禁闭环**（workflow 与本文档一一对应）：

```text
llmqa run --provider openai ... --soft      # 执行：退出码归属交给门禁步骤
  ↓ python scripts/gate.py reports           # 严重级门禁：CRITICAL/HIGH 拦截
  ↓ llmqa report baseline-ensure ...         # 基线引导：首跑自动登记，此后沿用已提交基线
  ↓ llmqa report compare --baseline ...      # 回归门禁：1=回归、2=基线不可对比
  ↓ publish / block
```

要点：
- L0/L1/L2 零成本、零密钥，任何 PR 都会跑；
- L3 只在 `schedule`（每夜）或 `workflow_dispatch`（手动发布前）运行；
- 固定 `TZ: UTC` 与 `PYTHONIOENCODING: utf-8`，报告时间戳跨环境可对齐；
- 基线是测试资产：`reports/baselines/` 已加入 .gitignore 豁免，随仓库提交共享；
- 报告 artifacts：`reports/` 整个目录上传（含 baselines/ 与 compare/），保留 30 天。

### 2.1 退出码归属（谁决定 job 成败）

`llmqa run` 默认语义是"任何 FAIL/ERROR → exit 1"（交互使用直观）；
CI 中改用 `--soft` 让执行不直接决定成败，判定权交给两个门禁步骤：
严重级由 `gate.py`（`--threshold` 控制，默认 HIGH：MEDIUM 仅告警不拦截），
回归由 `compare`（1=回归、2=基线不可对比、0=通过）。

## 3. 本地脚本（scripts/）

```text
scripts/gate.py           # 严重级门禁核心：解析最新 report.json 判定退出码（CI 实际执行它）
scripts/lint_ids.py       # 用例 id 唯一性与命名规范 lint（L0）
scripts/gen_catalog.py    # 从注册表自动生成 docs/test-catalog.md
scripts/ast_diff.py       # AST 比对安全网（注释-only 变更验证）
scripts/comment_survey.py # 注释覆盖率审计
scripts/run_demo.ps1      # PowerShell 便捷脚本：llmqa demo
scripts/run_smoke.ps1     # 冒烟（--tag smoke）
scripts/run_all_mock.ps1  # 全量 mock 回归
scripts/run_real.ps1      # 真实模型（需先设置 API Key）
```

（Makefile 提供同名目标，供 Linux/macOS 或 WSL 使用。）

## 4. 严重级门禁判定逻辑（scripts/gate.py）

```text
1. 读取 reports/最新/report.json
2. 统计 verdict==FAIL 或 ERROR 的用例
3. CRITICAL/HIGH 失败数 > 0 → 退出 1（拦截）
4. MEDIUM 失败 → 退出 2（告警，CI 可配置为不拦截）
5. 其余 → 退出 0
```

回归门禁（`llmqa report compare --baseline`，退出码 1 = 存在回归）与
严重级门禁**串联**：任一拦截 → 发布受阻。
判定语义（指标策略/SKIP 分类/中性）见 [regression-testing.md](regression-testing.md)。

## 5. 与其他系统集成

- **JUnit 消费者**（Jenkins/GitLab/Azure DevOps）：直接归档 `junit.xml`；
- **趋势看板**：从 `report.json` 提取 `metrics`（延迟分位、judge 分数、
  检索指标）入库画趋势，漂移阈值对齐 `config/metrics_policy.yaml`；
- **告警**：CRITICAL 失败触发即时通知；性能 P95 相对基线漂移 >20% 触发性能告警；
- **Prompt 变更门禁**：PR 中 `prompts/` 有改动时强制跑 L1 + 安全套件。

## 6. 推荐排期

```text
每次 PR：L0 + L1（3-5 分钟）
每次 merge：L2（10 分钟级，mock 全量）
每夜：L3 真实模型全量（成本可控时）或 --tag smoke 采样
每次发布：L3 全量 + Prompt scan + 安全套件
         → 全绿登记基线（baseline-set）
         → 后续运行 compare --baseline 回归门禁
```
