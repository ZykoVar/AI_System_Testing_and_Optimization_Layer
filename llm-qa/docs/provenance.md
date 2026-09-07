# 运行溯源（Provenance）

目标：任何一次运行都能回答——
**用什么代码、什么数据集、什么 Prompt、什么模型，跑出了什么结果**。

## 1. 记录了什么

每份 `report.json`（schema_version=2）携带 provenance 块：

```json
{
  "schema_version": 2,
  "provenance": {
    "git_commit": "a9999b0d...", "git_dirty": false,
    "python_version": "3.10.11", "platform": "...", "timezone": "UTC",
    "model": {"provider": "openai", "kind": "openai_compat", "model": "gpt-4o-mini"},
    "prompts_used": [{"id": "support-agent", "version": 2, "content_hash": "38938c12b7230670"}],
    "datasets_used": [{"name": "adversarial/injections", "content_hash": "dbdbcdb38864c8af"}],
    "test_identity": {"sec-inj-001": "ed92e76c9e35"}
  }
}
```

| 字段 | 含义 | 采集方式 |
| --- | --- | --- |
| git_commit / git_dirty | 代码基线（脏=有未提交修改） | 运行时 `git rev-parse`，非 git 环境降级为空 |
| python_version / platform / timezone | 运行环境 | 运行时采集 |
| model | 被测模型接入信息 | 本次 provider 配置 |
| prompts_used | 实际渲染的 Prompt：id + version + **content_hash** | PromptManager 在 render 时记录 |
| datasets_used | 实际加载的数据集：name + **content_hash** | DatasetManager 在 load 时记录 |
| test_identity | 每个 case_id → 用例源码指纹 | 注册时对源码 sha256；数据驱动用例 = 断言函数 + 数据集记录 |

## 2. 为什么 version 不够，要 hash

`version` 只是 YAML 里的命名约定，**不防内容被修改**；
`content_hash` 是文件字节级 sha256（前 16 位），任何改动都会改变指纹。
同理 `test_identity` 让"用例 id 没变但代码被改过"在对比中可见。

## 3. 怎么用

1. **回归定位**：`llmqa report compare` 输出两次运行的 commit 差异——
   "这次回归发生在哪次代码变更上"；
2. **实验可重建**：报告 + 基线文件（commit + 哈希）即可复现实验条件；
3. **审计**：基线文件记录创建人与时间，Prompt/数据集变更可追责；
4. **团队共享**：基线文件随仓库提交，全队对比同一锚点。

## 4. 当前局限与演进

- 数据集没有独立版本号字段——当前靠 content_hash + git commit 间接版本化，
  未来可给数据集 YAML 增加显式 `version` 字段并纳入溯源；
- 模型参数（temperature/max_tokens）为每调用参数，未逐条记录——
  需要时可在 ModelUsage 中扩展默认参数块；
- git 信息采集宽容降级：非 git 环境运行不阻断测试，但溯源不完整。
