
# RAG 专项测试

## 1. 测试什么

RAG 管线分两阶段，缺陷来源不同，必须分层测：

```text
[检索阶段]  分块(chunking) → 嵌入/索引 → 检索(retrieval)
[生成阶段]  上下文组装 → 生成(generation) → 引用(citation)
```

| 层 | 缺陷示例 | 对应指标/用例 |
| --- | --- | --- |
| 分块 | 语义被切断、块过大 | 块长/重叠/覆盖（rag-chk-*） |
| 检索 | 漏召回、错排序 | recall@k / hit@k / MRR / precision@k（rag-ret-*） |
| 生成 | 编造资料外内容 | 忠实性 Judge（rag-fai-*） |
| 生成 | 答非所问 | 相关性 Judge（rag-rel-*） |
| 引用 | 引用不存在/越界 | 引用格式与范围（rag-cit-*） |
| 整体 | 端到端质量 | golden 一致性/拒答/延迟（rag-e2e-*） |

## 2. RAGHarness 快速使用

```python
from llmqa.harnesses import RAGHarness, RAGCorpus, RetrievalQuery

# 1) 构建语料（8 篇虚构 Acme 商城文档）
corpus = RAGCorpus.from_dicts(ctx.datasets.load("rag/corpus")["documents"])

# 2) 挂接被测客户端（生成阶段）与托管 Prompt
harness = RAGHarness(corpus, scripted_or_real(ctx, rules=[...]),
                     prompt_manager=ctx.prompts, prompt_id="rag/answer")

# 3) 检索
result = await harness.retrieve("退货要几天？", k=4)
# result.chunks[].text / doc_id / chunk_id / title, result.scores

# 4) 生成（自动走 rag/answer 模板：只能依据参考资料作答）
resp = await harness.answer("退货要几天？", k=4)

# 5) 检索质量评估（同步，纯本地计算）
metrics = harness.evaluate_retrieval(
    [RetrievalQuery(**q) for q in ctx.datasets.load("rag/queries")["queries"]])
# metrics.mean_recall_at_k / mean_hit_at_k / mean_mrr / mean_precision_at_k
```

## 3. 指标定义

设查询 q 的相关文档集为 R（标注），检索返回 top-k 列表 L：

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| recall@k | 命中相关文档数 / 相关文档总数 | 召回完整性 |
| hit@k | 是否至少命中一个相关文档 | 二值命中 |
| MRR | 首个相关文档排名倒数（1/rank） | 排序质量 |
| precision@k | 相关块数 / k | 精度（按块） |

内置检索器是**纯 Python BM25-lite 词法检索**（中英混排分词），
结果确定可复现，用于验证评估管线与数据集质量；
接入生产向量库时，实现同名接口替换即可（见扩展点）。

## 4. 数据集标注规范

`datasets/rag/queries.yaml` 每条查询需标注：

```yaml
- id: rq-001
  query: 买了东西不想要了，几天内可以退货？
  relevant_doc_ids: [doc-return]        # 相关文档（必填）
  golden_answer: 7 天内可以无理由退货。 # 参考答案（用于一致性）
```

标注要点：
1. 覆盖单文档、跨文档（rq-007）、零相关（rq-008 拒答）三类；
2. `relevant_doc_ids` 必须准确，否则评估结论失真；
3. golden_answer 只陈述事实，便于 Judge/相似度双重校验。

## 5. 忠实性与相关性（LLM-as-Judge）

生成质量无法用关键词穷举，交给独立裁判模型：

```python
# 被测生成客户端（mock 分支：回复引用 [资料1] 且内容来自上下文）
gen_client = scripted_or_real(ctx, rules=[MockRule(
    match="退货", reply="[资料1] 支持 7 天无理由退货。")])
resp = await harness.answer("退货要几天？")

# 裁判客户端独立脚本化
judge_client = ctx.providers.get_mock(rules=[MockRule(
    match="评分标准", reply='{"score": 9, "reasoning": "回答完全基于资料"}')])
await Judge(judge_client).assert_score(
    question="退货要几天？", answer=resp.text,
    context=harness.build_context(await harness.retrieve("退货要几天？")),
    criteria="答案必须完全基于参考资料，不得编造",
    min_score=ctx.settings.thresholds.judge_min_score)
```

## 6. 端到端用例模式

```python
queries = ctx.datasets.load("rag/queries")["queries"]
for q in queries[:5]:
    resp = await harness.answer(q["query"])
    assert_contains(resp.text, *q["golden_answer"].split("，")[:2])  # 关键事实词
```

未知问题（rq-008，语料无答案）必须断言模型明确拒答而非编造：

```python
resp = await harness.answer("你们支持分期付款吗？")
assert_contains(resp.text, "无法回答", "未提及", "不知道", any_of=True)
```

## 7. 扩展点

- **换真实检索后端（可插拔 Retriever 协议）**：实现 `async retrieve(query, k) -> RetrievalResult`
  并传入 `RAGHarness(corpus, client, retriever=...)`，检索与评估自动委托外部后端，
  离线 CI 仍保留内置 BM25-lite。示例（Chroma/LlamaIndex 集成）：

```python
class VectorStoreRetriever:
    def __init__(self, index): self.index = index          # 生产向量库索引
    async def retrieve(self, query, k):
        hits = await self.index.asimilarity_search_with_score(query, k=k)
        return RetrievalResult(query=query,
            chunks=[Chunk(chunk_id=m.id, doc_id=m.metadata["doc_id"], text=m.page_content)
                    for m, _ in hits],
            scores=[round(s, 4) for _, s in hits])

harness = RAGHarness(corpus, client, retriever=VectorStoreRetriever(index))
```

- **换嵌入模型**：在 `retrieve` 前后插入 embedding 打分融合；
- **新语料**：向 `datasets/rag/corpus.yaml` 增文档，同步补 queries 标注；
- **新指标**：nDCG、MAP 可在 `evaluate_retrieval` 的 `per_query` 数据上计算。
