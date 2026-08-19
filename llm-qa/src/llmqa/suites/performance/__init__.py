"""性能测试套件：延迟、吞吐、并发扩展、成本与 token 效率、限流行为。

子模块：
- latency:               端到端延迟分位与流式首 token 延迟（TTFT）
- throughput:            固定并发的错误率与吞吐统计
- concurrency_scaling:   阶梯并发下的稳定性与并行加速
- cost_efficiency:       单请求与批量成本预算
- token_efficiency:      token 上限、简洁性与输入 token 统计
- long_context:          长载荷下的成功与延迟退化
- rate_limit:            429 限流的故障注入与重试语义
"""

# 注册子模块以触发 @test 装饰器（discover 只 import 包，需在此显式导入各模块）
from llmqa.suites.performance import (  # noqa: F401
    concurrency_scaling,
    cost_efficiency,
    latency,
    long_context,
    rate_limit,
    throughput,
    token_efficiency,
)

__all__ = [
    "latency",
    "throughput",
    "concurrency_scaling",
    "cost_efficiency",
    "token_efficiency",
    "long_context",
    "rate_limit",
]
