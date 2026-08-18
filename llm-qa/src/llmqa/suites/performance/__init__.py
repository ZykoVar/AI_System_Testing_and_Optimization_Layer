"""性能测试套件：延迟、吞吐、并发扩展、成本与 token 效率、限流行为。"""

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
