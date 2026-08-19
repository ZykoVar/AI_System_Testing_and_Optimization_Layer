"""内置测试套件注册：llm / rag / agent / security / performance。"""
# discover() 会 import 这些包，触发各模块 @test 装饰器的注册副作用
DEFAULT_PACKAGES = [
    "llmqa.suites.llm",
    "llmqa.suites.rag",
    "llmqa.suites.agent",
    "llmqa.suites.security",
    "llmqa.suites.performance",
]

__all__ = ["DEFAULT_PACKAGES"]
