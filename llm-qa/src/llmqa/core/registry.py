"""测试用例注册表：@test 装饰器 + 包发现。"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from llmqa.core.models import Severity


class SkipTest(Exception):
    """在用例中抛出即标记为 SKIP（如：缺少 API Key 的环境）。"""
    def __init__(self, reason: str = "环境不满足，跳过"):
        super().__init__(reason)


@dataclass
class TestCaseDef:
    """一个已注册用例的静态定义，由 @test 装饰器填充，运行期只读。"""
    fn: Callable
    id: str
    suite: str
    name: str
    description: str
    tags: frozenset[str]
    severity: Severity
    timeout: float | None
    retries: int | None
    skip: bool
    cost: int = 1   # 预估 LLM 调用次数（成本单位），供 --max-cost 预算控制
    module: str = ""
    qualname: str = field(default="")

    def matches(self, *, suites: set[str] | None, tags: set[str] | None,
                exclude_tags: set[str] | None, min_severity: Severity | None) -> bool:
        """判断用例是否命中全部过滤条件；条件为空表示不过滤，各条件取"与"关系。"""
        if suites and self.suite not in suites:
            return False
        if tags and not (tags <= self.tags):
            # 要求的标签必须是用例标签的子集，即全部命中才通过。
            return False
        if exclude_tags and (exclude_tags & self.tags):
            return False
        if min_severity and self.severity < min_severity:
            # Severity.__ge__ 已重载，此处按等级阈值过滤低级别用例。
            return False
        return True


_REGISTRY: dict[str, TestCaseDef] = {}


def test(
    id: str | None = None,
    *,
    suite: str = "default",
    name: str | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
    severity: Severity = Severity.MEDIUM,
    timeout: float | None = None,
    retries: int | None = None,
    skip: bool = False,
    cost: int = 1,
) -> Callable:
    """用例注册装饰器。

    用法::

        @test(id="llm-fmt-001", suite="llm", tags=("smoke", "format"), severity=Severity.HIGH)
        async def case(ctx: TestContext) -> None:
            ...  # 断言失败时 raise AssertionFailed；跳过时 raise SkipTest
    """
    def decorator(fn: Callable) -> Callable:
        def _default_name(fn: Callable) -> str:
            # 未显式指定 name 时，取 docstring 首行作为可读名，回退到函数名。
            if fn.__doc__:
                first = fn.__doc__.strip().splitlines()[0].strip()
                if first:
                    return first
            return fn.__name__
        # 默认 id 用"套件/模块.函数名"，模块取短名，避免导入路径差异导致 id 不稳定。
        case_id = id or f"{suite}/{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
        _REGISTRY[fn.__qualname__] = TestCaseDef(
            fn=fn, id=case_id, suite=suite,
            name=name or _default_name(fn),
            description=description or (fn.__doc__ or "").strip(),
            tags=frozenset(tags), severity=severity,
            timeout=timeout, retries=retries, skip=skip, cost=cost,
            module=fn.__module__, qualname=fn.__qualname__,
        )
        return fn
    return decorator


def get_registered_cases() -> list[TestCaseDef]:
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """清空注册表，主要用于测试隔离，避免跨用例残留。"""
    _REGISTRY.clear()


def discover(packages: list[str]) -> list[TestCaseDef]:
    """导入指定包以触发 @test 注册，返回全部已注册用例。"""
    for pkg in packages:
        # 仅靠 import 的副作用触发 @test 装饰器注册，无需遍历文件。
        importlib.import_module(pkg)
    return get_registered_cases()
