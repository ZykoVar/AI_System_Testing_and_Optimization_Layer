"""测试用例注册表：@test 装饰器 + 包发现。"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from llmqa.core.models import Severity


def _source_hash(fn: Callable, item: dict | None = None) -> str:
    """用例源码指纹（前 12 位 sha256）：同一 id 的内容变化在溯源中可见。

    - 普通用例：取函数源码字节；
    - 数据驱动生成的闭包：取父函数源码 + 序列化后的数据集记录（数据即用例身份的一部分）。
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        # 生成的闭包无源码，回退到父级信息；调用方应传入 item 一起参与指纹
        src = repr(fn)
    if item is not None:
        src += "\n" + json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


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
    source_hash: str = ""   # 用例源码指纹：Stable Test ID 的内容维度
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
    source_hash: str | None = None,
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
            # 普通用例直接取源码指纹；数据驱动生成用例由调用方显式传入
            source_hash=source_hash or _source_hash(fn),
            module=fn.__module__, qualname=fn.__qualname__,
        )
        return fn
    return decorator


def data_driven(
    dataset: str | dict,
    *,
    items_key: str = "items",
    suite: str,
    id_prefix: str,
    name_field: str = "name",
    severity: Severity | None = None,
    tags: tuple[str, ...] = (),
    cost: int = 1,
    timeout: float | None = None,
) -> Callable:
    """数据集驱动注册：数据集每条记录自动生成一个用例。

    - 用例 id = f"{id_prefix}-{序号:03d}"（按数据集条目顺序，顺序即稳定 id）；
    - 记录可选字段覆盖默认值：name / severity / tags / cost；
    - 被装饰函数签名：async def case(ctx: TestContext, item: dict) -> None，
      同一份断言逻辑作用于每条记录，样板收敛到框架层；
    - dataset 可为数据集名（import 时从仓库 datasets/ 加载）或已加载的 dict（测试用）。

    用法::

        @data_driven("adversarial/injections", suite="security", id_prefix="sec-inj")
        async def case_injection(ctx, item):
            resp = await refusing_client(ctx, item["mock_match"])...
            assert_refusal(resp.text)
    """
    def decorator(fn: Callable) -> Callable:
        items = _load_items(dataset, items_key)
        for idx, item in enumerate(items, 1):
            case_id = "{}-{:03d}".format(id_prefix, idx)
            # 记录级元数据优先，缺失时回退到装饰器默认值
            item_sev = (Severity(item["severity"].upper()) if item.get("severity")
                        else (severity or Severity.MEDIUM))
            item_tags = tuple(item.get("tags") or ()) or tags
            item_cost = int(item.get("cost", cost))
            record_name = item.get(name_field) or ""
            if not record_name:
                raise ValueError(
                    "数据集 {} 第 {} 条缺少名称字段 {}".format(dataset, idx, name_field))
            generated = _make_generated_case(fn, item, case_id)
            test(case_id, suite=suite, name=record_name,
                 description="数据集记录 {}".format(item.get("id", case_id)),
                 tags=item_tags, severity=item_sev, timeout=timeout,
                 cost=item_cost,
                 # 数据驱动用例身份 = 断言函数源码 + 数据集记录内容
                 source_hash=_source_hash(fn, item))(generated)
        return fn
    return decorator


def _make_generated_case(fn: Callable, item: dict, case_id: str) -> Callable:
    """包装被装饰函数：固定 item 参数并保证注册键（qualname）唯一。"""
    async def generated(ctx) -> None:
        return await fn(ctx, item)
    # 注册表以 __qualname__ 为键，必须唯一化，否则多条记录会互相覆盖
    unique = fn.__name__ + "_" + case_id.replace("-", "_")
    generated.__name__ = unique
    generated.__qualname__ = unique
    # module 归因到源套件文件，保证失败 traceback / 报告指向业务代码而非注册器
    generated.__module__ = fn.__module__
    return generated


def _load_items(dataset: str | dict, items_key: str) -> list[dict]:
    """加载数据集条目：dict 直接使用；字符串按仓库 datasets/<name>.yaml 解析。"""
    if isinstance(dataset, dict):
        data = dataset
    else:
        from llmqa.config import repo_root
        import yaml
        path = repo_root() / "datasets" / (dataset + ".yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get(items_key)
    if not isinstance(items, list) or not items:
        raise ValueError("数据集 {} 缺少条目列表 '{}'".format(dataset, items_key))
    return items


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
