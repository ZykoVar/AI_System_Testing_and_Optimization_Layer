"""单用例调试入口：在 IDE 中断点逐行执行异步用例。

为什么需要它：
    正常 `llmqa run` 会把用例丢进 asyncio.Semaphore 并发池、套上
    wait_for 超时、出错还会重试——断点停久了会被超时取消、多个用例
    会同时停在断点上、失败还会被重试，逐行调试体验很差。
    这个脚本直接 await 用例函数本身：无并发、无超时、无重试，
    断点可以想停多久停多久。

用法（在 llm-qa 目录，或从 PyCharm 直接 Run/Debug 本文件并改参数）：
    python scripts/debug_case.py --list               # 列出全部可调试的用例 id
    python scripts/debug_case.py demo-001             # 调试一个用例
    python scripts/debug_case.py demo-002 llm-len-001 # 调试多个
    python scripts/debug_case.py --tag length         # 调试某个标签下的全部用例

调试步骤（PyCharm）：
    1. 打开本文件或被测用例源码（如 src/llmqa/suites/llm/length_constraints.py），
       在函数体内点行号左侧打红点断点；
    2. 右键本文件 → Debug 'debug_case'，或菜单 Run → Debug；
    3. 命中断点后：F7=进入被调函数（含 await 的内部实现），
       F8=执行到下一行（被 await 的调用整体跑完），F9=继续到下一个断点；
    4. 左下 Debug 面板的 Debugger 标签里点眼睛图标（Evaluate Expression）
       可临时查看 resp / trace / ctx 等变量。
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from llmqa.clients import ClientPool
from llmqa.config import Settings, repo_root
from llmqa.core.models import TestContext
from llmqa.core.registry import discover, get_registered_cases
from llmqa.datasets import DatasetManager
from llmqa.prompts import PromptManager

# 与 src/llmqa/suites/__init__.py 的 DEFAULT_PACKAGES 保持一致
DEFAULT_PACKAGES = (
    "llmqa.suites.llm",
    "llmqa.suites.rag",
    "llmqa.suites.agent",
    "llmqa.suites.security",
    "llmqa.suites.performance",
)


def build_ctx() -> TestContext:
    """按真实运行路径构造一个 TestContext（配置/提示词/数据集/连接池齐全）。"""
    root = repo_root()
    settings = Settings.load(root / "config")
    pool = ClientPool(settings)
    return TestContext(
        run_id="debug",
        settings=settings,
        providers=pool,
        prompts=PromptManager(root / "prompts").load(),
        datasets=DatasetManager(root / "datasets"),
    )


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*", help="用例 id，如 demo-001 / llm-len-001")
    ap.add_argument("--tag", help="按标签调试该标签下的全部用例")
    ap.add_argument("--list", action="store_true", help="列出全部已注册用例")
    args = ap.parse_args(argv)

    # Windows 控制台默认 GBK，强制 UTF-8 以免中文输出乱码（与 demo.py 一致）
    from llmqa.core.reporter import _ensure_utf8_stdout

    _ensure_utf8_stdout()

    discover(list(DEFAULT_PACKAGES))  # 导入五套件 → @test 注册
    import llmqa.demo  # noqa: F401  demo 套件不在五套件内，单独注册

    cases = get_registered_cases()
    if args.list:
        for c in cases:
            print(f"{c.id:16s} {c.suite:10s} {c.name}")
        return 0

    picked = [c for c in cases if c.id in args.ids or (args.tag and args.tag in c.tags)]
    if not args.ids and not args.tag and not picked:
        # 不带参数时给个默认用例,右键直接 Debug 即可命中
        picked = [c for c in cases if c.id == "demo-001"]
        print("未指定用例,默认调试 demo-001(可在 Run/Debug 配置的 Parameters 里传 id)")
    if not picked:
        print("未找到用例。用 --list 查看可用 id。")
        return 1

    ctx = build_ctx()
    for c in picked:
        print(f"▶ 调试用例 {c.id}（{c.name}）—— 在函数体内打断点即可逐行执行")
        await c.fn(ctx)  # ← 直接 await：无并发、无超时、无重试
        print(f"✔ {c.id} 执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
