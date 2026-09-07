"""用例 id 唯一性与命名规范 lint（CI 静态门禁）。

检查项：
1. id 全局唯一（注册表以 qualname 为键，跨模块重复 id 不会被自动发现）；
2. id 只含小写字母/数字/连字符；
3. id 前缀与所属套件名一致（如 security 套件的用例应以 sec- 开头）。
用法：python scripts/lint_ids.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmqa.core.registry import discover  # noqa: E402
from llmqa.suites import DEFAULT_PACKAGES  # noqa: E402

_ID_RE = re.compile(r"^[a-z0-9-]+$")

# 套件 → id 前缀约定（新增套件时在此登记）
_SUITE_PREFIX = {"llm": "llm-", "rag": "rag-", "agent": "agt-",
                 "security": "sec-", "performance": "perf-"}


def main() -> int:
    cases = discover(DEFAULT_PACKAGES)
    problems: list[str] = []
    seen: dict[str, str] = {}
    for c in cases:
        if c.id in seen:
            problems.append("id 重复: {} 出现在 {} 与 {}".format(
                c.id, seen[c.id], c.module))
        seen[c.id] = c.module
        if not _ID_RE.match(c.id):
            problems.append("id 含非法字符: {}".format(c.id))
        prefix = _SUITE_PREFIX.get(c.suite)
        if prefix and not c.id.startswith(prefix):
            problems.append("id 前缀与套件不符: {} 属于 {} 套件".format(c.id, c.suite))
    if problems:
        print("发现 {} 个问题:".format(len(problems)))
        for p in problems:
            print("  - " + p)
        return 1
    print("lint_ids 通过: {} 个用例，id 唯一且命名规范".format(len(cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
