"""安全网：比对两个源码目录的 AST（剥离 docstring 后），
检测"只应加注释"的任务是否意外改动了代码逻辑。

用法：python scripts/_ast_diff.py <目录A> <目录B>
输出：列出非 docstring AST 有差异的文件；无差异则退出码 0。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """删除模块/函数/类的 docstring 节点（注释增强允许改动 docstring）。"""
    for node in ast.walk(tree):
        for holder in ("body", "orelse"):
            body = getattr(node, holder, None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                body.pop(0)
    return tree


def normalized_dump(path: Path) -> str:
    tree = strip_docstrings(ast.parse(path.read_text(encoding="utf-8")))
    return ast.dump(tree, include_attributes=False)


def main() -> int:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a_files = {p.relative_to(a_dir).as_posix(): p
               for p in a_dir.rglob("*.py") if "__pycache__" not in str(p)}
    b_files = {p.relative_to(b_dir).as_posix(): p
               for p in b_dir.rglob("*.py") if "__pycache__" not in str(p)}
    changed: list[str] = []
    for rel in sorted(a_files.keys() & b_files.keys()):
        if normalized_dump(a_files[rel]) != normalized_dump(b_files[rel]):
            changed.append(rel)
    if changed:
        print("检测到代码逻辑发生变化的文件（应仅有注释差异）:")
        for rel in changed:
            print("  - " + rel)
        return 1
    print(f"AST 比对通过：{len(a_files.keys() & b_files.keys())} 个文件仅有注释/docstring 差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
