"""临时工具：统计各 Python 文件的注释覆盖（用后可删）。"""
import ast
from pathlib import Path

root = Path(".")
files = sorted(p for p in root.rglob("*.py")
               if "__pycache__" not in str(p) and ".venv" not in str(p))
rows = []
for p in files:
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len([l for l in lines if l.strip()])
    comments = sum(1 for l in lines if l.strip().startswith("#"))
    tree = ast.parse(text)
    has_module_doc = ast.get_docstring(tree) is not None
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    with_doc = sum(1 for d in defs if ast.get_docstring(d))
    rows.append((comments / max(1, total), p.as_posix(), total, comments,
                 with_doc, len(defs), has_module_doc))
rows.sort()
print("{:>6} {:>5} {:>5} {:>6} {:>5} {:>7}  文件".format(
    "注释比", "代码行", "注释行", "函数/类", "有doc", "模块doc"))
for ratio, path, total, comments, with_doc, ndefs, has_doc in rows:
    flag = " <<< 需增强" if (ratio < 0.08 or not has_doc
                             or (ndefs and with_doc / ndefs < 0.5)) else ""
    print(f"{ratio:6.1%} {total:5d} {comments:5d} {ndefs:6d} {with_doc:5d} {has_doc!s:>7}  {path}{flag}")
