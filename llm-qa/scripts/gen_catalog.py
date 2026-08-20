"""从用例注册表自动生成 docs/test-catalog.md，保证目录与代码同步。

用法：python scripts/gen_catalog.py [输出路径，默认 docs/test-catalog.md]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmqa.core.registry import discover
from llmqa.suites import DEFAULT_PACKAGES

# 套件 ID → 中文展示名与说明，供生成的目录文档引用
SUITE_NAMES = {
    "llm": "LLM 功能与质量",
    "rag": "RAG 专项",
    "agent": "Agent",
    "security": "安全红队",
    "performance": "性能",
}

SUITE_DESC = {
    "llm": "格式合规、指令遵循、事实准确性、幻觉/拒答、一致性、多语言、语气、长度控制",
    "rag": "检索质量（recall/hit/MRR/precision）、分块、忠实性、相关性、引用、端到端",
    "agent": "工具选择与参数、多步规划、状态记忆、循环检测、预算与工具护栏",
    "security": "直接/间接注入、越狱、提示词窃取、PII 金丝雀、有害内容、数据外泄、混淆绕过",
    "performance": "延迟分位、TTFT、吞吐、并发扩展、成本与 token 效率、长上下文、限流",
}


def main() -> int:
    """发现全部用例并按套件分组，生成 Markdown 测试目录。"""
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "docs" / "test-catalog.md")
    # discover 触发各套件模块的 @test 注册，返回全部用例
    cases = discover(DEFAULT_PACKAGES)
    by_suite: dict[str, list] = defaultdict(list)
    for c in cases:
        by_suite[c.suite].append(c)

    lines = [
        "# 测试目录",
        "",
        "> 本文件由 scripts/gen_catalog.py 从用例注册表自动生成，请勿手改；",
        "> 修改用例后运行 python scripts/gen_catalog.py 同步。",
        "",
        f"共 **{len(cases)}** 个用例，覆盖 5 个套件。",
        "",
    ]
    for suite in sorted(by_suite):
        suite_cases = sorted(by_suite[suite], key=lambda c: c.id)
        lines += [
            f"## {SUITE_NAMES.get(suite, suite)}（{len(suite_cases)} 例）",
            "",
            SUITE_DESC.get(suite, ""),
            "",
            "| 用例 ID | 名称 | 严重级 | 标签 |",
            "| --- | --- | --- | --- |",
        ]
        for c in suite_cases:
            tags = ",".join(sorted(c.tags)) or "-"
            lines.append(f"| {c.id} | {c.name} | {c.severity.value} | {tags} |")
        lines.append("")
    # 整份文档一次性写出，保证原子性（避免半写状态被误读）
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {out_path}（{len(cases)} 个用例）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
