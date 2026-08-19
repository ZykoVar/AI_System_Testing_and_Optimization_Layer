"""Prompt 注入扫描：在发布前检测模板与用户变量中的风险模式。

覆盖技术（按风险分级）：
- 指令覆盖 / 角色切换（ignore previous instructions、你现在是…）
- 提示词窃取试探（system prompt、系统提示词）
- 分隔符欺骗（END OF INSTRUCTION 等）
- 混淆绕过（Unicode 全角、零宽字符、大小写混排）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llmqa.prompts.manager import PromptManager, PromptTemplate


@dataclass
class InjectionFinding:
    """单条注入命中：命中规则、风险等级、匹配片段与位置。"""

    rule: str
    risk: str          # HIGH / MEDIUM / LOW
    matched: str
    location: str      # template | variable:<name>


@dataclass
class ScanReport:
    """单个 Prompt 的扫描结果：携带最高风险等级与"是否干净"的便捷判断。"""

    prompt_id: str
    findings: list[InjectionFinding] = field(default_factory=list)

    @property
    def highest_risk(self) -> str:
        """全部命中中的最高风险等级；无命中时返回 NONE。"""
        # 用数值映射比较：字符串序 "LOW" < "MEDIUM" < "HIGH" 恰好与风险序相反。
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return max((f.risk for f in self.findings), key=lambda r: order[r], default="NONE")

    @property
    def clean(self) -> bool:
        """没有任何命中即视为干净。"""
        return not self.findings


# (规则名, 正则, 风险) —— 正则大小写不敏感
_RULES: list[tuple[str, str, str]] = [
    ("指令覆盖-英文", r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?)", "HIGH"),
    ("指令覆盖-中文", r"(忽略|无视)(所有|之前|以上|上述)?(的)?(指令|指示|要求|提示)", "HIGH"),
    ("角色切换", r"(you\s+are\s+now|你现在是|从现在起你是|扮演)", "HIGH"),
    ("系统越权", r"(act\s+as\s+the\s+system|你是系统|system\s+override)", "HIGH"),
    ("提示词窃取", r"(system\s+prompt|system\s+message|系统提示词|初始指令|原始指令)", "MEDIUM"),
    ("分隔符欺骗", r"(END|STOP|IGNORE)\s*(OF|ALL)?\s*(INSTRUCTIONS?|PROMPTS?)?", "MEDIUM"),
    ("开发者模式", r"(developer\s+mode|开发者模式|DAN\s+mode)", "MEDIUM"),
    ("输出格式劫持", r"(忽略\s*JSON\s*格式|不要输出\s*JSON)", "LOW"),
    ("零宽字符混淆", r"[\u200b\u200c\u200d\ufeff]", "LOW"),
    ("全角混淆", r"[\uff21-\uff3a\uff41-\uff5a]", "LOW"),
]


class PromptScanner:
    """基于正则规则库的注入扫描器；可注入自定义规则覆盖默认库。"""

    def __init__(self, rules: list[tuple[str, str, str]] | None = None):
        self.rules = rules or _RULES

    def scan_text(self, text: str, location: str = "template") -> list[InjectionFinding]:
        """对单段文本跑全部规则；每条规则至多一条命中，超长匹配被截断。"""
        findings: list[InjectionFinding] = []
        for rule, pattern, risk in self.rules:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                findings.append(InjectionFinding(
                    rule=rule, risk=risk, matched=m.group(0)[:80],  # 截断超长匹配，避免污染报告
                    location=location))
        return findings

    def scan_prompt(self, template: PromptTemplate,
                    variables: dict[str, str] | None = None) -> ScanReport:
        """扫描模板消息与用户变量；模板命中可被 scan_ignore 豁免，变量命中不可豁免。"""
        report = ScanReport(prompt_id=template.id)
        ignored = set(template.scan_ignore or [])
        for m in template.messages:
            for f in self.scan_text(m.content, "template"):
                if f.rule not in ignored:
                    report.findings.append(f)
        # 用户可控变量是注入主入口，必须扫描（不受模板豁免影响）
        for name, value in (variables or {}).items():
            if isinstance(value, str) and value.strip():   # 跳过非字符串与空串，避免误报
                report.findings.extend(self.scan_text(value, "variable:" + name))
        return report

    def scan_library(self, manager: PromptManager,
                     variables: dict[str, str] | None = None) -> dict[str, ScanReport]:
        """扫描全库所有模板，返回 {prompt_id: ScanReport} 映射。"""
        return {t.id: self.scan_prompt(t, variables) for t in manager.list()}
