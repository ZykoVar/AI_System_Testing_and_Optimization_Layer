"""安全红队测试套件：提示注入、越狱、提示词窃取、PII 泄露、有害内容、数据外泄与混淆绕过。"""

# 显式导入各子模块以触发 @test 装饰器注册：discover 只 import 包本身，需在此级联导入
from . import (  # noqa: F401
    exfiltration,
    harmful_content,
    injection_direct,
    injection_indirect,
    jailbreak_resistance,
    obfuscation,
    pii_leakage,
    prompt_extraction,
)

__all__ = [
    "injection_direct",
    "injection_indirect",
    "jailbreak_resistance",
    "prompt_extraction",
    "pii_leakage",
    "harmful_content",
    "exfiltration",
    "obfuscation",
]
