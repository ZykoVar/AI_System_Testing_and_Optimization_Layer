"""断言库：确定性文本/JSON 断言与 LLM-as-Judge 软断言。"""
# 断言库的公共 API 面：用例里只需 from llmqa.assertors import assert_contains 即可使用。
from llmqa.assertors.base import AssertionFailed
from llmqa.assertors.jsoncheck import assert_json_schema, assert_json_valid, parse_json, validate
from llmqa.assertors.judge import Judge, JudgeVerdict
from llmqa.assertors.text import (
    assert_char_length,
    assert_contains,
    assert_in_language,
    assert_matches,
    assert_not_contains,
    assert_not_refusal,
    assert_refusal,
    assert_similarity,
    assert_word_count,
    is_refusal,
    language_ratio,
    text_similarity,
)

__all__ = [
    "AssertionFailed",
    "Judge",
    "JudgeVerdict",
    "assert_char_length",
    "assert_contains",
    "assert_in_language",
    "assert_json_schema",
    "assert_json_valid",
    "assert_matches",
    "assert_not_contains",
    "assert_not_refusal",
    "assert_refusal",
    "assert_similarity",
    "assert_word_count",
    "is_refusal",
    "language_ratio",
    "parse_json",
    "text_similarity",
    "validate",
]
