"""断言库：确定性文本/JSON 断言与 LLM-as-Judge 软断言。"""
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
    "AssertionFailed", "assert_contains", "assert_not_contains", "assert_matches",
    "assert_word_count", "assert_char_length", "assert_similarity", "text_similarity",
    "assert_refusal", "assert_not_refusal", "is_refusal", "assert_in_language", "language_ratio",
    "assert_json_valid", "assert_json_schema", "parse_json", "validate",
    "Judge", "JudgeVerdict",
]
