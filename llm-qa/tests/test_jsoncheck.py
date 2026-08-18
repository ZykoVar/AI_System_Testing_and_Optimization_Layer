"""JSON 解析与 Schema 校验自测。"""
import pytest

from llmqa.assertors import AssertionFailed, assert_json_schema, assert_json_valid

FENCE = chr(96) * 3


def test_parse_plain_json():
    assert assert_json_valid('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    text = "好的，结果如下：\n" + FENCE + "json\n{\"a\": 1}\n" + FENCE + "\n希望有帮助"
    assert assert_json_valid(text) == {"a": 1}


def test_parse_array():
    assert assert_json_valid("[1, 2, 3]") == [1, 2, 3]


def test_parse_invalid():
    with pytest.raises(AssertionFailed):
        assert_json_valid("这不是 JSON")


def test_schema_pass():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string", "minLength": 1},
                             "age": {"type": "integer", "minimum": 0}},
              "additionalProperties": False}
    assert_json_schema('{"name": "张三", "age": 30}', schema)


def test_schema_missing_required():
    schema = {"type": "object", "required": ["name"],
              "properties": {"name": {"type": "string"}}}
    with pytest.raises(AssertionFailed) as exc_info:
        assert_json_schema('{"age": 30}', schema)
    assert "缺少必填字段" in str(exc_info.value)


def test_schema_extra_field():
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "additionalProperties": False}
    with pytest.raises(AssertionFailed) as exc_info:
        assert_json_schema('{"name": "x", "evil": 1}', schema)
    assert "未声明" in str(exc_info.value)


def test_schema_type_mismatch():
    with pytest.raises(AssertionFailed):
        assert_json_schema('{"age": "30"}',
                           {"type": "object", "properties": {"age": {"type": "integer"}}})
