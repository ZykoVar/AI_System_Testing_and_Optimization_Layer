"""轻量 JSON Schema 校验（无 jsonschema 依赖）。

支持子集：type / properties / required / additionalProperties / items /
enum / const / minimum / maximum / minLength / maxLength / pattern / minItems / maxItems。
"""
from __future__ import annotations

import json
import re
from typing import Any

from llmqa.assertors.base import AssertionFailed

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict, "null": type(None)}


def _type_ok(instance: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if t == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if t == "array":
        return isinstance(instance, list)
    if t == "object":
        return isinstance(instance, dict)
    if t == "null":
        return instance is None
    return isinstance(instance, _TYPES.get(t, object))


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """返回违规描述列表；空列表表示通过。"""
    errors: list[str] = []
    t = schema.get("type")
    if t and not _type_ok(instance, t):
        errors.append("{}: 期望类型 {}，实际 {}".format(path, t, type(instance).__name__))
        return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("{}: 值不在枚举中: {}".format(path, schema["enum"]))
    if "const" in schema and instance != schema["const"]:
        errors.append("{}: 值不等于常量 {}".format(path, schema["const"]))
    if t == "object" and isinstance(instance, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errors.append("{}.{}: 缺少必填字段".format(path, req))
        if schema.get("additionalProperties") is False:
            for k in instance:
                if k not in props:
                    errors.append("{}.{}: 未声明的字段".format(path, k))
        for k, subschema in props.items():
            if k in instance:
                errors.extend(validate(instance[k], subschema, "{}.{}".format(path, k)))
    elif t == "array" and isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(instance):
                errors.extend(validate(item, items, "{}[{}]".format(path, i)))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("{}: 数组长度 {} 小于 minItems {}".format(path, len(instance), schema["minItems"]))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append("{}: 数组长度 {} 大于 maxItems {}".format(path, len(instance), schema["maxItems"]))
    if t == "string" and isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("{}: 字符串长度 {} 小于 minLength {}".format(path, len(instance), schema["minLength"]))
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append("{}: 字符串长度 {} 大于 maxLength {}".format(path, len(instance), schema["maxLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append("{}: 未匹配正则 {}".format(path, schema["pattern"]))
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("{}: {} 小于 minimum {}".format(path, instance, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("{}: {} 大于 maximum {}".format(path, instance, schema["maximum"]))
    return errors


def parse_json(text: str) -> Any:
    """宽松 JSON 解析：剥除代码围栏、截取首个 {...} 或 [...]。"""
    fence = chr(96) * 3
    cleaned = text.strip()
    cleaned = re.sub("^" + fence + "(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(fence + "$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == open_ch:
                depth += 1
            elif cleaned[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:i + 1])
    raise AssertionFailed("无法从回复中解析 JSON", evidence=[text[:500]])


def assert_json_valid(text: str, message: str | None = None) -> Any:
    try:
        return parse_json(text)
    except (json.JSONDecodeError, AssertionFailed) as e:
        raise AssertionFailed(message or "JSON 解析失败: {}".format(e), evidence=[text[:500]])


def assert_json_schema(text: str, schema: dict, message: str | None = None) -> Any:
    """解析 JSON 并校验 Schema，失败时抛出包含全部违规项的 AssertionFailed。"""
    obj = assert_json_valid(text)
    errors = validate(obj, schema)
    if errors:
        raise AssertionFailed(
            message or "JSON Schema 校验失败: " + "; ".join(errors[:8]),
            metrics={"schema_violations": len(errors)}, evidence=errors[:20])
    return obj
