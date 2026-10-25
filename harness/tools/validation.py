"""A small JSON-schema subset validator — no dependency.

Tools describe their arguments as JSON Schema. The harness only needs the subset
that tool definitions realistically use, and it needs *readable* errors that can
be handed back to the model.
"""

from __future__ import annotations

from typing import Any

_TYPE_NAMES = {
    "object": "an object",
    "array": "an array",
    "string": "a string",
    "integer": "an integer",
    "number": "a number",
    "boolean": "a boolean",
    "null": "null",
}


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: list[str] = []
    if schema.get("type", "object") != "object":
        return [f"tool schema must describe an object, got {schema.get('type')!r}"]
    if not isinstance(arguments, dict):
        return [f"arguments must be an object, got {_describe(arguments)}"]
    _check_object(schema, arguments, "arguments", problems)
    return problems


def _check_object(
    schema: dict[str, Any],
    value: dict[str, Any],
    path: str,
    problems: list[str],
) -> None:
    properties: dict[str, Any] = schema.get("properties") or {}
    for required in schema.get("required") or []:
        if required not in value:
            problems.append(f"{path}.{required} is required")
    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in properties:
                allowed = ", ".join(sorted(properties)) or "none"
                problems.append(f"{path}.{key} is not an allowed argument (allowed: {allowed})")
    for key, sub_schema in properties.items():
        if key in value:
            _check(sub_schema, value[key], f"{path}.{key}", problems)


def _check(schema: dict[str, Any], value: Any, path: str, problems: list[str]) -> None:
    expected = schema.get("type")
    if expected and not _matches_type(expected, value):
        problems.append(f"{path} must be {_TYPE_NAMES.get(expected, expected)}, got {_describe(value)}")
        return

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(option) for option in schema["enum"])
        problems.append(f"{path} must be one of {allowed}, got {value!r}")

    if expected == "object":
        _check_object(schema, value, path, problems)

    elif expected == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _check(items, item, f"{path}[{index}]", problems)
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path} must have at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path} must have at most {schema['maxItems']} items")

    elif expected in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path} must be <= {schema['maximum']}")

    elif expected == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path} must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            problems.append(f"{path} must be at most {schema['maxLength']} characters")


def _matches_type(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"boolean {str(value).lower()}"
    if isinstance(value, str):
        shown = value if len(value) <= 40 else value[:37] + "..."
        return f"string {shown!r}"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__} {value}"
    if isinstance(value, list):
        return f"array of {len(value)} items"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__
