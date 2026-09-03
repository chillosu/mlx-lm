# Copyright © 2026 Apple Inc.
"""
Parser for the DeepSeek V4 DSML tool-call format.

The model emits tool calls inside a tool_calls block:

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter>
    <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

String parameters carry string="true" and are taken verbatim; all other
types carry string="false" and are JSON-decoded, falling back to the raw
text when the value is not valid JSON.
"""

import json
from typing import Any, Optional

import regex as re

DSML = "｜DSML｜"

_invoke_regex = re.compile(
    rf"<{DSML}invoke name=\"(?P<name>[^\"]*)\">\s*(?P<body>.*?)</{DSML}invoke>",
    re.DOTALL,
)
_param_regex = re.compile(
    rf"<{DSML}parameter name=\"(?P<key>[^\"]*)\" string=\"(?P<is_str>true|false)\">"
    rf"(?P<value>.*?)</{DSML}parameter>",
    re.DOTALL,
)


def _decode_value(value: str, is_str: str):
    if is_str == "true":
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _parse_single(match: re.Match) -> dict:
    arguments = {}
    for p in _param_regex.finditer(match.group("body")):
        arguments[p.group("key")] = _decode_value(p.group("value"), p.group("is_str"))
    return dict(name=match.group("name"), arguments=arguments)


def parse_tool_call(text: str, _: Optional[Any] = None):
    matches = list(_invoke_regex.finditer(text))
    if not matches:
        raise ValueError("No function provided.")
    if len(matches) == 1:
        return _parse_single(matches[0])
    return [_parse_single(m) for m in matches]


tool_call_start = f"<{DSML}tool_calls>"
tool_call_end = f"</{DSML}tool_calls>"
