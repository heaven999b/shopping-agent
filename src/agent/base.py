"""Base utilities for Claude-powered agent stages."""
from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

CLAUDE_MODEL = "claude-opus-4-6"


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text response."""
    # Try ```json ... ``` blocks first
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        return json.loads(match.group(1))
    # Try raw { ... }
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response:\n{text[:300]}")


def call_claude(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    output_model: Type[T],
    use_thinking: bool = False,
) -> T:
    """
    Call claude-opus-4-6 and parse the response into a Pydantic model.

    Args:
        client: Anthropic client
        system: System prompt
        user: User message
        output_model: Pydantic model class to parse the JSON response into
        use_thinking: Whether to enable adaptive thinking for complex stages
    """
    schema_hint = json.dumps(output_model.model_json_schema(), indent=2)
    full_system = (
        f"{system}\n\n"
        f"Respond with a JSON object that exactly matches this schema:\n"
        f"{schema_hint}\n\n"
        "Output only the JSON object, no additional text."
    )

    kwargs: dict[str, Any] = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "system": full_system,
        "messages": [{"role": "user", "content": user}],
    }
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    response = client.messages.create(**kwargs)

    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    data = extract_json(text)
    return output_model(**data)


def mock_call(output_model: Type[T], **overrides: Any) -> T:
    """Return a zeroed-out mock instance for testing without API calls."""
    defaults: dict[str, Any] = {}
    for field_name, field_info in output_model.model_fields.items():
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        if annotation is bool or annotation == "bool":
            defaults[field_name] = False
        elif annotation is float or annotation == "float":
            defaults[field_name] = 0.5
        elif annotation is int or annotation == "int":
            defaults[field_name] = 0
        elif annotation is str or annotation == "str":
            defaults[field_name] = "mock"
        elif origin is list:
            defaults[field_name] = []
        elif origin is dict:
            defaults[field_name] = {}
        elif field_info.default is not None:
            defaults[field_name] = field_info.default
    defaults.update(overrides)
    return output_model(**defaults)
