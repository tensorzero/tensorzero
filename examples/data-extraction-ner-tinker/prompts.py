"""System prompt and JSON parsing utilities for NER extraction."""

import json
import re
from typing import Optional

SYSTEM_PROMPT = """\
You are an assistant that is performing a named entity recognition task.
Your job is to extract entities from a given text.

The entities you are extracting are:
- people
- organizations
- locations
- miscellaneous other entities

Please return the entities in the following JSON format:

{
    "person": ["person1", "person2", ...],
    "organization": ["organization1", "organization2", ...],
    "location": ["location1", "location2", ...],
    "miscellaneous": ["miscellaneous1", "miscellaneous2", ...]
}"""

REQUIRED_KEYS = {"person", "organization", "location", "miscellaneous"}


def build_chat_messages(
    text: str, assistant_response: Optional[str] = None
) -> list[dict]:
    """Build a chat message list for the NER task.

    If assistant_response is provided, includes it as the final assistant turn
    (used for supervised training data).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    return messages


def _validate_ner_dict(d: dict) -> Optional[dict]:
    """Validate that d has the required NER schema. Returns d if valid, else None."""
    if not isinstance(d, dict):
        return None
    if not REQUIRED_KEYS.issubset(d.keys()):
        return None
    for key in REQUIRED_KEYS:
        if not isinstance(d[key], list):
            return None
        if not all(isinstance(item, str) for item in d[key]):
            return None
    return d


def parse_json_output(text: str) -> Optional[dict]:
    """Extract and parse a JSON NER dict from model output.

    Tries multiple strategies since open-weight models may wrap JSON in
    code fences, add explanatory text, etc.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        result = _validate_ner_dict(json.loads(text))
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from code fence (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            result = _validate_ner_dict(json.loads(fence_match.group(1).strip()))
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost braces
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = _validate_ner_dict(
                json.loads(text[brace_start : brace_end + 1])
            )
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    return None
