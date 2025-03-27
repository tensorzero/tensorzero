import json
from pathlib import Path
from typing import Any, Dict


def write_json_schema(path: Path, schema_dict: Dict[str, Any]) -> None:
    """
    Writes a JSON schema to a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)


def write_text_file(path: Path, content: str) -> None:
    """
    Writes text content to a file, ensuring parent directories exist.

    Args:
        path (Path): Path to save the text file.
        content (str): Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
