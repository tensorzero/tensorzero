import json
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel


def write_pydantic_schema(path: Path, schema_model: type[BaseModel]) -> None:
    """
    Writes a Pydantic model's JSON schema to a file.

    Args:
        path (Path): Path to save the JSON schema.
        schema_model (BaseModel): Pydantic model to serialize.
    """
    # Convert to dictionary
    schema_dict = schema_model.model_json_schema()

    # Write the JSON schema to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)


def write_output_schema(path: Path, schema_model: type[BaseModel]) -> None:
    """
    Writes an output schema as a JSON file using `to_strict_json_schema`.

    Args:
        path (Path): Path to save the JSON schema.
        schema_model (BaseModel): Pydantic model to serialize.
    """
    schema_dict = to_strict_json_schema(schema_model)

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
