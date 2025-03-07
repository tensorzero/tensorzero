import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from .base import BaseConfigs


class ToolConfig(BaseModel):
    """
    Configuration for a tool including its name, description, and associated parameters.
    """

    config_dir: Path = Field(default=Path("config/"))

    description: str
    parameters: Dict[str, Any]
    name: str

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(
        cls, value: Union[str, Dict[str, Any]], values: ValidationInfo
    ) -> Dict[str, Any]:
        if isinstance(value, Dict):
            return value
        schema_path = values.data["config_dir"] / value
        if schema_path.is_file():
            with open(schema_path, "r") as f:
                schema_dict = json.load(f)
                parameters_dict: Dict[str, Any] = {}
                parameters_dict["type"] = schema_dict.get("type")
                parameters_dict["properties"] = schema_dict.get("properties")
                parameters_dict["required"] = schema_dict.get("required")
                parameters_dict["additionalProperties"] = schema_dict.get(
                    "additionalProperties", False
                )
                return parameters_dict
        else:
            raise ValueError(f"Schema file not found: {schema_path}")

    @field_serializer("name")
    def serialize_name(self, value: str) -> None:
        return None

    @field_serializer("parameters")
    def serialize_parameters(self, value: Dict[str, Any]) -> str:
        return f"tools/{self.name}.json"


class ToolChoice(str, Enum):
    """
    Enumeration of possible tool selection strategies.
    """

    AUTO = "auto"
    ANY = "any"


class ToolCallConfig(BaseModel):
    """
    Configuration for calling tools, including the available tools,
    tool choice strategy, and whether tool calls can be parallel.
    """

    tools_available: Optional[List[ToolConfig]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None
    parallel_tool_calls: Optional[bool] = False

    @field_serializer("tool_choice")
    def serialize_tool_choice(self, value: Optional[ToolChoice]) -> Optional[str]:
        if value is None:
            return None
        return value.value


class ToolConfigs(BaseConfigs[ToolConfig]):
    """
    Container for ToolConfig objects, acting like a dictionary mapping tool names to ToolConfig.
    """

    @model_validator(mode="before")
    @classmethod
    def convert_dicts_to_configs(
        cls, values: Dict[str, Union[Dict[str, Any], ToolConfig]]
    ) -> Dict[str, ToolConfig]:
        """Convert dictionaries to the correct FunctionConfig type before validation."""
        converted: Dict[str, ToolConfig] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                converted[key] = ToolConfig(**value)
            else:
                converted[key] = value  # Already a valid config object

        return converted
