import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from .base import BaseConfigs
from .tools import ToolCallConfig, ToolChoice
from .variants import VariantConfigs
from .writers import write_json_schema

T_fn = TypeVar("T_fn")
T = TypeVar("T")


class FunctionConfigType(str, Enum):
    """
    Enumeration of function configuration types.
    """

    CHAT = "chat"
    JSON = "json"


class FunctionConfig(BaseModel, Generic[T_fn]):
    """
    Base class for function configurations, including common fields such as system, user,
    and assistant schemas (as pointers to BaseModel subclasses) and corresponding templates.

    Attributes:
        type (FunctionConfigType): The type of function configuration.
        system_schema (Optional[Type[BaseModel]]): A reference to a BaseModel subclass used for the system schema.
        user_schema (Optional[Type[BaseModel]]): A reference to a BaseModel subclass used for the user schema.
        assistant_schema (Optional[Type[BaseModel]]): A reference to a BaseModel subclass used for the assistant schema.
        example_system_template (Optional[str]): An example template for system prompts.
        example_user_template (Optional[str]): An example template for user prompts.
        example_assistant_template (Optional[str]): An example template for assistant prompts.
    """

    type: T_fn

    name: str

    config_dir: Path = Field(default=Path("config/"))

    system_schema: Optional[Dict[str, Any]] = None
    user_schema: Optional[Dict[str, Any]] = None
    assistant_schema: Optional[Dict[str, Any]] = None

    variants: VariantConfigs

    class Config:
        extra = "forbid"

    @field_validator("system_schema", "user_schema", "assistant_schema", mode="before")
    @classmethod
    def validate_schema(
        cls, value: Union[str, Dict[str, Any]], values: ValidationInfo
    ) -> Dict[str, Any]:
        """Allow system_schema and output_schema to be either a file path or a BaseModel subclass."""
        if isinstance(value, Dict):
            return value  # Already a BaseModel subclass

        schema_path = values.data["config_dir"] / value
        if not schema_path.is_file():
            raise ValueError(f"Schema file not found: {schema_path}")

        with schema_path.open("r", encoding="utf-8") as f:
            schema_dict = json.load(f)

        return schema_dict

    @field_serializer("type")
    def serialize_type(self, value: FunctionConfigType) -> str:
        return value.value

    @field_serializer("system_schema")
    def serialize_system_schema(
        self, value: Optional[Type[BaseModel]]
    ) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.name}/system_schema.json"

    @field_serializer("user_schema")
    def serialize_user_schema(self, value: Optional[Type[BaseModel]]) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.name}/user_schema.json"

    @field_serializer("assistant_schema")
    def serialize_assistant_schema(
        self, value: Optional[Type[BaseModel]]
    ) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.name}/assistant_schema.json"

    @field_serializer("name")
    def serialize_name(self, value: str) -> None:
        return None

    def write(self, function_dir: Path):
        if self.system_schema:
            write_json_schema(function_dir / "system_schema.json", self.system_schema)
        if self.user_schema:
            write_json_schema(function_dir / "user_schema.json", self.user_schema)
        if self.assistant_schema:
            write_json_schema(
                function_dir / "assistant_schema.json", self.assistant_schema
            )


class FunctionConfigChat(FunctionConfig[Literal[FunctionConfigType.CHAT]]):
    """
    Function configuration for chat-based responses.

    Inherits common fields from FunctionConfig and adds chat-specific fields.
    """

    type: Literal[FunctionConfigType.CHAT] = Field(default=FunctionConfigType.CHAT)

    # Chat-specific fields.
    tools: Optional[List[str]] = None
    tool_choice: Optional[ToolChoice] = None
    parallel_tool_calls: Optional[bool] = None

    @field_serializer("tool_choice")
    def serialize_tool_choice(self, value: Optional[ToolChoice]) -> Optional[str]:
        if value is None:
            return value
        return value.value


class FunctionConfigJson(FunctionConfig[Literal[FunctionConfigType.JSON]]):
    """
    Function configuration for JSON-formatted responses.

    Inherits common fields from FunctionConfig and adds JSON-specific fields.
    """

    type: Literal[FunctionConfigType.JSON] = Field(default=FunctionConfigType.JSON)

    # JSON-specific field: a pointer to a BaseModel subclass defining the output schema.
    output_schema: Dict[str, Any]

    implicit_tool_call_config: Optional[ToolCallConfig] = None

    @field_validator(
        "system_schema",
        "user_schema",
        "assistant_schema",
        "output_schema",
        mode="before",
    )
    @classmethod
    def validate_schema(
        cls, value: Union[str, Dict[str, Any]], values: ValidationInfo
    ) -> Dict[str, Any]:
        """Allow system_schema and output_schema to be either a file path or a BaseModel subclass."""
        return super().validate_schema(value, values)

    @field_serializer("output_schema")
    def serialize_output_schema(self, value: Type[BaseModel]) -> str:
        return f"functions/{self.name}/output_schema.json"

    def write(self, function_dir: Path):
        super().write(function_dir)
        write_json_schema(function_dir / "output_schema.json", self.output_schema)


class FunctionConfigs(BaseConfigs[Union[FunctionConfigChat, FunctionConfigJson]]):
    """
    Container for TensorZeroFunctionConfig objects, acting like a dictionary mapping
    function names to their respective TensorZeroFunctionConfig.
    """

    @model_validator(mode="before")
    @classmethod
    def convert_dicts_to_configs(
        cls,
        values: Dict[
            str, Union[Dict[str, Any], FunctionConfigChat, FunctionConfigJson]
        ],
    ) -> Dict[str, Union[FunctionConfigChat, FunctionConfigJson]]:
        """Convert dictionaries to the correct FunctionConfig type before validation."""
        converted: Dict[str, Union[FunctionConfigChat, FunctionConfigJson]] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                # Determine which model to use based on available keys
                if value["type"] == "json":
                    converted[key] = FunctionConfigJson(name=key, **value)
                elif value["type"] == "chat":  # Example key for FunctionConfigChat
                    converted[key] = FunctionConfigChat(name=key, **value)
                else:
                    raise ValueError(f"Unknown function config format for key: {key}")
            else:
                converted[key] = value  # Already a valid config object

        return converted

    def write(self, functions_dir: Path):
        for function_name, function_config in self:
            function_dir = functions_dir / function_name
            function_dir.mkdir(exist_ok=True)
            function_config.write(function_dir)
            function_config.variants.write(function_dir)
