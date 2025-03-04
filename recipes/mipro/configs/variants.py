from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from .base import BaseConfigs
from .writers import write_text_file


class JsonMode(str, Enum):
    """
    Enumeration for JSON response modes.
    """

    On = "on"
    Off = "off"
    Strict = "strict"
    ImplicitTool = "implicit_tool"


class RetryConfig(BaseModel):
    """
    Configuration model for defining retry behavior.

    Attributes:
        num_retries (int): Number of retries allowed.
        max_delay_s (int): Maximum delay in seconds between retries.
    """

    num_retries: int
    max_delay_s: int


class BaseVariantConfig(BaseModel):
    """
    Base configuration class for defining variant parameters.

    Attributes:
        weight (float): Weight assigned to the variant.
    """

    weight: float = Field(default=0)


class BaseClientConfig(BaseVariantConfig):
    """
    Base configuration for client settings, including retries, temperature,
    penalties, and token limits.

    Attributes:
        model (str): Name of the model.
        retries (RetryConfig): Retry settings.
        json_mode (JsonMode): Mode for JSON responses.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling probability.
        presence_penalty (Optional[float]): Presence penalty.
        frequency_penalty (Optional[float]): Frequency penalty.
        max_tokens (Optional[int]): Maximum token limit.
        seed (Optional[int]): Random seed for deterministic behavior.
        weight (float): Weight assigned to the variant.
    """

    name: str
    function_name: str
    config_dir: Path = Field(default=Path("config/"))

    model: str
    retries: RetryConfig = Field(default=RetryConfig(num_retries=0, max_delay_s=10))
    json_mode: JsonMode = Field(default=JsonMode.On)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None

    @field_serializer("json_mode")
    def serialize_enum(self, value: Enum) -> str:
        return value.value

    @field_serializer("name")
    def serialize_name(self, value: str) -> None:
        return None

    @field_serializer("function_name")
    def serialize_function_name(self, value: str) -> None:
        return None


class BaseChatCompletionConfig(BaseClientConfig):
    """
    Configuration for chat completion models, including system, user, and assistant templates.

    Attributes:
        system_template (Optional[str]): Template for system messages.
        user_template (Optional[str]): Template for user messages.
        assistant_template (Optional[str]): Template for assistant responses.
        model (str): Name of the model.
        retries (RetryConfig): Retry settings.
        json_mode (JsonMode): Mode for JSON responses.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling probability.
        presence_penalty (Optional[float]): Presence penalty.
        frequency_penalty (Optional[float]): Frequency penalty.
        max_tokens (Optional[int]): Maximum token limit.
        seed (Optional[int]): Random seed for deterministic behavior.
        weight (float): Weight assigned to the variant.
    """

    system_template: Optional[str] = None
    user_template: Optional[str] = None
    assistant_template: Optional[str] = None

    model_config = {"extra": "forbid"}

    @field_validator(
        "system_template", "user_template", "assistant_template", mode="after"
    )
    @classmethod
    def validate_templates(
        cls, value: Union[str, None], values: ValidationInfo
    ) -> Optional[str]:
        if value is None:
            return value
        if ".minijinja" in value:
            file_path = values.data["config_dir"] / value
            if not file_path.exists():
                # Warning(f"Template could be a schema or file path: {value}")
                return value
            with open(file_path, "r") as f:
                return f.read()
        return value

    @field_serializer("system_template")
    def serialize_system_template(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.function_name}/{self.name}/system_template.minijinja"

    @field_serializer("user_template")
    def serialize_user_template(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.function_name}/{self.name}/user_template.minijinja"

    @field_serializer("assistant_template")
    def serialize_assistant_template(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return (
            f"functions/{self.function_name}/{self.name}/assistant_template.minijinja"
        )

    def write(self, variant_dir: Path):
        """
        Write template files to the specified directory.

        Args:
            variant_dir (Path): Directory where templates should be saved.
        """
        if self.system_template is not None:
            write_text_file(
                variant_dir / "system_template.minijinja", self.system_template
            )
        if self.user_template is not None:
            write_text_file(variant_dir / "user_template.minijinja", self.user_template)
        if self.assistant_template is not None:
            write_text_file(
                variant_dir / "assistant_template.minijinja", self.assistant_template
            )


class ChatCompletionConfig(BaseChatCompletionConfig):
    """
    Configuration class for chat completion models.

    Attributes:
        type (Literal["chat_completion"]): Specifies the type of configuration as chat completion.
        system_template (Optional[str]): Template for system messages.
        user_template (Optional[str]): Template for user messages.
        assistant_template (Optional[str]): Template for assistant responses.
        model (str): Name of the model.
        retries (RetryConfig): Retry settings.
        json_mode (JsonMode): Mode for JSON responses.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling probability.
        presence_penalty (Optional[float]): Presence penalty.
        frequency_penalty (Optional[float]): Frequency penalty.
        max_tokens (Optional[int]): Maximum token limit.
        seed (Optional[int]): Random seed for deterministic behavior.
        weight (float): Weight assigned to the variant.
    """

    type: Literal["chat_completion"] = Field(default="chat_completion")

    config_dir: Path = Field(default=Path("config/"))


class FuserConfig(BaseChatCompletionConfig):
    """
    Configuration for fusers.
    """


class MixtureOfNConfig(BaseVariantConfig):
    """
    Configuration for mixture of N.
    """

    type: Literal["experimental_mixture_of_n"] = Field(
        default="experimental_mixture_of_n"
    )
    timeout_s: float = Field(default=300)
    candidates: List[str]
    fuser: FuserConfig

    def write(self, variant_dir: Path):
        """
        Write template files to the specified directory.

        Args:
            variant_dir (Path): Directory where templates should be saved.
        """
        fuser_dir = variant_dir / "fuser"
        fuser_dir.mkdir(exist_ok=True)
        self.fuser.write(fuser_dir)


class EvaluatorConfig(BaseChatCompletionConfig):
    """
    Configuration for evaluators.
    """

    model_config = {"fields": {"weight": {"exclude": True}}}


class BestOfNConfig(BaseVariantConfig):
    """
    Configuration for best of N.
    """

    type: Literal["experimental_best_of_n_sampling"] = Field(
        default="experimental_best_of_n_sampling"
    )
    timeout_s: Optional[float] = 300
    candidates: List[str]
    evaluator: EvaluatorConfig

    def write(self, variant_dir: Path):
        """
        Write template files to the specified directory.

        Args:
            variant_dir (Path): Directory where templates should be saved.
        """
        evaluator_dir = variant_dir / "evaluator"
        evaluator_dir.mkdir(exist_ok=True)
        self.evaluator.write(evaluator_dir)


class DiclConfig(BaseClientConfig):
    """
    Configuration for dynamic in-context learning.
    """

    type: Literal["experimental_dynamic_in_context_learning"] = Field(
        default="experimental_dynamic_in_context_learning"
    )
    embedding_model: str
    k: int
    system_instructions: Optional[str] = None

    @field_serializer("system_instructions")
    def serialize_system_instructions(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return f"functions/{self.function_name}/{self.name}/system_instructions.txt"

    def write(self, variant_dir: Path):
        """
        Write template files to the specified directory.

        Args:
            variant_dir (Path): Directory where templates should be saved.
        """
        if self.system_instructions is not None:
            write_text_file(
                variant_dir / "system_instructions.txt", self.system_instructions
            )


VariantConfig = Union[ChatCompletionConfig, MixtureOfNConfig, BestOfNConfig, DiclConfig]


class VariantConfigs(BaseConfigs[VariantConfig]):
    """
    Container for VariantConfig objects, acting like a dictionary mapping
    function names to their respective VariantConfig.
    """

    @model_validator(mode="before")
    @classmethod
    def convert_dicts_to_configs(
        cls, values: Dict[str, Union[Dict[str, Any], ChatCompletionConfig]]
    ) -> Dict[str, ChatCompletionConfig]:
        """Convert dictionaries to the correct FunctionConfig type before validation."""
        converted: Dict[str, ChatCompletionConfig] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                # Determine which model to use based on available keys
                if value["type"] == "chat_completion":
                    converted[key] = ChatCompletionConfig(name=key, **value)
                else:
                    raise ValueError(f"Unknown function config format for key: {key}")
            else:
                converted[key] = value  # Already a valid config object

        return converted

    def write(self, function_dir: Path):
        """
        Write template files to the specified directory.

        Args:
            function_dir (Path): Base directory where templates should be saved.
        """
        for variant_name, variant_config in self:
            variant_dir = function_dir / variant_name
            variant_dir.mkdir(exist_ok=True)
            variant_config.write(variant_dir)
