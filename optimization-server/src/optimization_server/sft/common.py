import random
import typing as t
from abc import abstractmethod
from typing import Any, List, Tuple

from minijinja import Environment, TemplateError
from pydantic import BaseModel, Field
from tensorzero import System
from tensorzero.internal_optimization_server_types import Sample
from typing_extensions import TypedDict


class Model(BaseModel):
    displayName: str
    name: str
    provider: t.Literal["openai", "fireworks"]


class FineTuningRequest(BaseModel):
    kind: t.Literal["sft"]
    function: str
    # TODO - require nonempty
    metric: t.Optional[str]
    variant: str
    validationSplitPercent: int = Field(ge=0, le=100)
    maxSamples: int = Field(ge=10)
    threshold: float
    jobId: str
    model: Model


class ParsedInferenceExample(TypedDict):
    variant_name: str
    input: Any
    output: Any
    episode_id: str


T = t.TypeVar("T")


class ValidationError(Exception):
    pass


class BaseSFTJob(BaseModel):
    @abstractmethod
    async def poll(self) -> "BaseSFTJob": ...

    @abstractmethod
    def status(self) -> t.Any: ...


def render_message(content: t.Dict[str, Any], role: str, env: Environment) -> str:
    assert role in ["user", "assistant"], f"Invalid role: {role}"

    if content["type"] != "text":
        raise ValueError(f"Content block must be of type text: {content}")

    content = content["value"]

    if isinstance(content, str):
        return content
    else:
        return env.render_template(role, **content)


def try_template_system(
    sample: Sample, env: Environment
) -> t.Optional[t.Dict[str, str]]:
    system: t.Optional[System] = sample["input"].get("system")

    if system is None:
        return None
    elif isinstance(system, str):
        return {
            "role": "system",
            "content": system,
        }
    else:
        # TODO: add a 'has_template' to the minijinja python bindings
        try:
            # TODO: better error message when 'system' is a string and we have a template
            rendered_system = env.render_template("system", **system)
            return {
                "role": "system",
                "content": rendered_system,
            }

        except TemplateError as e:
            if e.kind == "TemplateNotFound":
                if not isinstance(system, str):
                    raise ValidationError(
                        "System message must be a string when not using templates"
                    )
                return {
                    "role": "system",
                    "content": system,
                }
            else:
                raise


def split_validation_data(
    inferences: List[T], validation_split_percent: float
) -> Tuple[List[T], List[T]]:
    """
    Randomize and split a list of inference examples into training and validation sets.

    Args:
        inferences: List of inference examples
        validation_split_percent: Percentage of data to use for validation (0-100)

    Returns:
        Tuple of (training_inferences, validation_inferences)
    """
    # Randomize order of 'inferences'
    inferences = list(inferences)
    random.shuffle(inferences)
    validation_split = validation_split_percent / 100

    split_index = (
        int(len(inferences) * (1 - validation_split))
        if validation_split > 0
        else len(inferences)
    )

    train_inferences = inferences[:split_index]
    val_inferences = inferences[split_index:] if validation_split_percent > 0 else []

    return train_inferences, val_inferences
