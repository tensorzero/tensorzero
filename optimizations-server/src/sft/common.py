from typing import List, Any, Tuple, TypedDict
import typing as t
import random

from pydantic import BaseModel, Field


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
