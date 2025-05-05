from enum import Enum
from typing import Any, Dict, Union

from pydantic import BaseModel, field_serializer, model_validator

from .base import BaseConfigs


class MetricConfigType(str, Enum):
    """
    Enumeration of possible metric configuration types.
    """

    boolean = "boolean"
    float = "float"
    comment = "comment"
    demonstration = "demonstration"


class MetricConfigOptimize(str, Enum):
    """
    Enumeration of possible optimization strategies for a metric.
    """

    min = "min"
    max = "max"


class MetricConfigLevel(str, Enum):
    """
    Enumeration of levels at which a metric is applied.
    """

    inference = "inference"
    episode = "episode"


class MetricConfig(BaseModel):
    """
    Configuration for a metric including its type, optimization strategy, and level.
    """

    type: MetricConfigType
    optimize: MetricConfigOptimize
    level: MetricConfigLevel

    @field_serializer("type", "optimize", "level")
    def serialize_metric(
        self, value: Union[MetricConfigType, MetricConfigOptimize, MetricConfigLevel]
    ) -> str:
        return value.value


class MetricConfigs(BaseConfigs[MetricConfig]):
    """
    Container for MetricConfig objects, acting like a dictionary mapping metric names to MetricConfig.
    """

    @model_validator(mode="before")
    @classmethod
    def convert_dicts_to_configs(
        cls, values: Dict[str, Union[Dict[str, Any], MetricConfig]]
    ) -> Dict[str, MetricConfig]:
        """Convert dictionaries to the correct FunctionConfig type before validation."""
        converted: Dict[str, MetricConfig] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                converted[key] = MetricConfig(**value)
            else:
                converted[key] = value  # Already a valid config object

        return converted
