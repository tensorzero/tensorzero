"""
Deprecated type aliases for backward compatibility.

This module contains deprecated types that have been renamed.
Import from the main tensorzero module instead of using the old names directly.
"""

import warnings

from typing_extensions import Any, deprecated

from .generated_types import (
    InferenceFilter,
    InferenceFilterAnd,
    InferenceFilterBooleanMetric,
    InferenceFilterFloatMetric,
    InferenceFilterNot,
    InferenceFilterOr,
    InferenceFilterTag,
    InferenceFilterTime,
    InputContentBlock,
    InputContentBlockFile,
    InputContentBlockRawText,
    InputContentBlockTemplate,
    InputContentBlockText,
    InputContentBlockThought,
    InputContentBlockToolCall,
    InputContentBlockToolResult,
    InputContentBlockUnknown,
    StoredInputContentBlock,
    StoredInputContentBlockFile,
    StoredInputContentBlockRawText,
    StoredInputContentBlockTemplate,
    StoredInputContentBlockText,
    StoredInputContentBlockThought,
    StoredInputContentBlockToolCall,
    StoredInputContentBlockToolResult,
    StoredInputContentBlockUnknown,
)
from .generated_types import (
    ProviderExtraBody as _ProviderExtraBody,
)
from .generated_types import (
    ProviderExtraBodyDelete as _ProviderExtraBodyDelete,
)
from .generated_types import (
    ProviderExtraHeader as _ProviderExtraHeader,
)
from .generated_types import (
    ProviderExtraHeaderDelete as _ProviderExtraHeaderDelete,
)
from .tensorzero import (
    DICLConfig,
    DICLOptimizationConfig,
    RenderedSample,
)
from .types import (
    WorkflowEvaluationRunEpisodeResponse,
    WorkflowEvaluationRunResponse,
)

# DEPRECATED: use RenderedSample instead
RenderedStoredInference = RenderedSample


# CAREFUL: deprecated
class DiclOptimizationConfig:
    def __new__(cls, *args: Any, **kwargs: Any):
        warnings.warn(
            "Please use `DICLOptimizationConfig` instead of `DiclOptimizationConfig`. In a future release, `DiclOptimizationConfig` will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DICLOptimizationConfig(*args, **kwargs)


# CAREFUL: deprecated alias
DiclConfig = deprecated("Use DICLConfig instead")(DICLConfig)

# CAREFUL: deprecated aliases for InputMessageContent* -> InputContentBlock*
InputMessageContent = InputContentBlock
"""Deprecated: Use InputContentBlock instead."""
InputMessageContentFile = deprecated("Use InputContentBlockFile instead")(InputContentBlockFile)
InputMessageContentRawText = deprecated("Use InputContentBlockRawText instead")(InputContentBlockRawText)
InputMessageContentTemplate = deprecated("Use InputContentBlockTemplate instead")(InputContentBlockTemplate)
InputMessageContentText = deprecated("Use InputContentBlockText instead")(InputContentBlockText)
InputMessageContentThought = deprecated("Use InputContentBlockThought instead")(InputContentBlockThought)
InputMessageContentToolCall = deprecated("Use InputContentBlockToolCall instead")(InputContentBlockToolCall)
InputMessageContentToolResult = deprecated("Use InputContentBlockToolResult instead")(InputContentBlockToolResult)
InputMessageContentUnknown = deprecated("Use InputContentBlockUnknown instead")(InputContentBlockUnknown)

# CAREFUL: deprecated aliases for StoredInputMessageContent* -> StoredInputContentBlock*
StoredInputMessageContent = StoredInputContentBlock
"""Deprecated: Use StoredInputContentBlock instead."""
StoredInputMessageContentFile = deprecated("Use StoredInputContentBlockFile instead")(StoredInputContentBlockFile)
StoredInputMessageContentRawText = deprecated("Use StoredInputContentBlockRawText instead")(
    StoredInputContentBlockRawText
)
StoredInputMessageContentTemplate = deprecated("Use StoredInputContentBlockTemplate instead")(
    StoredInputContentBlockTemplate
)
StoredInputMessageContentText = deprecated("Use StoredInputContentBlockText instead")(StoredInputContentBlockText)
StoredInputMessageContentThought = deprecated("Use StoredInputContentBlockThought instead")(
    StoredInputContentBlockThought
)
StoredInputMessageContentToolCall = deprecated("Use StoredInputContentBlockToolCall instead")(
    StoredInputContentBlockToolCall
)
StoredInputMessageContentToolResult = deprecated("Use StoredInputContentBlockToolResult instead")(
    StoredInputContentBlockToolResult
)
StoredInputMessageContentUnknown = deprecated("Use StoredInputContentBlockUnknown instead")(
    StoredInputContentBlockUnknown
)

# CAREFUL: deprecated aliases for ProviderExtra* -> ModelProviderExtra*
ProviderExtraBody = deprecated("Use ModelProviderExtraBody instead")(_ProviderExtraBody)
ProviderExtraBodyDelete = deprecated("Use ModelProviderExtraBodyDelete instead")(_ProviderExtraBodyDelete)
ProviderExtraHeader = deprecated("Use ModelProviderExtraHeader instead")(_ProviderExtraHeader)
ProviderExtraHeaderDelete = deprecated("Use ModelProviderExtraHeaderDelete instead")(_ProviderExtraHeaderDelete)

# CAREFUL: deprecated aliases for DynamicEvaluation* -> WorkflowEvaluation*
DynamicEvaluationRunResponse = WorkflowEvaluationRunResponse
"""Deprecated: Use WorkflowEvaluationRunResponse instead."""
DynamicEvaluationRunEpisodeResponse = WorkflowEvaluationRunEpisodeResponse
"""Deprecated: Use WorkflowEvaluationRunEpisodeResponse instead."""

# CAREFUL: deprecated aliases for inference filters
InferenceFilterTreeNode = InferenceFilter
"""Deprecated: Use InferenceFilter instead."""


@deprecated("Deprecated; use InferenceFilterFloatMetric instead. This alias will be removed in a future version.")
class FloatMetricFilter(InferenceFilterFloatMetric):
    """Deprecated: Use InferenceFilterFloatMetric instead."""

    pass


@deprecated("Deprecated; use InferenceFilterBooleanMetric instead. This alias will be removed in a future version.")
class BooleanMetricFilter(InferenceFilterBooleanMetric):
    """Deprecated: Use InferenceFilterBooleanMetric instead."""

    pass


@deprecated("Deprecated; use InferenceFilterTag instead. This alias will be removed in a future version.")
class TagFilter(InferenceFilterTag):
    """Deprecated: Use InferenceFilterTag instead."""

    pass


@deprecated("Deprecated; use InferenceFilterTime instead. This alias will be removed in a future version.")
class TimeFilter(InferenceFilterTime):
    """Deprecated: Use InferenceFilterTime instead."""

    pass


@deprecated("Deprecated; use InferenceFilterAnd instead. This alias will be removed in a future version.")
class AndFilter(InferenceFilterAnd):
    """Deprecated: Use InferenceFilterAnd instead."""

    pass


@deprecated("Deprecated; use InferenceFilterOr instead. This alias will be removed in a future version.")
class OrFilter(InferenceFilterOr):
    """Deprecated: Use InferenceFilterOr instead."""

    pass


@deprecated("Deprecated; use InferenceFilterNot instead. This alias will be removed in a future version.")
class NotFilter(InferenceFilterNot):
    """Deprecated: Use InferenceFilterNot instead."""

    pass
