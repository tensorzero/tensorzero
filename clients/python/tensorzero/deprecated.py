"""
Deprecated type aliases for backward compatibility.

This module contains deprecated types that have been renamed.
Import from the main tensorzero module instead of using the old names directly.
"""

from typing_extensions import deprecated

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

# 2026.1+

DiclConfig = deprecated("Use DICLConfig instead")(DICLConfig)
DiclOptimizationConfig = deprecated("Use DICLOptimizationConfig instead")(DICLOptimizationConfig)

RenderedStoredInference = deprecated("Use RenderedSample instead")(RenderedSample)

InferenceFilterTreeNode = InferenceFilter

FloatMetricFilter = deprecated("Use InferenceFilterFloatMetric instead")(InferenceFilterFloatMetric)
BooleanMetricFilter = deprecated("Use InferenceFilterBooleanMetric instead")(InferenceFilterBooleanMetric)
TagFilter = deprecated("Use InferenceFilterTag instead")(InferenceFilterTag)
TimeFilter = deprecated("Use InferenceFilterTime instead")(InferenceFilterTime)
AndFilter = deprecated("Use InferenceFilterAnd instead")(InferenceFilterAnd)
OrFilter = deprecated("Use InferenceFilterOr instead")(InferenceFilterOr)
NotFilter = deprecated("Use InferenceFilterNot instead")(InferenceFilterNot)

# 2026.2+

ProviderExtraBody = deprecated("Use ModelProviderExtraBody instead")(_ProviderExtraBody)
ProviderExtraBodyDelete = deprecated("Use ModelProviderExtraBodyDelete instead")(_ProviderExtraBodyDelete)
ProviderExtraHeader = deprecated("Use ModelProviderExtraHeader instead")(_ProviderExtraHeader)
ProviderExtraHeaderDelete = deprecated("Use ModelProviderExtraHeaderDelete instead")(_ProviderExtraHeaderDelete)

DynamicEvaluationRunResponse = deprecated("Use WorkflowEvaluationRunResponse instead")(WorkflowEvaluationRunResponse)
DynamicEvaluationRunEpisodeResponse = deprecated("Use WorkflowEvaluationRunEpisodeResponse instead")(
    WorkflowEvaluationRunEpisodeResponse
)

# 2026.3+

InputMessageContent = InputContentBlock
InputMessageContentFile = deprecated("Use InputContentBlockFile instead")(InputContentBlockFile)
InputMessageContentRawText = deprecated("Use InputContentBlockRawText instead")(InputContentBlockRawText)
InputMessageContentTemplate = deprecated("Use InputContentBlockTemplate instead")(InputContentBlockTemplate)
InputMessageContentText = deprecated("Use InputContentBlockText instead")(InputContentBlockText)
InputMessageContentThought = deprecated("Use InputContentBlockThought instead")(InputContentBlockThought)
InputMessageContentToolCall = deprecated("Use InputContentBlockToolCall instead")(InputContentBlockToolCall)
InputMessageContentToolResult = deprecated("Use InputContentBlockToolResult instead")(InputContentBlockToolResult)
InputMessageContentUnknown = deprecated("Use InputContentBlockUnknown instead")(InputContentBlockUnknown)

StoredInputMessageContent = StoredInputContentBlock
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
