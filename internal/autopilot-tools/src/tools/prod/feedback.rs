//! Feedback tool for calling TensorZero feedback endpoint.

use durable_tools::{SideInfo, ToolMetadata};

// Parameters for the feedback tool (visible to LLM).
pub struct FeedbackToolParams {}

// Side information for the feedback tool (hidden from LLM).
pub struct FeedbackToolSideInfo {}

impl SideInfo for FeedbackToolSideInfo {}

/// Tool for calling TensorZero feedback endpoints.
///
/// This tool allows autopilot to make feedback calls, optionally using
/// a historical config snapshot for reproducibility.

pub struct FeedbackTool;

impl ToolMetadata for FeedbackTool {
    type SideInfo = FeedbackToolSideInfo;
    type Output = FeedbackResponse;
    type LlmParams = FeedbackToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("feedback")
    }

    fn description() -> std::borrow::Cow<'static, str> {
        Cow::Borrowed(
            "Call TensorZero feedback endpoint. Optionally use a config snapshot hash to use historical configuration.",
        )
    }
}

impl SimpleTool for FeedbackTool {}
