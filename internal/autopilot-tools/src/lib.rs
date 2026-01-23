//! Tool definitions for TensorZero Autopilot.
//!
//! This crate provides tool traits and implementations for the autopilot system.
//!
//! # Overview
//!
//! - Re-exports tool traits from `durable-tools` for defining custom tools
//! - Provides production tools for autopilot operations
//! - Provides test tools (when `e2e_tests` feature is enabled) for testing the autopilot infrastructure
//!
//! # Production Tools
//!
//! - `InferenceTool` - Calls TensorZero inference endpoint, optionally with a historical config snapshot
//! - `FeedbackTool` - Submits feedback for inferences or episodes (comments, demonstrations, metrics)
//! - `CreateDatapointsTool` - Creates datapoints in a dataset
//! - `CreateDatapointsFromInferencesTool` - Creates datapoints from existing inferences
//! - `ListDatapointsTool` - Lists datapoints with filtering and pagination
//! - `GetDatapointsTool` - Gets specific datapoints by ID
//! - `UpdateDatapointsTool` - Updates existing datapoints
//! - `DeleteDatapointsTool` - Deletes datapoints by ID
//! - `LaunchOptimizationWorkflowTool` - Launches an optimization workflow (e.g., fine-tuning)
//! - `GetLatestFeedbackByMetricTool` - Gets the latest feedback ID for each metric for a target
//! - `GetFeedbackByVariantTool` - Gets feedback statistics (mean, variance, count) by variant for a function and metric
//! - `RunEvaluationTool` - Runs an evaluation on a dataset and returns statistics
//! - `ListInferencesTool` - Lists inferences with filtering and pagination
//!
//! # Test Tools (e2e_tests feature)
//!
//! When the `e2e_tests` feature is enabled, this crate provides several test tools:
//!
//! ## TaskTools
//! - `EchoTool` - Echoes back input message
//! - `SlowTool` - Sleeps for configurable duration
//! - `FailingTool` - Always returns an error
//! - `FlakyTool` - Fails deterministically based on attempt number
//! - `PanicTool` - Panics with the given message
//!
//! ## SimpleTools
//! - `GoodSimpleTool` - Echoes back input message
//! - `ErrorSimpleTool` - Always returns an error
//! - `SlowSimpleTool` - Sleeps for configurable duration

use std::collections::HashSet;

use durable_tools::ToolMetadata;

pub mod error;
pub mod tools;
mod visitor;

pub use error::{AutopilotToolError, AutopilotToolResult};

pub use visitor::ToolVisitor;

/// Collect all available tool names synchronously.
///
/// This is used by `AutopilotClient` to know which tools are available
/// for filtering unknown tool calls.
pub fn collect_tool_names() -> HashSet<String> {
    let mut names = HashSet::new();

    // Production tools
    names.insert(tools::InferenceTool::name().to_string());
    names.insert(tools::FeedbackTool::name().to_string());
    names.insert(tools::CreateDatapointsTool::name().to_string());
    names.insert(tools::CreateDatapointsFromInferencesTool::name().to_string());
    names.insert(tools::ListDatapointsTool::name().to_string());
    names.insert(tools::GetDatapointsTool::name().to_string());
    names.insert(tools::UpdateDatapointsTool::name().to_string());
    names.insert(tools::DeleteDatapointsTool::name().to_string());
    names.insert(tools::LaunchOptimizationWorkflowTool::name().to_string());
    names.insert(tools::GetLatestFeedbackByMetricTool::name().to_string());
    names.insert(tools::GetFeedbackByVariantTool::name().to_string());
    names.insert(tools::RunEvaluationTool::name().to_string());
    names.insert(tools::GetConfigTool::name().to_string());
    names.insert(tools::WriteConfigTool::name().to_string());
    names.insert(tools::ListInferencesTool::name().to_string());
    names.insert(tools::GetInferencesTool::name().to_string());
    names.insert(tools::AutoRejectToolCallTool::name().to_string());

    // Test tools (e2e_tests feature)
    #[cfg(feature = "e2e_tests")]
    {
        names.insert(tools::EchoTool::name().to_string());
        names.insert(tools::SlowTool::name().to_string());
        names.insert(tools::FailingTool::name().to_string());
        names.insert(tools::FlakyTool::name().to_string());
        names.insert(tools::PanicTool::name().to_string());
        names.insert(tools::GoodSimpleTool::name().to_string());
        names.insert(tools::ErrorSimpleTool::name().to_string());
        names.insert(tools::SlowSimpleTool::name().to_string());
    }

    names
}

/// Iterate over all tools with a visitor.
///
/// This is the single source of truth for all tools. Both local execution
/// (tensorzero repo) and remote execution (autopilot repo) use this function
/// with different visitor implementations.
///
/// When the `e2e_tests` feature is enabled, test tools are also included.
///
/// # Local Execution
///
/// Register tools with their `execute()` implementations:
///
/// ```ignore
/// struct LocalVisitor<'a>(&'a ToolExecutor);
///
/// #[async_trait]
/// impl ToolVisitor for LocalVisitor<'_> {
///     type Error = ToolError;
///
///     async fn visit_task_tool<T: TaskTool + Default>(&self) -> Result<(), ToolError> {
///         self.0.register_task_tool::<T>().await?;
///         Ok(())
///     }
///
///     async fn visit_simple_tool<T: SimpleTool + Default>(&self) -> Result<(), ToolError> {
///         self.0.register_simple_tool::<T>().await?;
///         Ok(())
///     }
/// }
///
/// for_each_tool(&LocalVisitor(&executor)).await?;
/// ```
///
/// # Remote Execution
///
/// Wrap tools for client-side execution:
///
/// ```ignore
/// struct RemoteVisitor<'a>(&'a ToolExecutor);
///
/// #[async_trait]
/// impl ToolVisitor for RemoteVisitor<'_> {
///     type Error = ToolError;
///
///     async fn visit_task_tool<T: TaskTool + Default>(&self) -> Result<(), ToolError> {
///         self.0.register_client_tool::<T>().await?;
///         Ok(())
///     }
///
///     async fn visit_simple_tool<T: SimpleTool + Default>(&self) -> Result<(), ToolError> {
///         self.0.register_client_tool::<T>().await?;
///         Ok(())
///     }
/// }
///
/// for_each_tool(&RemoteVisitor(&executor)).await?;
/// ```
///
/// # Errors
///
/// Returns an error if any tool visit fails.
pub async fn for_each_tool<V: ToolVisitor>(visitor: &V) -> Result<(), V::Error> {
    // Production tools
    // ----------------

    // Inference tool
    visitor.visit_simple_tool::<tools::InferenceTool>().await?;

    // Feedback tool
    visitor.visit_simple_tool::<tools::FeedbackTool>().await?;

    // Datapoint CRUD tools
    visitor
        .visit_simple_tool::<tools::CreateDatapointsTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::CreateDatapointsFromInferencesTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::ListDatapointsTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::GetDatapointsTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::UpdateDatapointsTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::DeleteDatapointsTool>()
        .await?;
    visitor
        .visit_task_tool::<tools::LaunchOptimizationWorkflowTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::GetLatestFeedbackByMetricTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::GetFeedbackByVariantTool>()
        .await?;

    // Evaluation tool
    visitor
        .visit_simple_tool::<tools::RunEvaluationTool>()
        .await?;

    // Config snapshot tools
    visitor.visit_simple_tool::<tools::GetConfigTool>().await?;
    visitor
        .visit_simple_tool::<tools::WriteConfigTool>()
        .await?;

    // Inference query tools
    visitor
        .visit_simple_tool::<tools::ListInferencesTool>()
        .await?;
    visitor
        .visit_simple_tool::<tools::GetInferencesTool>()
        .await?;

    // Internal tools
    visitor
        .visit_simple_tool::<tools::AutoRejectToolCallTool>()
        .await?;

    // Test tools (e2e_tests feature)
    // ------------------------------
    #[cfg(feature = "e2e_tests")]
    {
        // TaskTools
        visitor.visit_task_tool::<tools::EchoTool>().await?;
        visitor.visit_task_tool::<tools::SlowTool>().await?;
        visitor.visit_task_tool::<tools::FailingTool>().await?;
        visitor.visit_task_tool::<tools::FlakyTool>().await?;
        visitor.visit_task_tool::<tools::PanicTool>().await?;

        // SimpleTools
        visitor.visit_simple_tool::<tools::GoodSimpleTool>().await?;
        visitor
            .visit_simple_tool::<tools::ErrorSimpleTool>()
            .await?;
        visitor.visit_simple_tool::<tools::SlowSimpleTool>().await?;
    }

    Ok(())
}
