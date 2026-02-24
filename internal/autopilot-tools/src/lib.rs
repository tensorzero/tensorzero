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
//! - `ListDatasetsTool` - Lists available datasets with metadata
//! - `ListDatapointsTool` - Lists datapoints with filtering and pagination
//! - `GetDatapointsTool` - Gets specific datapoints by ID
//! - `UpdateDatapointsTool` - Updates existing datapoints
//! - `DeleteDatapointsTool` - Deletes datapoints by ID
//! - `LaunchOptimizationWorkflowTool` - Launches an optimization workflow (e.g., fine-tuning)
//! - `GetLatestFeedbackByMetricTool` - Gets the latest feedback ID for each metric for a target
//! - `GetFeedbackByVariantTool` - Gets feedback statistics (mean, variance, count) by variant for a function and metric
//! - `RunEvaluationTool` - Runs an evaluation on a dataset and returns statistics
//! - `ListInferencesTool` - Lists inferences with filtering and pagination
//! - `UploadDatasetTool` - Uploads a dataset to S3 as JSONL using multipart upload
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

pub mod error;
pub mod fix_strict_tool_schema;
pub mod tools;
mod visitor;

pub use error::{AutopilotToolError, AutopilotToolResult};

pub use visitor::{ToolNameCollector, ToolVisitor};

/// Collect all available tool names.
///
/// This uses the visitor pattern with `for_each_tool` to derive the set of
/// tool names from the single source of truth.
///
/// This is used by `AutopilotClient` to know which tools are available
/// for filtering unknown tool calls.
///
/// # Errors
///
/// Returns an error if visiting any tool fails (e.g., lock poisoning).
pub async fn collect_tool_names() -> Result<HashSet<String>, String> {
    let collector = ToolNameCollector::new();
    for_each_tool(&collector).await?;
    Ok(collector.into_names())
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
///     async fn visit_task_tool<T: TaskTool>(&self, tool: T) -> Result<(), ToolError> {
///         self.0.register_task_tool_instance(tool).await?;
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
///     async fn visit_task_tool<T: TaskTool>(&self, tool: T) -> Result<(), ToolError> {
///         self.0.register_client_tool_instance(tool).await?;
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
        .visit_simple_tool::<tools::ListDatasetsTool>()
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
        .visit_task_tool(tools::LaunchOptimizationWorkflowTool)
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

    // Dataset upload tool
    visitor.visit_task_tool(tools::UploadDatasetTool).await?;

    // Episode query tools
    visitor
        .visit_simple_tool::<tools::ListEpisodesTool>()
        .await?;

    // Test tools (e2e_tests feature)
    // ------------------------------
    #[cfg(feature = "e2e_tests")]
    {
        // TaskTools
        visitor.visit_task_tool(tools::EchoTool).await?;
        visitor.visit_task_tool(tools::SlowTool).await?;
        visitor.visit_task_tool(tools::FailingTool).await?;
        visitor.visit_task_tool(tools::FlakyTool).await?;
        visitor.visit_task_tool(tools::PanicTool).await?;

        // SimpleTools
        visitor.visit_simple_tool::<tools::GoodSimpleTool>().await?;
        visitor
            .visit_simple_tool::<tools::ErrorSimpleTool>()
            .await?;
        visitor.visit_simple_tool::<tools::SlowSimpleTool>().await?;
    }

    Ok(())
}
