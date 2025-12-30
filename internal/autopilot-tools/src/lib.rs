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
//! - `GetLatestFeedbackByMetricTool` - Gets the latest feedback ID for each metric for a target
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

pub mod tools;
pub mod types;

pub use types::AutopilotToolSideInfo;

// Re-export ToolVisitor for use with for_each_tool
pub use durable_tools::ToolVisitor;

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
