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
//! - `CreateDatapointsTool` - Creates datapoints in a dataset
//! - `CreateDatapointsFromInferencesTool` - Creates datapoints from existing inferences
//! - `ListDatapointsTool` - Lists datapoints with filtering and pagination
//! - `GetDatapointsTool` - Gets specific datapoints by ID
//! - `UpdateDatapointsTool` - Updates existing datapoints
//! - `DeleteDatapointsTool` - Deletes datapoints by ID
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

// Re-export from durable-tools
pub use durable_tools::{
    ErasedTool, SimpleTool, SimpleToolContext, TaskTool, ToolContext, ToolError, ToolMetadata,
    ToolRegistry, ToolResult,
};

/// Register production tools with the given registry.
///
/// This registers all production tools for autopilot operations.
///
/// # Errors
///
/// Returns an error if any tool registration fails.
pub fn register_production_tools(registry: &mut ToolRegistry) -> ToolResult<()> {
    // Inference tool
    registry.register_simple_tool::<tools::InferenceTool>()?;

    // Datapoint CRUD tools
    registry.register_simple_tool::<tools::CreateDatapointsTool>()?;
    registry.register_simple_tool::<tools::CreateDatapointsFromInferencesTool>()?;
    registry.register_simple_tool::<tools::ListDatapointsTool>()?;
    registry.register_simple_tool::<tools::GetDatapointsTool>()?;
    registry.register_simple_tool::<tools::UpdateDatapointsTool>()?;
    registry.register_simple_tool::<tools::DeleteDatapointsTool>()?;

    // Optimization tools
    registry.register_simple_tool::<tools::LaunchOptimizationWorkflowTool>()?;

    Ok(())
}

/// Register all test tools with the given registry.
///
/// This registers both TaskTools and SimpleTools used for e2e testing.
///
/// # Errors
///
/// Returns an error if any tool registration fails.
#[cfg(feature = "e2e_tests")]
pub fn register_test_tools(registry: &mut ToolRegistry) -> ToolResult<()> {
    // TaskTools
    registry.register_task_tool::<tools::EchoTool>()?;
    registry.register_task_tool::<tools::SlowTool>()?;
    registry.register_task_tool::<tools::FailingTool>()?;
    registry.register_task_tool::<tools::FlakyTool>()?;
    registry.register_task_tool::<tools::PanicTool>()?;

    // SimpleTools
    registry.register_simple_tool::<tools::GoodSimpleTool>()?;
    registry.register_simple_tool::<tools::ErrorSimpleTool>()?;
    registry.register_simple_tool::<tools::SlowSimpleTool>()?;

    Ok(())
}
