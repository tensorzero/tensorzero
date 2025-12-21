//! Tool registry functions for autopilot worker.
//!
//! Tools are registered using the [`define_client_tool_wrapper`] macro and then
//! registered with the executor.
//!
//! # Example
//!
//! ```ignore
//! use autopilot_worker::{define_client_tool_wrapper, ExecutableClientTool};
//!
//! // Define a tool
//! struct MyTool;
//! // ... implement ClientTool and ExecutableClientTool for MyTool ...
//!
//! // Create a wrapper
//! define_client_tool_wrapper!(MyToolWrapper, MyTool, "my_tool");
//!
//! // Register the wrapper with the executor
//! executor.register_task_tool::<MyToolWrapper>().await;
//! ```

use durable_tools::ToolExecutor;

/// Build the default registry with all autopilot tools.
///
/// This function registers all available autopilot client tools with the executor.
/// Currently this is a placeholder - concrete tools and their wrappers will be
/// added here as they are implemented.
///
/// Tools should be defined using the [`define_client_tool_wrapper`] macro and
/// then registered with [`ToolExecutor::register_task_tool`].
pub async fn build_default_registry(_executor: &ToolExecutor) {
    // Future: register concrete tools here using their wrapper types
    // Example:
    //
    // // First, define tools in a tools module:
    // // define_client_tool_wrapper!(ReadFileToolWrapper, ReadFileTool, "read_file");
    // // define_client_tool_wrapper!(WriteFileToolWrapper, WriteFileTool, "write_file");
    //
    // // Then register them:
    // executor.register_task_tool::<ReadFileToolWrapper>().await;
    // executor.register_task_tool::<WriteFileToolWrapper>().await;
}
