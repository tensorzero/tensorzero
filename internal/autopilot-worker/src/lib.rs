//! Autopilot worker for executing client tools.
//!
//! This crate provides infrastructure for running TensorZero Autopilot tools
//! in a durable execution environment alongside the gateway.

mod side_info;
mod tools;
mod worker;
mod wrapper;

pub use side_info::AutopilotSideInfo;
#[cfg(feature = "e2e_tests")]
pub use tools::EchoTool;
pub use worker::{
    AutopilotWorker, AutopilotWorkerConfig, AutopilotWorkerHandle, spawn_autopilot_worker,
};
pub use wrapper::ClientToolWrapper;

// Re-export useful types from durable-tools
pub use durable_tools::{
    SideInfo, SimpleTool, SimpleToolContext, TaskTool, ToolContext, ToolError, ToolMetadata,
    ToolResult,
};
