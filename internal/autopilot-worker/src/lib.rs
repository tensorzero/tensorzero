//! Autopilot worker for executing client tools.
//!
//! This crate provides infrastructure for running TensorZero Autopilot tools
//! in a durable execution environment alongside the gateway.

mod worker;
mod wrapper;

pub use worker::{
    AutopilotWorker, AutopilotWorkerConfig, AutopilotWorkerHandle, spawn_autopilot_worker,
};
pub use wrapper::ClientTaskToolWrapper;

// Re-export useful types from durable-tools and autopilot-tools
pub use durable_tools::{
    SimpleTool, SimpleToolContext, TaskTool, ToolContext, ToolError, ToolMetadata, ToolResult,
};

pub use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
