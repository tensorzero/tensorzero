//! Autopilot worker for executing client tools.
//!
//! This crate provides the infrastructure for running TensorZero Autopilot client tools
//! in a durable execution environment. It runs inside the gateway process alongside
//! the Axum server.
//!
//! # Overview
//!
//! The autopilot worker:
//! - Executes tools defined with [`ExecutableClientTool`]
//! - Wraps tools with [`ClientToolWrapper`] to automatically send results back to the autopilot API
//! - Runs durably using the `durable-tools` infrastructure
//!
//! # Example
//!
//! ```ignore
//! use autopilot_worker::{
//!     ExecutableClientTool, AutopilotToolContext, AutopilotToolResult,
//!     AutopilotWorker, AutopilotWorkerConfig, spawn_autopilot_worker,
//! };
//! use autopilot_tools::ClientTool;
//! use async_trait::async_trait;
//! use schemars::{Schema, schema_for};
//! use serde::{Deserialize, Serialize};
//! use std::borrow::Cow;
//!
//! // Define a tool
//! #[derive(Serialize, Deserialize, schemars::JsonSchema)]
//! struct EchoParams {
//!     message: String,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct EchoOutput {
//!     echoed: String,
//! }
//!
//! #[derive(Default)]
//! struct EchoTool;
//!
//! impl ClientTool for EchoTool {
//!     fn name() -> Cow<'static, str> {
//!         Cow::Borrowed("echo")
//!     }
//!
//!     type LlmParams = EchoParams;
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Echoes the input message")
//!     }
//!
//!     fn parameters_schema() -> Schema {
//!         schema_for!(EchoParams).into()
//!     }
//! }
//!
//! #[async_trait]
//! impl ExecutableClientTool for EchoTool {
//!     type Output = EchoOutput;
//!     type SideInfo = ();
//!
//!     async fn execute(
//!         llm_params: Self::LlmParams,
//!         _side_info: Self::SideInfo,
//!         _ctx: &mut AutopilotToolContext<'_, '_>,
//!     ) -> AutopilotToolResult<Self::Output> {
//!         Ok(EchoOutput {
//!             echoed: llm_params.message,
//!         })
//!     }
//! }
//!
//! // Spawn the worker alongside the gateway
//! spawn_autopilot_worker(&gateway_handle, config);
//! ```

mod context;
mod error;
mod executable;
mod registry;
pub mod state;
mod worker;
mod wrapper;

// Re-export main types
pub use context::AutopilotToolContext;
pub use error::{AutopilotToolError, AutopilotToolResult};
pub use executable::ExecutableClientTool;
pub use registry::build_default_registry;
pub use state::AutopilotExtension;
pub use worker::{AutopilotWorker, AutopilotWorkerConfig, spawn_autopilot_worker};
pub use wrapper::{execute_client_tool_impl, AutopilotSideInfo, ClientToolWrapper};

// Re-export commonly used types from dependencies
pub use async_trait::async_trait;
pub use durable_tools::SideInfo;
