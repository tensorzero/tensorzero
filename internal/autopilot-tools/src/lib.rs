//! Client-side tool definitions for TensorZero Autopilot.
//!
//! This crate re-exports tool traits from `durable-tools` for defining tools
//! that are executed client-side (outside of the autopilot server).
//!
//! # Overview
//!
//! Client tools differ from server-side tools in that:
//! - The tool metadata (name, description, schema) is defined here
//! - The actual execution happens on the client
//! - The server writes a `ToolCall` event and waits for a `ToolResult` event
//!
//! # Example
//!
//! ```ignore
//! use autopilot_tools::ToolMetadata;
//! use schemars::{Schema, schema_for};
//! use serde::{Deserialize, Serialize};
//! use std::borrow::Cow;
//!
//! #[derive(Serialize, Deserialize, schemars::JsonSchema)]
//! struct ReadFileParams {
//!     path: String,
//! }
//!
//! #[derive(Default)]
//! struct ReadFileTool;
//!
//! impl ToolMetadata for ReadFileTool {
//!     fn name() -> Cow<'static, str> {
//!         Cow::Borrowed("read_file")
//!     }
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Read the contents of a file at the given path")
//!     }
//!
//!     fn parameters_schema() -> ToolResult<Schema> {
//!         Ok(schema_for!(ReadFileParams))
//!     }
//!
//!     type LlmParams = ReadFileParams;
//! }
//!
//! // Use durable_tools::ToolRegistry for registration
//! ```
//! TODO: implement client-side tools in this crate, export as a registry.

// Re-export from durable-tools
pub use durable_tools::{ErasedTool, ToolMetadata, ToolRegistry};
