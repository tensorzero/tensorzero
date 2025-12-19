//! Client-side tool definitions for TensorZero Autopilot.
//!
//! This crate provides the [`ClientTool`] trait and [`ClientToolRegistry`] for defining
//! tools that are executed client-side (outside of the autopilot server).
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
//! use autopilot_tools::{ClientTool, ClientToolRegistry};
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
//! impl ClientTool for ReadFileTool {
//!     const NAME: &'static str = "read_file";
//!     type LlmParams = ReadFileParams;
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Read the contents of a file at the given path")
//!     }
//!
//!     fn parameters_schema() -> Schema {
//!         schema_for!(ReadFileParams).into()
//!     }
//! }
//!
//! // Register tools in a registry
//! let mut registry = ClientToolRegistry::new();
//! registry.register::<ReadFileTool>();
//! ```

mod client_tool;
mod registry;

pub use client_tool::{ClientTool, ErasedClientTool};
pub use registry::ClientToolRegistry;
