//! Visitor pattern for tool registration.
//!
//! This module provides a trait for iterating over tool definitions, allowing
//! different registration strategies while ensuring the same set of tools is
//! processed regardless of the visitor implementation.

use std::fmt::Display;

use async_trait::async_trait;
use durable_tools::{SimpleTool, TaskTool};
use serde::Serialize;

use crate::types::AutopilotSideInfo;

/// Visitor trait for iterating over tool definitions.
///
/// This allows different registration strategies while ensuring the same set
/// of tools is processed regardless of the visitor implementation.
///
/// # Type Parameters
///
/// The `Default` bound on tool types is required because:
/// - `SimpleTool` registration requires `Default` for instantiation
/// - Remote execution adapters (like `ClientToolTaskAdapter`) require `Default`
///
/// The bounds on `SideInfo` are required for:
/// - `TryFrom<AutopilotSideInfo>`: Converting caller params to tool-specific side info
/// - `Serialize`: Serializing side info into tool call events
///
/// # Implementors
///
/// - **Local execution**: Call `register_task_tool`/`register_simple_tool` directly
/// - **Remote execution**: Wrap tools in an adapter (e.g., `ClientToolTaskAdapter`)
///   that delegates execution to a remote client
#[async_trait]
pub trait ToolVisitor {
    /// The error type returned by visitor methods.
    type Error;

    /// Visit a `TaskTool`.
    ///
    /// For local execution, this typically calls `register_task_tool`.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_task_tool<T>(&self) -> Result<(), Self::Error>
    where
        T: TaskTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: Into<anyhow::Error> + Display;

    /// Visit a `SimpleTool`.
    ///
    /// For local execution, this typically calls `register_simple_tool`.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_simple_tool<T>(&self) -> Result<(), Self::Error>
    where
        T: SimpleTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: Into<anyhow::Error> + Display;
}
