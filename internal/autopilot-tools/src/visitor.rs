//! Visitor pattern for tool registration.
//!
//! This module provides a trait for iterating over tool definitions, allowing
//! different registration strategies while ensuring the same set of tools is
//! processed regardless of the visitor implementation.

use std::collections::HashSet;
use std::sync::Mutex;

use async_trait::async_trait;
use durable_tools::{SimpleTool, TaskTool};
use serde::Serialize;

use autopilot_client::AutopilotSideInfo;

/// Visitor trait for iterating over tool definitions.
///
/// This allows different registration strategies while ensuring the same set
/// of tools is processed regardless of the visitor implementation.
///
/// # Type Parameters
///
/// The `Default` bound on tool types is required for the type-based registration
/// helpers. If you need runtime-configured tools, use the instance registration
/// helpers on `ToolExecutor` instead.
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
    async fn visit_task_tool<T>(&self, tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display;

    /// Visit a `SimpleTool`.
    ///
    /// For local execution, this typically calls `register_simple_tool`.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_simple_tool<T>(&self) -> Result<(), Self::Error>
    where
        T: SimpleTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display;
}

/// A visitor that collects tool names from `for_each_tool`.
///
/// This is used by `collect_tool_names` to derive the set of available
/// tool names from the single source of truth in `for_each_tool`.
pub struct ToolNameCollector {
    names: Mutex<HashSet<String>>,
}

impl ToolNameCollector {
    pub fn new() -> Self {
        Self {
            names: Mutex::new(HashSet::new()),
        }
    }

    /// Consume the collector and return the collected tool names.
    ///
    /// If the mutex was poisoned (which would only happen if a panic occurred
    /// while holding the lock), this returns the data anyway since we still
    /// want to use it.
    pub fn into_names(self) -> HashSet<String> {
        self.names.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for ToolNameCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolVisitor for ToolNameCollector {
    type Error = std::convert::Infallible;

    async fn visit_task_tool<T>(&self, tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
    {
        // unwrap_or_else handles the (practically impossible) poisoned case
        if let Ok(mut names) = self.names.lock() {
            names.insert(tool.name().to_string());
        }
        Ok(())
    }

    async fn visit_simple_tool<T>(&self) -> Result<(), Self::Error>
    where
        T: SimpleTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
    {
        if let Ok(mut names) = self.names.lock() {
            names.insert(T::default().name().to_string());
        }
        Ok(())
    }
}
