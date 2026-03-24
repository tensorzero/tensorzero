//! Visitor pattern for tool registration.
//!
//! This module provides a trait for iterating over tool definitions, allowing
//! different registration strategies while ensuring the same set of tools is
//! processed regardless of the visitor implementation.

use std::collections::HashSet;

use async_trait::async_trait;
use durable_tools::{SimpleTool, TaskTool};

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
/// helpers on `ToolExecutorBuilder` instead.
///
/// The bounds on `SideInfo` are required for:
/// - `Serialize`: Serializing side info into tool call events
///
/// # Implementors
///
/// - **Local execution**: Push tools onto a `ToolExecutorBuilder`
/// - **Remote execution**: Wrap tools in an adapter (e.g., `ClientToolTaskAdapter`)
///   that delegates execution to a remote client
#[async_trait]
pub trait ToolVisitor {
    /// The error type returned by visitor methods.
    type Error;

    /// Visit a `TaskTool`.
    ///
    /// For local execution, this typically pushes the tool onto a builder.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_task_tool<T>(&mut self, tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool<SideInfo = AutopilotSideInfo, ExtraState = ()>;

    /// Visit a `SimpleTool`.
    ///
    /// For local execution, this typically pushes the tool onto a builder.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_simple_tool<T>(&mut self) -> Result<(), Self::Error>
    where
        T: SimpleTool<SideInfo = AutopilotSideInfo> + Default;

    /// Visit a standalone `TaskTool` (no `SideInfo`, no result publishing).
    ///
    /// Standalone task tools are registered directly without wrapping.
    /// They are not visible to the autopilot server.
    async fn visit_standalone_task_tool<T>(&mut self, tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool<SideInfo = (), ExtraState = ()>;
}

/// A visitor that collects tool names from `for_each_tool`.
///
/// This is used by `collect_tool_names` to derive the set of available
/// tool names from the single source of truth in `for_each_tool`.
pub struct ToolNameCollector {
    names: HashSet<String>,
}

impl ToolNameCollector {
    pub fn new() -> Self {
        Self {
            names: HashSet::new(),
        }
    }

    /// Consume the collector and return the collected tool names.
    pub fn into_names(self) -> HashSet<String> {
        self.names
    }
}

impl Default for ToolNameCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolVisitor for ToolNameCollector {
    type Error = String;

    async fn visit_task_tool<T>(&mut self, tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool<SideInfo = AutopilotSideInfo, ExtraState = ()>,
    {
        self.names.insert(tool.name().to_string());
        Ok(())
    }

    async fn visit_simple_tool<T>(&mut self) -> Result<(), Self::Error>
    where
        T: SimpleTool<SideInfo = AutopilotSideInfo> + Default,
    {
        self.names.insert(T::default().name().to_string());
        Ok(())
    }

    async fn visit_standalone_task_tool<T>(&mut self, _tool: T) -> Result<(), Self::Error>
    where
        T: TaskTool<SideInfo = (), ExtraState = ()>,
    {
        // Standalone tools are not visible to the autopilot server
        Ok(())
    }
}
