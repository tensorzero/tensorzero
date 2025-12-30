//! Visitor pattern for tool registration.
//!
//! This module provides a trait for iterating over tool definitions, allowing
//! different registration strategies while ensuring the same set of tools is
//! processed regardless of the visitor implementation.
//!
//! # Use Cases
//!
//! - **Local execution**: Register tools with their `execute()` implementations
//!   for execution on the worker.
//! - **Remote execution**: Wrap tools for client-side execution (e.g., via
//!   `ClientToolTaskAdapter`), using only the `ToolMetadata`.
//!
//! # Example
//!
//! ```ignore
//! use durable_tools::{ToolVisitor, TaskTool, SimpleTool, ToolError};
//!
//! // Define tools once
//! pub async fn for_each_tool<V: ToolVisitor>(visitor: &V) -> Result<(), V::Error> {
//!     visitor.visit_task_tool::<MyTaskTool>().await?;
//!     visitor.visit_simple_tool::<MySimpleTool>().await?;
//!     Ok(())
//! }
//!
//! // Local execution visitor
//! struct LocalVisitor<'a>(&'a ToolExecutor);
//!
//! #[async_trait]
//! impl ToolVisitor for LocalVisitor<'_> {
//!     type Error = ToolError;
//!
//!     async fn visit_task_tool<T: TaskTool + Default>(&self) -> Result<(), ToolError> {
//!         self.0.register_task_tool::<T>().await?;
//!         Ok(())
//!     }
//!
//!     async fn visit_simple_tool<T: SimpleTool + Default>(&self) -> Result<(), ToolError> {
//!         self.0.register_simple_tool::<T>().await?;
//!         Ok(())
//!     }
//! }
//!
//! // Remote execution visitor (wraps in adapter)
//! struct RemoteVisitor<'a>(&'a ToolExecutor);
//!
//! #[async_trait]
//! impl ToolVisitor for RemoteVisitor<'_> {
//!     type Error = ToolError;
//!
//!     async fn visit_task_tool<T: TaskTool + Default>(&self) -> Result<(), ToolError> {
//!         // Wrap in ClientToolTaskAdapter for remote execution
//!         self.0.register_client_tool::<T>().await?;
//!         Ok(())
//!     }
//!
//!     async fn visit_simple_tool<T: SimpleTool + Default>(&self) -> Result<(), ToolError> {
//!         self.0.register_client_tool::<T>().await?;
//!         Ok(())
//!     }
//! }
//! ```

use async_trait::async_trait;

use crate::simple_tool::SimpleTool;
use crate::task_tool::TaskTool;

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
/// The `Default + PartialEq` bounds on `SideInfo` are required for:
/// - Wrapper types that need to construct default side info
/// - Comparison operations during tool execution
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
    async fn visit_task_tool<T: TaskTool + Default>(&self) -> Result<(), Self::Error>
    where
        T::SideInfo: Default + PartialEq;

    /// Visit a `SimpleTool`.
    ///
    /// For local execution, this typically calls `register_simple_tool`.
    /// For remote execution, this wraps the tool in an adapter.
    async fn visit_simple_tool<T: SimpleTool + Default>(&self) -> Result<(), Self::Error>;
}
