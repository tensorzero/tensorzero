use schemars::{JsonSchema, Schema};
use serde::{Serialize, de::DeserializeOwned};
use std::borrow::Cow;
use std::time::Duration;

use crate::error::ToolResult;

/// Marker trait for side information types.
///
/// Types implementing this can be used as side information for tools.
/// Side information is provided at spawn time and is hidden from the LLM
/// (not included in the tool's JSON schema).
///
/// The unit type `()` implements this trait for tools that don't need side info.
pub trait SideInfo: Serialize + DeserializeOwned + Send + 'static {}

/// Unit type implements `SideInfo` for tools without side information.
impl SideInfo for () {}

/// Common metadata trait for all tools (both `TaskTool` and `SimpleTool`).
///
/// This trait defines the metadata required to expose a tool to an LLM,
/// including its name, description, parameter schema, and timeout.
///
/// Both `TaskTool` and `SimpleTool` extend this trait via supertrait bounds,
/// allowing for unified handling of tool metadata across the system.
///
/// # Example
///
/// ```ignore
/// use durable_tools::ToolMetadata;
/// use schemars::{schema_for, JsonSchema, Schema};
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
/// use std::time::Duration;
///
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct MyToolParams {
///     query: String,
/// }
///
/// struct MyTool;
///
/// impl ToolMetadata for MyTool {
///     fn name() -> Cow<'static, str> {
///         Cow::Borrowed("my_tool")
///     }
///
///     fn description() -> Cow<'static, str> {
///         Cow::Borrowed("A tool that does something")
///     }
///
///     fn parameters_schema() -> ToolResult<Schema> {
///         Ok(schema_for!(MyToolParams))
///     }
///
///     type LlmParams = MyToolParams;
/// }
/// ```
pub trait ToolMetadata: Send + Sync + 'static {
    /// Side information type provided at spawn time (hidden from LLM).
    ///
    /// Use `()` if no side information is needed.
    type SideInfo: SideInfo;

    /// The output type for this tool (must be JSON-serializable).
    type Output: Serialize + DeserializeOwned + Send + Sync + 'static;
    /// Unique name for this tool.
    ///
    /// This is used for registration, invocation, and as an identifier in the LLM.
    fn name() -> Cow<'static, str>;

    /// Human-readable description of what this tool does.
    ///
    /// Used for generating LLM function definitions.
    fn description() -> Cow<'static, str>;

    /// JSON Schema for the tool's LLM-visible parameters.
    fn parameters_schema() -> ToolResult<Schema>;

    /// The LLM-visible parameter type.
    ///
    /// This is what the LLM sees and can fill in when calling the tool.
    ///
    /// Must implement:
    /// - `Serialize` and `DeserializeOwned` for JSON serialization
    /// - `JsonSchema` for schema generation
    /// - `Send + Sync + 'static` for thread-safety
    type LlmParams: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static;

    /// Execution timeout for this tool.
    ///
    /// Defaults to 60 seconds. Override this for tools with different requirements.
    fn timeout() -> Duration {
        Duration::from_secs(60)
    }
}
