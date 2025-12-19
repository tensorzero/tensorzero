use schemars::Schema;
use serde::{Serialize, de::DeserializeOwned};
use std::borrow::Cow;
use std::time::Duration;

/// A trait for defining client-side tools that can be executed by the autopilot client.
///
/// Client tools are tools where the execution happens on the TensorZero OSS deployment.
/// The server will create a ToolCall event, wait for the client to
/// execute the tool, and receive a ToolResult event back.
///
/// This trait is metadata-only for now - it defines the tool's name, description,
/// and parameters schema.
///
/// # Example
///
/// ```ignore
/// use autopilot_tools::ClientTool;
/// use schemars::{Schema, schema_for};
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
///
/// #[derive(Serialize, Deserialize, schemars::JsonSchema)]
/// struct ReadFileParams {
///     path: String,
/// }
///
/// struct ReadFileTool;
///
/// impl ClientTool for ReadFileTool {
///     const NAME: &'static str = "read_file";
///
///     type LlmParams = ReadFileParams;
///
///     fn description() -> Cow<'static, str> {
///         Cow::Borrowed("Read the contents of a file at the given path")
///     }
///
///     fn parameters_schema() -> Schema {
///         schema_for!(ReadFileParams).into()
///     }
/// }
/// ```
pub trait ClientTool {
    /// The unique name of this tool. Used as the tool name in ToolCall events.
    fn name() -> Cow<'static, str>;

    /// The parameters type that the LLM will provide when calling this tool.
    /// Must be JSON-serializable and have a JSON Schema.
    type LlmParams: Serialize + DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static;

    /// A human-readable description of what this tool does.
    /// This is shown to the LLM when presenting available tools.
    fn description() -> Cow<'static, str>;

    /// The JSON Schema for the tool's parameters.
    /// This defines what arguments the LLM can provide when calling the tool.
    fn parameters_schema() -> Schema;

    /// The timeout for this tool's execution.
    /// Defaults to 5 minutes (300 seconds).
    /// This should NOT account for approval.
    fn timeout() -> Duration {
        Duration::from_secs(300)
    }
}

/// Type-erased version of [`ClientTool`] for use in the registry.
///
/// This trait allows storing different ClientTool implementations in a heterogeneous
/// collection without knowing the concrete types at compile time.
pub trait ErasedClientTool: Send + Sync {
    /// Returns the tool's unique name.
    fn name(&self) -> Cow<'static, str>;

    /// Returns the tool's description.
    fn description(&self) -> Cow<'static, str>;

    /// Returns the JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> Schema;

    /// Returns the timeout for this tool's execution.
    fn timeout(&self) -> Duration;
}

/// Blanket implementation of ErasedClientTool for any ClientTool.
impl<T: ClientTool + Default + Send + Sync + 'static> ErasedClientTool for T {
    fn name(&self) -> Cow<'static, str> {
        T::name()
    }

    fn description(&self) -> Cow<'static, str> {
        T::description()
    }

    fn parameters_schema(&self) -> Schema {
        T::parameters_schema()
    }

    fn timeout(&self) -> Duration {
        T::timeout()
    }
}
