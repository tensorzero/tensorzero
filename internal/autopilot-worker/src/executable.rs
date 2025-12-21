//! Executable client tool trait definition.

use async_trait::async_trait;
use autopilot_tools::ClientTool;
use durable_tools::SideInfo;
use serde::{Serialize, de::DeserializeOwned};

use crate::context::AutopilotToolContext;
use crate::error::AutopilotToolResult;

/// A client tool that can be executed in the autopilot worker.
///
/// This trait extends [`ClientTool`] with execution logic. Tools implementing
/// this trait can be registered with the autopilot worker and executed durably.
///
/// # Side Information
///
/// Tools can receive "side information" - parameters provided at spawn time
/// but hidden from the LLM. Set `type SideInfo = ()` for tools that don't need it.
///
/// # Example
///
/// ```ignore
/// use autopilot_worker::{ExecutableClientTool, AutopilotToolContext, AutopilotToolResult};
/// use autopilot_tools::ClientTool;
/// use async_trait::async_trait;
/// use schemars::{Schema, schema_for};
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
///
/// #[derive(Serialize, Deserialize, schemars::JsonSchema)]
/// struct ReadFileParams {
///     path: String,
/// }
///
/// #[derive(Serialize, Deserialize)]
/// struct ReadFileOutput {
///     content: String,
/// }
///
/// #[derive(Default)]
/// struct ReadFileTool;
///
/// impl ClientTool for ReadFileTool {
///     fn name() -> Cow<'static, str> {
///         Cow::Borrowed("read_file")
///     }
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
///
/// #[async_trait]
/// impl ExecutableClientTool for ReadFileTool {
///     type Output = ReadFileOutput;
///     type SideInfo = ();
///
///     async fn execute(
///         llm_params: Self::LlmParams,
///         _side_info: Self::SideInfo,
///         ctx: &mut AutopilotToolContext<'_, '_>,
///     ) -> AutopilotToolResult<Self::Output> {
///         // Read file contents...
///         Ok(ReadFileOutput {
///             content: format!("Contents of {}", llm_params.path),
///         })
///     }
/// }
/// ```
#[async_trait]
pub trait ExecutableClientTool: ClientTool + Default + Send + Sync + 'static {
    /// The output type for this tool (must be JSON-serializable).
    type Output: Serialize + DeserializeOwned + Send + 'static;

    /// Side information type provided at spawn time (hidden from LLM).
    ///
    /// Use `()` if no side information is needed.
    type SideInfo: SideInfo;

    /// Execute the tool logic.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - Parameters provided by the LLM
    /// * `side_info` - Side information provided at spawn time (hidden from LLM)
    /// * `ctx` - The autopilot tool context (provides access to TensorZero functionality)
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut AutopilotToolContext<'_, '_>,
    ) -> AutopilotToolResult<Self::Output>;
}
