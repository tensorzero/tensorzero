use async_trait::async_trait;
use durable::{Task, TaskContext, TaskResult};
use std::borrow::Cow;
use std::sync::Arc;

use crate::context::{ToolAppState, ToolContext};
use crate::error::ToolResult as ToolExecResult;
use crate::tool_metadata::ToolMetadata;

/// A durable tool that runs as a full durable Task.
///
/// `TaskTools` have access to the full `ToolContext` which provides:
/// - Checkpointing via `step()`
/// - Calling other tools via `call_tool()`
/// - Durable sleep, events, and random values
/// - Access to the database pool
///
/// Implement this trait for tools that need durable execution guarantees
/// or need to call other tools. Note that `TaskTool` extends [`ToolMetadata`],
/// so you must implement both traits.
///
/// # Side Information
///
/// Tools can receive "side information" - parameters provided at spawn time
/// but hidden from the LLM. Set `type SideInfo = ()` for tools that don't
/// need side info, or define a custom type for tools that do.
///
/// # Example (without side info)
///
/// ```ignore
/// use durable_tools::{TaskTool, ToolContext, ToolResult, ToolMetadata, async_trait};
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
///
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct ResearchParams {
///     topic: String,
/// }
///
/// #[derive(Serialize, Deserialize)]
/// struct ResearchResult {
///     summary: String,
/// }
///
/// #[derive(Default)]
/// struct ResearchTool;
///
/// impl ToolMetadata for ResearchTool {
///     type SideInfo = ();
///     type Output = ResearchResult;
///     type LlmParams = ResearchParams;
///
///     fn name(&self) -> Cow<'static, str> {
///         Cow::Borrowed("research")
///     }
///
///     fn description(&self) -> Cow<'static, str> {
///         Cow::Borrowed("Research a topic")
///     }
///     // parameters_schema() is automatically derived from LlmParams
/// }
///
/// #[async_trait]
/// impl TaskTool for ResearchTool {
///     async fn execute(
///         &self,
///         llm_params: <Self as ToolMetadata>::LlmParams,
///         _side_info: <Self as ToolMetadata>::SideInfo,
///         ctx: &mut ToolContext<'_>,
///     ) -> ToolResult<<Self as ToolMetadata>::Output> {
///         // Call other tools
///         let search = ctx.call_tool("search", serde_json::json!({"query": llm_params.topic}), serde_json::json!(null)).await?;
///
///         // Use checkpointed steps
///         let analysis = ctx
///             .step("analyze", (), |(), _state| async {
///                 Ok("Analysis of results".to_string())
///             })
///             .await?;
///
///         Ok(ResearchResult { summary: analysis })
///     }
/// }
/// ```
///
/// # Example (with side info)
///
/// ```ignore
/// use durable_tools::{TaskTool, ToolContext, ToolResult, ToolMetadata, SideInfo, async_trait};
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
///
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct GitHubSearchParams {
///     query: String,
/// }
///
/// // Side info is NOT visible to LLM
/// #[derive(Serialize, Deserialize)]
/// struct GitHubCredentials {
///     api_token: String,
/// }
///
/// impl SideInfo for GitHubCredentials {}
///
/// #[derive(Default)]
/// struct GitHubSearchTool;
///
/// impl ToolMetadata for GitHubSearchTool {
///     type LlmParams = GitHubSearchParams;
///     type SideInfo = GitHubCredentials;
///     type Output = Vec<String>;
///
///     fn name(&self) -> Cow<'static, str> {
///         Cow::Borrowed("github_search")
///     }
///
///     fn description(&self) -> Cow<'static, str> {
///         Cow::Borrowed("Search GitHub")
///     }
///     // parameters_schema() is automatically derived from LlmParams
/// }
///
/// #[async_trait]
/// impl TaskTool for GitHubSearchTool {
///     async fn execute(
///         &self,
///         llm_params: <Self as ToolMetadata>::LlmParams,
///         side_info: <Self as ToolMetadata>::SideInfo,
///         ctx: &mut ToolContext<'_>,
///     ) -> ToolResult<<Self as ToolMetadata>::Output> {
///         // Use llm_params.query (from LLM)
///         // Use side_info.api_token (hidden from LLM)
///         Ok(vec![])
///     }
/// }
/// ```
#[async_trait]
pub trait TaskTool: ToolMetadata {
    /// Execute the tool logic.
    ///
    /// This is called by the durable worker when the tool is invoked.
    /// The context is passed by mutable reference to allow wrapper types
    /// to perform additional checkpointed operations after execution.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - Parameters provided by the LLM
    /// * `side_info` - Side information provided at spawn time (hidden from LLM)
    /// * `ctx` - The tool execution context
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecResult<<Self as ToolMetadata>::Output>;
}

// Re-export TaskToolParams from spawn crate
pub use durable_tools_spawn::TaskToolParams;

/// Adapter that implements `durable::Task` for any `TaskTool`.
///
/// This allows `TaskTools` to be registered with the durable worker.
pub struct TaskToolAdapter<T: TaskTool>(Arc<T>);

impl<T: TaskTool> TaskToolAdapter<T> {
    /// Create a new adapter instance wrapping the given tool.
    pub fn new(tool: Arc<T>) -> Self {
        Self(tool)
    }
}

#[async_trait]
impl<T: TaskTool> Task<ToolAppState> for TaskToolAdapter<T> {
    fn name(&self) -> Cow<'static, str> {
        self.0.name()
    }

    type Params = TaskToolParams<<T as ToolMetadata>::LlmParams, T::SideInfo>;
    type Output = T::Output;

    async fn run(
        &self,
        wrapped: Self::Params,
        mut task_ctx: TaskContext<ToolAppState>,
        app_ctx: ToolAppState,
    ) -> TaskResult<Self::Output> {
        let mut tool_ctx = ToolContext::new(&mut task_ctx, &app_ctx, wrapped.episode_id);
        self.0
            .execute(wrapped.llm_params, wrapped.side_info, &mut tool_ctx)
            .await
            .map_err(Into::into)
    }
}
