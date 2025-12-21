use async_trait::async_trait;
use durable::{Task, TaskContext, TaskResult};
use schemars::schema::RootSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::time::Duration;
use uuid::Uuid;

use crate::context::{ToolAppState, ToolContext};
use crate::error::ToolResult as ToolExecResult;

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

/// A durable tool that runs as a full durable Task.
///
/// `TaskTools` have access to the full `ToolContext` which provides:
/// - Checkpointing via `step()`
/// - Calling other tools via `call_tool()`
/// - Durable sleep, events, and random values
/// - Access to the database pool
///
/// Implement this trait for tools that need durable execution guarantees
/// or need to call other tools.
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
/// use durable_tools::{TaskTool, ToolContext, ToolResult, async_trait};
/// use schemars::{schema_for, schema::RootSchema, JsonSchema};
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
/// struct ResearchTool;
///
/// #[async_trait]
/// impl TaskTool for ResearchTool {
///     fn name() -> Cow<'static, str> {
///         Cow::Borrowed("research")
///     }
///
///     fn description() -> Cow<'static, str> {
///         Cow::Borrowed("Research a topic")
///     }
///
///     fn parameters_schema() -> RootSchema {
///         schema_for!(ResearchParams)
///     }
///
///     type LlmParams = ResearchParams;
///     type SideInfo = ();
///     type Output = ResearchResult;
///
///     async fn execute(
///         llm_params: Self::LlmParams,
///         _side_info: Self::SideInfo,
///         ctx: &mut ToolContext<'_>,
///     ) -> ToolResult<Self::Output> {
///         // Call other tools
///         let search = ctx.call_tool("search", serde_json::json!({"query": llm_params.topic})).await?;
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
/// use durable_tools::{TaskTool, ToolContext, ToolResult, SideInfo, async_trait};
/// use schemars::{schema_for, schema::RootSchema, JsonSchema};
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
/// struct GitHubSearchTool;
///
/// #[async_trait]
/// impl TaskTool for GitHubSearchTool {
///     fn name() -> Cow<'static, str> {
///         Cow::Borrowed("github_search")
///     }
///
///     fn description() -> Cow<'static, str> {
///         Cow::Borrowed("Search GitHub")
///     }
///
///     fn parameters_schema() -> RootSchema {
///         schema_for!(GitHubSearchParams)  // Only LlmParams in schema
///     }
///
///     type LlmParams = GitHubSearchParams;
///     type SideInfo = GitHubCredentials;
///     type Output = Vec<String>;
///
///     async fn execute(
///         llm_params: Self::LlmParams,
///         side_info: Self::SideInfo,
///         ctx: &mut ToolContext<'_>,
///     ) -> ToolResult<Self::Output> {
///         // Use llm_params.query (from LLM)
///         // Use side_info.api_token (hidden from LLM)
///         Ok(vec![])
///     }
/// }
/// ```
#[async_trait]
pub trait TaskTool: Send + Sync + 'static {
    /// Unique name for this tool.
    ///
    /// This is used for registration, invocation, and as the durable task name.
    fn name() -> Cow<'static, str>;

    /// Human-readable description of what this tool does.
    ///
    /// Used for generating LLM function definitions.
    fn description() -> Cow<'static, str>;

    /// JSON Schema for the tool's LLM-visible parameters.
    ///
    /// This should return the schema for `LlmParams` only.
    /// Side information is not included in the schema.
    fn parameters_schema() -> RootSchema;

    /// The LLM-visible parameter type (must be JSON-serializable).
    ///
    /// This is what the LLM sees and can fill in when calling the tool.
    type LlmParams: Serialize + DeserializeOwned + Send + 'static;

    /// Side information type provided at spawn time (hidden from LLM).
    ///
    /// Use `()` if no side information is needed.
    type SideInfo: SideInfo;

    /// The output type for this tool (must be JSON-serializable).
    type Output: Serialize + DeserializeOwned + Send + 'static;

    /// Execution timeout for this tool.
    ///
    /// Defaults to 120 seconds.
    fn timeout() -> Duration {
        Duration::from_secs(120)
    }

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
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecResult<Self::Output>;
}

/// Wrapper params that include `episode_id`, LLM params, and side info.
///
/// This is what gets serialized as the durable task params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskToolParams<L, S = ()> {
    /// The LLM-provided parameters.
    pub llm_params: L,
    /// Side information (hidden from LLM).
    pub side_info: S,
    /// The episode ID for this execution.
    pub episode_id: Uuid,
}

/// Adapter that implements `durable::Task` for any `TaskTool`.
///
/// This allows `TaskTools` to be registered with the durable worker.
pub struct TaskToolAdapter<T: TaskTool>(PhantomData<T>);

impl<T: TaskTool> TaskToolAdapter<T> {
    /// Create a new adapter instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: TaskTool> Default for TaskToolAdapter<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: TaskTool> Task<ToolAppState> for TaskToolAdapter<T> {
    fn name() -> Cow<'static, str> {
        T::name()
    }

    type Params = TaskToolParams<T::LlmParams, T::SideInfo>;
    type Output = T::Output;

    async fn run(
        wrapped: Self::Params,
        mut task_ctx: TaskContext<ToolAppState>,
        app_ctx: ToolAppState,
    ) -> TaskResult<Self::Output> {
        let mut tool_ctx = ToolContext::new(&mut task_ctx, &app_ctx, wrapped.episode_id);
        T::execute(wrapped.llm_params, wrapped.side_info, &mut tool_ctx)
            .await
            .map_err(Into::into)
    }
}
