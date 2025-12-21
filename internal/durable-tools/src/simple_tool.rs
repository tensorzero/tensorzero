use async_trait::async_trait;
use schemars::schema::RootSchema;
use serde::{Serialize, de::DeserializeOwned};
use std::borrow::Cow;
use std::time::Duration;

use crate::context::SimpleToolContext;
use crate::error::ToolResult;
use crate::task_tool::SideInfo;

/// A lightweight tool that runs inside a `TaskTool`'s `step()` checkpoint.
///
/// `SimpleTools` are simpler than `TaskTools` - they execute within the
/// checkpoint of a parent `TaskTool` and don't have access to checkpointing
/// operations themselves. Use `SimpleTools` for:
///
/// - Simple, stateless operations
/// - External API calls (use the `idempotency_key` for deduplication)
/// - Database queries
///
/// `SimpleTools` receive a `SimpleToolContext` which provides access to
/// the database pool. The `idempotency_key` parameter can be used to make
/// external API calls idempotent.
///
/// # Side Information
///
/// Like `TaskTools`, `SimpleTools` can receive "side information" - parameters
/// provided at spawn time but hidden from the LLM. Set `type SideInfo = ()` for
/// tools that don't need side info.
///
/// # Example (without side info)
///
/// ```ignore
/// use durable_tools::{SimpleTool, SimpleToolContext, ToolResult, async_trait};
/// use schemars::{schema_for, schema::RootSchema, JsonSchema};
/// use serde::{Deserialize, Serialize};
/// use std::borrow::Cow;
///
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct SearchParams {
///     query: String,
/// }
///
/// #[derive(Serialize, Deserialize)]
/// struct SearchResult {
///     results: Vec<String>,
/// }
///
/// struct SearchTool;
///
/// #[async_trait]
/// impl SimpleTool for SearchTool {
///     fn name() -> Cow<'static, str> {
///         Cow::Borrowed("search")
///     }
///
///     fn description() -> Cow<'static, str> {
///         Cow::Borrowed("Search the web")
///     }
///
///     fn parameters_schema() -> RootSchema {
///         schema_for!(SearchParams)
///     }
///
///     type LlmParams = SearchParams;
///     type SideInfo = ();
///     type Output = SearchResult;
///
///     async fn execute(
///         llm_params: Self::LlmParams,
///         _side_info: Self::SideInfo,
///         ctx: SimpleToolContext<'_>,
///         idempotency_key: &str,
///     ) -> ToolResult<Self::Output> {
///         // Use idempotency_key for external API calls
///         let results = external_search_api(&llm_params.query, idempotency_key).await?;
///         Ok(SearchResult { results })
///     }
/// }
/// ```
#[async_trait]
pub trait SimpleTool: Send + Sync + 'static {
    /// Unique name for this tool.
    ///
    /// This is used for registration and invocation.
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

    /// Side information type provided at call time (hidden from LLM).
    ///
    /// Use `()` if no side information is needed.
    type SideInfo: SideInfo;

    /// The output type for this tool (must be JSON-serializable).
    type Output: Serialize + DeserializeOwned + Send + 'static;

    /// Execution timeout for this tool.
    ///
    /// Defaults to 30 seconds (shorter than `TaskTool` since these are lightweight).
    fn timeout() -> Duration {
        Duration::from_secs(30)
    }

    /// Execute the tool logic.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - Parameters provided by the LLM
    /// * `side_info` - Side information provided at call time (hidden from LLM)
    /// * `ctx` - The simple tool context (provides database access)
    /// * `idempotency_key` - A unique key for this execution (use for external API calls)
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> ToolResult<Self::Output>;
}
