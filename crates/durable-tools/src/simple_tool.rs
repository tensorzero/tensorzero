use async_trait::async_trait;

use crate::context::SimpleToolContext;
use crate::error::ToolResult;
use crate::tool_metadata::ToolMetadata;

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
/// Note that `SimpleTool` extends [`ToolMetadata`], so you must implement
/// both traits.
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
/// use durable_tools::{SimpleTool, SimpleToolContext, ToolResult, ToolMetadata, async_trait};
/// use schemars::JsonSchema;
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
/// #[derive(Default)]
/// struct SearchTool;
///
/// impl ToolMetadata for SearchTool {
///     type SideInfo = ();
///     type Output = SearchResult;
///     type LlmParams = SearchParams;
///
///     fn name(&self) -> Cow<'static, str> {
///         Cow::Borrowed("search")
///     }
///
///     fn description(&self) -> Cow<'static, str> {
///         Cow::Borrowed("Search the web")
///     }
///     // parameters_schema() is automatically derived from LlmParams
/// }
///
/// #[async_trait]
/// impl SimpleTool for SearchTool {
///     async fn execute(
///         llm_params: <Self as ToolMetadata>::LlmParams,
///         _side_info: <Self as ToolMetadata>::SideInfo,
///         ctx: SimpleToolContext<'_>,
///         idempotency_key: &str,
///     ) -> ToolResult<<Self as ToolMetadata>::Output> {
///         // Use idempotency_key for external API calls
///         let results = external_search_api(&llm_params.query, idempotency_key).await?;
///         Ok(SearchResult { results })
///     }
/// }
/// ```
#[async_trait]
pub trait SimpleTool: ToolMetadata {
    /// Execute the tool logic.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - Parameters provided by the LLM
    /// * `side_info` - Side information provided at call time (hidden from LLM)
    /// * `ctx` - The simple tool context (provides database access)
    /// * `idempotency_key` - A unique key for this execution (use for external API calls)
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: Self::SideInfo,
        ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> ToolResult<Self::Output>;
}
