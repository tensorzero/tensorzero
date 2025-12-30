//! Tool for launching optimization workflows.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{InferenceFilter, OrderBy};
use tensorzero_core::optimization::UninitializedOptimizerInfo;
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;

use crate::types::AutopilotToolSideInfo;

/// Parameters for the launch_optimization_workflow tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LaunchOptimizationWorkflowToolParams {
    /// The function name to optimize.
    pub function_name: String,
    /// The variant name to use as a template for rendering inferences.
    pub template_variant_name: String,
    /// Optional variant name to filter inferences by (defaults to all variants).
    #[serde(default)]
    pub query_variant_name: Option<String>,
    /// Optional filters to apply when querying inferences.
    #[serde(default)]
    pub filters: Option<InferenceFilter>,
    /// Source of the output data (inference output, demonstration, etc.).
    pub output_source: InferenceOutputSource,
    /// Optional ordering for the inferences.
    #[serde(default)]
    pub order_by: Option<Vec<OrderBy>>,
    /// Maximum number of inferences to use.
    #[serde(default)]
    pub limit: Option<u32>,
    /// Offset for pagination.
    #[serde(default)]
    pub offset: Option<u32>,
    /// Fraction of data to use for validation (0.0 to 1.0, exclusive).
    #[serde(default)]
    pub val_fraction: Option<f64>,
    /// The optimizer configuration (e.g., SFT, DPO, MIPROv2).
    pub optimizer_config: UninitializedOptimizerInfo,
}

/// Response from launching an optimization workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchOptimizationWorkflowToolOutput {
    /// The encoded job handle that can be used to poll optimization status.
    pub job_handle: String,
}

/// Tool for launching optimization workflows.
///
/// This tool queries stored inferences, renders them with a template variant,
/// and launches an optimization job (e.g., fine-tuning, prompt optimization).
/// Returns a job handle that can be used to poll the optimization status.
#[derive(Default)]
pub struct LaunchOptimizationWorkflowTool;

impl ToolMetadata for LaunchOptimizationWorkflowTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = LaunchOptimizationWorkflowToolOutput;
    type LlmParams = LaunchOptimizationWorkflowToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("launch_optimization_workflow")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Launch an optimization workflow (fine-tuning, prompt optimization, etc.) \
             using stored inferences. Returns a job handle for polling status.",
        )
    }
}

#[async_trait]
impl SimpleTool for LaunchOptimizationWorkflowTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Convert tool params to the endpoint params
        let params = LaunchOptimizationWorkflowParams {
            function_name: llm_params.function_name,
            template_variant_name: llm_params.template_variant_name,
            query_variant_name: llm_params.query_variant_name,
            filters: llm_params.filters,
            output_source: llm_params.output_source,
            order_by: llm_params.order_by,
            limit: llm_params.limit,
            offset: llm_params.offset,
            val_fraction: llm_params.val_fraction,
            optimizer_config: llm_params.optimizer_config,
        };

        let job_handle = ctx
            .client()
            .launch_optimization_workflow(params)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))?;

        Ok(LaunchOptimizationWorkflowToolOutput { job_handle })
    }
}
