//! Tool for launching optimization workflows and polling until completion.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{SerializableToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;

use autopilot_client::OptimizationWorkflowSideInfo;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{InferenceFilter, OrderBy};
use tensorzero_core::optimization::{
    OptimizationJobHandle, OptimizationJobInfo, UninitializedOptimizerInfo,
};
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;

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

/// Response from the optimization workflow tool.
#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchOptimizationWorkflowToolOutput {
    /// The final job info (Completed or Failed).
    pub result: OptimizationJobInfo,
}

/// Tool for launching optimization workflows and polling until completion.
///
/// This tool queries stored inferences, renders them with a template variant,
/// launches an optimization job (e.g., fine-tuning, prompt optimization),
/// and polls until the job completes or fails.
#[derive(Default)]
pub struct LaunchOptimizationWorkflowTool;

impl ToolMetadata for LaunchOptimizationWorkflowTool {
    type SideInfo = OptimizationWorkflowSideInfo;
    type Output = LaunchOptimizationWorkflowToolOutput;
    type LlmParams = LaunchOptimizationWorkflowToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("launch_optimization_workflow")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Launch an optimization workflow (fine-tuning, prompt optimization, etc.) \
             using stored inferences and poll until completion.",
        )
    }

    fn timeout() -> Duration {
        Duration::from_secs(default_max_wait_secs())
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Launch an optimization workflow using stored inferences.",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The function name to optimize."
                },
                "template_variant_name": {
                    "type": "string",
                    "description": "The variant name to use as a template for rendering inferences."
                },
                "query_variant_name": {
                    "type": "string",
                    "description": "Optional variant name to filter inferences by (defaults to all variants)."
                },
                "output_source": {
                    "type": "string",
                    "enum": ["inference_output", "demonstration"],
                    "description": "Source of output data: 'inference_output' (model outputs) or 'demonstration' (human demonstrations)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of inferences to use."
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination."
                },
                "val_fraction": {
                    "type": "number",
                    "description": "Fraction of data for validation (0.0 to 1.0, exclusive)."
                },
                "optimizer_config": {
                    "type": "object",
                    "description": "The optimizer configuration.",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["sft", "dpo", "mipro_v2"],
                            "description": "Optimizer type: 'sft' (supervised fine-tuning), 'dpo' (direct preference optimization), 'mipro_v2' (prompt optimization)."
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Model to fine-tune (for SFT/DPO)."
                        },
                        "num_epochs": {
                            "type": "integer",
                            "description": "Number of training epochs (for SFT/DPO)."
                        }
                    },
                    "required": ["type"]
                }
            },
            "required": ["function_name", "template_variant_name", "output_source", "optimizer_config"]
        });

        serde_json::from_value(schema).map_err(|e| {
            SerializableToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl TaskTool for LaunchOptimizationWorkflowTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Step 1: Launch the optimization workflow
        let job_handle: OptimizationJobHandle = ctx
            .step("launch", llm_params.clone(), |params, state| async move {
                let launch_params = LaunchOptimizationWorkflowParams {
                    function_name: params.function_name,
                    template_variant_name: params.template_variant_name,
                    query_variant_name: params.query_variant_name,
                    filters: params.filters,
                    output_source: params.output_source,
                    order_by: params.order_by,
                    limit: params.limit,
                    offset: params.offset,
                    val_fraction: params.val_fraction,
                    optimizer_config: params.optimizer_config,
                };

                state
                    .t0_client()
                    .launch_optimization_workflow(launch_params)
                    .await
                    .map_err(|e| anyhow::Error::msg(e.to_string()))
            })
            .await?;

        // Step 2: Poll until completion
        let poll_interval = Duration::from_secs(side_info.poll_interval_secs);
        let max_wait_secs = side_info.max_wait_secs as i64;
        let start = ctx.now().await?;
        let mut iteration = 0u32;

        loop {
            // Checkpointed poll
            let status: OptimizationJobInfo = ctx
                .step(
                    &format!("poll_{iteration}"),
                    job_handle.clone(),
                    |handle, state| async move {
                        state
                            .t0_client()
                            .poll_optimization(&handle)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            match &status {
                OptimizationJobInfo::Completed { .. } => {
                    return Ok(LaunchOptimizationWorkflowToolOutput { result: status });
                }
                OptimizationJobInfo::Failed { .. } => {
                    return Ok(LaunchOptimizationWorkflowToolOutput { result: status });
                }
                OptimizationJobInfo::Pending { .. } => {
                    // Check timeout
                    let elapsed = ctx.now().await? - start;
                    if elapsed.num_seconds() > max_wait_secs {
                        return Err(AutopilotToolError::validation(format!(
                            "Optimization timed out after {max_wait_secs} seconds"
                        ))
                        .into());
                    }

                    // Durable sleep before next poll
                    ctx.sleep_for(&format!("wait_{iteration}"), poll_interval)
                        .await?;

                    iteration += 1;
                }
            }
        }
    }
}

fn default_max_wait_secs() -> u64 {
    86400
}
