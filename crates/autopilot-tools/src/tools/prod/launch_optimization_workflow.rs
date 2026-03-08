//! Tool for launching optimization workflows and polling until completion.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;

use autopilot_client::AutopilotSideInfo;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{InferenceFilter, OrderBy};
use tensorzero_core::optimization::{
    OptimizationJobHandle, OptimizationJobInfo, UninitializedOptimizerInfo,
};
use tensorzero_optimizers::endpoints::{LaunchOptimizationWorkflowParams, OptimizationDataSource};

/// Parameters for the launch_optimization_workflow tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
    /// Provide either an inference query (e.g. `output_source`, `filters`) or `dataset_name`, not both.
    #[serde(default)]
    pub output_source: Option<InferenceOutputSource>,
    /// Name of the dataset to use as training data.
    /// Provide either an inference query (e.g. `output_source`, `filters`) or `dataset_name`, not both.
    #[serde(default)]
    pub dataset_name: Option<String>,
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    type SideInfo = AutopilotSideInfo;
    type Output = LaunchOptimizationWorkflowToolOutput;
    type LlmParams = LaunchOptimizationWorkflowToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::LAUNCH_OPTIMIZATION_WORKFLOW_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "LaunchOptimizationWorkflowToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::LAUNCH_OPTIMIZATION_WORKFLOW_TOOL_OUTPUT
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "LaunchOptimizationWorkflowToolOutput".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("launch_optimization_workflow")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Launch an optimization workflow (fine-tuning, prompt optimization, etc.) \
             using stored inferences or a dataset and poll until completion.",
        )
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(default_max_wait_secs())
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Launch an optimization workflow using stored inferences or a dataset. You must provide either `output_source` or `dataset_name` (but not both).",
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
                    "description": "Optional variant name to filter inferences by (defaults to all variants). Only used with `output_source`."
                },
                "output_source": {
                    "type": "string",
                    "enum": ["none", "inference", "demonstration"],
                    "description": "Source of the inference output. `inference` returns the original output, `demonstration` returns manually-curated output if available, `none` returns no output. Provide either `output_source` or `dataset_name`, not both."
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the dataset to use as training data. Provide either an inference query (e.g. `output_source`, `filters`) or `dataset_name`, not both."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of inferences to use. Only used with `output_source`."
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination. Only used with `output_source`."
                },
                "val_fraction": {
                    "type": "number",
                    "description": "Fraction of data for validation (0.0 to 1.0, exclusive)."
                },
                "optimizer_config": {
                    "type": "object",
                    "description": "The optimizer configuration. Set `type` to one of: `openai_sft`, `fireworks_sft`, `gcp_vertex_gemini_sft`, `together_sft`, `dicl`, `gepa`. Required fields depend on the type chosen. For SFT types (openai_sft, fireworks_sft, gcp_vertex_gemini_sft, together_sft): `model` is required. For dicl: `embedding_model`, `variant_name`, and `function_name` are required. For gepa: `function_name`, `evaluation_name`, `analysis_model`, and `mutation_model` are required.",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Optimizer type. One of: openai_sft, fireworks_sft, gcp_vertex_gemini_sft, together_sft, dicl, gepa."
                        },
                        "model": {
                            "type": "string",
                            "description": "The model to fine-tune or use. Required for SFT types and optional for dicl."
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Batch size for training (openai_sft, fireworks_sft in tokens, dicl for embeddings, gepa for samples per iteration)."
                        },
                        "learning_rate_multiplier": {
                            "type": "number",
                            "description": "Learning rate multiplier (openai_sft, gcp_vertex_gemini_sft)."
                        },
                        "n_epochs": {
                            "type": "integer",
                            "description": "Number of training epochs (openai_sft, gcp_vertex_gemini_sft, together_sft)."
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility (openai_sft, gcp_vertex_gemini_sft, gepa)."
                        },
                        "suffix": {
                            "type": "string",
                            "description": "Suffix for the fine-tuned model name (openai_sft, together_sft)."
                        },
                        "epochs": {
                            "type": "integer",
                            "description": "Number of training epochs (fireworks_sft)."
                        },
                        "learning_rate": {
                            "type": "number",
                            "description": "Learning rate (fireworks_sft, together_sft)."
                        },
                        "max_context_length": {
                            "type": "integer",
                            "description": "Maximum context length (fireworks_sft)."
                        },
                        "lora_rank": {
                            "type": "integer",
                            "description": "Rank of the LoRA matrix (fireworks_sft)."
                        },
                        "early_stop": {
                            "type": "boolean",
                            "description": "Whether to enable early stopping (fireworks_sft)."
                        },
                        "display_name": {
                            "type": "string",
                            "description": "Display name for the fine-tuning job (fireworks_sft)."
                        },
                        "output_model": {
                            "type": "string",
                            "description": "Model ID for the resulting fine-tuned model (fireworks_sft)."
                        },
                        "deploy_after_training": {
                            "type": "boolean",
                            "description": "Whether to deploy the model after training (fireworks_sft)."
                        },
                        "adapter_size": {
                            "type": "integer",
                            "description": "Adapter size for fine-tuning (gcp_vertex_gemini_sft)."
                        },
                        "tuned_model_display_name": {
                            "type": "string",
                            "description": "Display name for the tuned model (gcp_vertex_gemini_sft)."
                        },
                        "n_checkpoints": {
                            "type": "integer",
                            "description": "Number of checkpoints to save (together_sft). Default: 1."
                        },
                        "warmup_ratio": {
                            "type": "number",
                            "description": "Warmup ratio (together_sft). Default: 0.0."
                        },
                        "embedding_model": {
                            "type": "string",
                            "description": "The embedding model to use (dicl). Required for dicl."
                        },
                        "variant_name": {
                            "type": "string",
                            "description": "Name for the variant to create (dicl). Required for dicl."
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function to optimize (dicl, gepa). Required for dicl and gepa."
                        },
                        "dimensions": {
                            "type": "integer",
                            "description": "Dimensions of the embeddings (dicl). Uses model default if not specified."
                        },
                        "max_concurrency": {
                            "type": "integer",
                            "description": "Maximum concurrency (dicl, gepa). Default: 10."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of nearest neighbors (dicl). Default: 10."
                        },
                        "append_to_existing_variants": {
                            "type": "boolean",
                            "description": "Whether to append to existing variants (dicl). Default: false."
                        },
                        "evaluation_name": {
                            "type": "string",
                            "description": "Name of the evaluation used to score candidate variants (gepa). Required for gepa."
                        },
                        "analysis_model": {
                            "type": "string",
                            "description": "Model for analysis (gepa). Required for gepa."
                        },
                        "mutation_model": {
                            "type": "string",
                            "description": "Model for mutation (gepa). Required for gepa."
                        },
                        "initial_variants": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of variant names to initialize with (gepa)."
                        },
                        "variant_prefix": {
                            "type": "string",
                            "description": "Prefix for newly created optimized variants (gepa)."
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum training iterations (gepa). Default: 1."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Client timeout in seconds (gepa). Default: 300."
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max tokens for analysis/mutation model calls (gepa)."
                        }
                    },
                    "required": ["type"]
                }
            },
            "required": ["function_name", "template_variant_name", "optimizer_config"],
            "additionalProperties": false
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }

    fn strict(&self) -> bool {
        false // Too many optional parameters for anthropic
    }
}

#[async_trait]
impl TaskTool for LaunchOptimizationWorkflowTool {
    type ExtraState = ();
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Step 1: Launch the optimization workflow
        let job_handle: OptimizationJobHandle = ctx
            .step("launch", llm_params.clone(), |params, state| async move {
                let data_source = OptimizationDataSource::from_flat_fields(
                    params.output_source,
                    params.dataset_name,
                    params.query_variant_name,
                    params.filters,
                    params.order_by,
                    params.limit,
                    params.offset,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                let launch_params = LaunchOptimizationWorkflowParams {
                    function_name: params.function_name,
                    template_variant_name: params.template_variant_name,
                    data_source,
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
        let poll_interval = Duration::from_secs(side_info.optimization.poll_interval_secs);
        let max_wait_secs = side_info.optimization.max_wait_secs as i64;
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
                OptimizationJobInfo::Completed { .. } | OptimizationJobInfo::Failed { .. } => {
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
