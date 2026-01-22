//! Tool for launching optimization workflows and polling until completion.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};

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
                    "enum": ["none", "inference", "demonstration"],
                    "description": "Source of the inference output. 'inference' returns the original output, 'demonstration' returns manually-curated output if available, 'none' returns no output."
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
                    "description": "The optimizer configuration. Use 'type' to select the optimizer.",
                    "anyOf": [
                        {
                            "type": "object",
                            "description": "OpenAI supervised fine-tuning configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["openai_sft"],
                                    "description": "Optimizer type identifier."
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The model to fine-tune (e.g., 'gpt-4.1-2025-04-14')."
                                },
                                "batch_size": {
                                    "type": "integer",
                                    "description": "Batch size for training."
                                },
                                "learning_rate_multiplier": {
                                    "type": "number",
                                    "description": "Learning rate multiplier."
                                },
                                "n_epochs": {
                                    "type": "integer",
                                    "description": "Number of training epochs."
                                },
                                "seed": {
                                    "type": "integer",
                                    "description": "Random seed for reproducibility."
                                },
                                "suffix": {
                                    "type": "string",
                                    "description": "Suffix for the fine-tuned model name in OpenAI."
                                }
                            },
                            "required": ["type", "model"]
                        },
                        {
                            "type": "object",
                            "description": "Fireworks supervised fine-tuning configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["fireworks_sft"],
                                    "description": "Optimizer type identifier."
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The model to fine-tune."
                                },
                                "epochs": {
                                    "type": "integer",
                                    "description": "Number of training epochs."
                                },
                                "learning_rate": {
                                    "type": "number",
                                    "description": "Learning rate for training."
                                },
                                "batch_size": {
                                    "type": "integer",
                                    "description": "Batch size (in tokens) for training."
                                },
                                "max_context_length": {
                                    "type": "integer",
                                    "description": "Maximum context length."
                                },
                                "lora_rank": {
                                    "type": "integer",
                                    "description": "Rank of the LoRA matrix."
                                },
                                "early_stop": {
                                    "type": "boolean",
                                    "description": "Whether to enable early stopping."
                                },
                                "display_name": {
                                    "type": "string",
                                    "description": "Display name for the fine-tuning job."
                                },
                                "output_model": {
                                    "type": "string",
                                    "description": "Model ID for the resulting fine-tuned model."
                                },
                                "deploy_after_training": {
                                    "type": "boolean",
                                    "description": "Whether to deploy the model after training."
                                }
                            },
                            "required": ["type", "model"]
                        },
                        {
                            "type": "object",
                            "description": "GCP Vertex Gemini supervised fine-tuning configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["gcp_vertex_gemini_sft"],
                                    "description": "Optimizer type identifier."
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The model to fine-tune (e.g., 'gemini-2.5-flash')."
                                },
                                "learning_rate_multiplier": {
                                    "type": "number",
                                    "description": "Learning rate multiplier."
                                },
                                "adapter_size": {
                                    "type": "integer",
                                    "description": "Adapter size for fine-tuning."
                                },
                                "n_epochs": {
                                    "type": "integer",
                                    "description": "Number of training epochs."
                                },
                                "seed": {
                                    "type": "integer",
                                    "description": "Random seed for reproducibility."
                                },
                                "tuned_model_display_name": {
                                    "type": "string",
                                    "description": "Display name for the tuned model."
                                }
                            },
                            "required": ["type", "model"]
                        },
                        {
                            "type": "object",
                            "description": "Together AI supervised fine-tuning configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["together_sft"],
                                    "description": "Optimizer type identifier."
                                },
                                "model": {
                                    "type": "string",
                                    "description": "The base model to fine-tune."
                                },
                                "n_epochs": {
                                    "type": "integer",
                                    "description": "Number of training epochs. Default: 1."
                                },
                                "n_checkpoints": {
                                    "type": "integer",
                                    "description": "Number of checkpoints to save. Default: 1."
                                },
                                "learning_rate": {
                                    "type": "number",
                                    "description": "Learning rate. Default: 0.00001."
                                },
                                "warmup_ratio": {
                                    "type": "number",
                                    "description": "Warmup ratio. Default: 0.0."
                                },
                                "suffix": {
                                    "type": "string",
                                    "description": "Suffix for the fine-tuned model name."
                                }
                            },
                            "required": ["type", "model"]
                        },
                        {
                            "type": "object",
                            "description": "Dynamic In-Context Learning (DICL) optimization configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["dicl"],
                                    "description": "Optimizer type identifier."
                                },
                                "embedding_model": {
                                    "type": "string",
                                    "description": "The embedding model to use (e.g., 'openai::text-embedding-3-small')."
                                },
                                "variant_name": {
                                    "type": "string",
                                    "description": "Name for the DICL variant to create."
                                },
                                "function_name": {
                                    "type": "string",
                                    "description": "Name of the function to optimize."
                                },
                                "dimensions": {
                                    "type": "integer",
                                    "description": "Dimensions of the embeddings. Uses model default if not specified."
                                },
                                "batch_size": {
                                    "type": "integer",
                                    "description": "Batch size for getting embeddings. Default: 128."
                                },
                                "max_concurrency": {
                                    "type": "integer",
                                    "description": "Maximum concurrency for embeddings. Default: 10."
                                },
                                "k": {
                                    "type": "integer",
                                    "description": "Number of nearest neighbors for DICL. Default: 10."
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Model for the DICL variant. Default: 'openai::gpt-5-mini-2025-08-07'."
                                },
                                "append_to_existing_variants": {
                                    "type": "boolean",
                                    "description": "Whether to append to existing variants. Default: false."
                                }
                            },
                            "required": ["type", "embedding_model", "variant_name", "function_name"]
                        },
                        {
                            "type": "object",
                            "description": "GEPA (Genetic Evolution with Pareto Analysis) prompt optimization configuration.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["gepa"],
                                    "description": "Optimizer type identifier."
                                },
                                "function_name": {
                                    "type": "string",
                                    "description": "Name of the function to optimize."
                                },
                                "evaluation_name": {
                                    "type": "string",
                                    "description": "Name of the evaluation used to score candidate variants."
                                },
                                "analysis_model": {
                                    "type": "string",
                                    "description": "Model for analysis (e.g., 'anthropic::claude-sonnet-4-5')."
                                },
                                "mutation_model": {
                                    "type": "string",
                                    "description": "Model for mutation (e.g., 'anthropic::claude-sonnet-4-5')."
                                },
                                "initial_variants": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Optional list of variant names to initialize GEPA with."
                                },
                                "variant_prefix": {
                                    "type": "string",
                                    "description": "Prefix for newly created optimized variants."
                                },
                                "batch_size": {
                                    "type": "integer",
                                    "description": "Number of samples to analyze per iteration. Default: 5."
                                },
                                "max_iterations": {
                                    "type": "integer",
                                    "description": "Maximum training iterations. Default: 1."
                                },
                                "max_concurrency": {
                                    "type": "integer",
                                    "description": "Maximum concurrent inference calls. Default: 10."
                                },
                                "seed": {
                                    "type": "integer",
                                    "description": "Random seed for reproducibility."
                                },
                                "timeout": {
                                    "type": "integer",
                                    "description": "Client timeout in seconds. Default: 300."
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Max tokens for analysis/mutation model calls."
                                }
                            },
                            "required": ["type", "function_name", "evaluation_name", "analysis_model", "mutation_model"]
                        }
                    ]
                }
            },
            "required": ["function_name", "template_variant_name", "output_source", "optimizer_config"]
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
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
