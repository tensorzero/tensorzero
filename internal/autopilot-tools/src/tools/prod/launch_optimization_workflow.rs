//! Tool for launching optimization workflows and polling until completion.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{TaskTool, ToolContext, ToolError, ToolMetadata, ToolResult};

use autopilot_client::OptimizationWorkflowSideInfo;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{InferenceFilter, OrderBy};
use tensorzero_core::optimization::dicl::UninitializedDiclOptimizationConfig;
use tensorzero_core::optimization::fireworks_sft::UninitializedFireworksSFTConfig;
use tensorzero_core::optimization::gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig;
use tensorzero_core::optimization::gepa::UninitializedGEPAConfig;
use tensorzero_core::optimization::openai_sft::UninitializedOpenAISFTConfig;
use tensorzero_core::optimization::together_sft::UninitializedTogetherSFTConfig;
use tensorzero_core::optimization::{
    OptimizationJobHandle, OptimizationJobInfo, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};
use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;

// =============================================================================
// LLM-Facing Types (OpenAI-Compatible Schema)
// =============================================================================

/// Simplified optimizer type enum for LLM interaction.
///
/// This enum has a simple schema that OpenAI can handle.
/// Note: OpenAI RFT is excluded because it requires a complex grader configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LlmOptimizerType {
    /// Dynamic In-Context Learning - adds relevant examples to prompts dynamically.
    /// Fast, no external API calls for training. Requires: embedding_model.
    Dicl,
    /// OpenAI Supervised Fine-Tuning - trains a custom model on your data.
    /// Requires: model (e.g., "gpt-4o-mini-2024-07-18").
    OpenaiSft,
    /// Fireworks AI Supervised Fine-Tuning.
    /// Requires: model.
    FireworksSft,
    /// Google Cloud Vertex AI Gemini Fine-Tuning.
    /// Requires: model.
    GcpVertexGeminiSft,
    /// Genetic Evolution Prompt Algorithm - evolves prompts using genetic algorithms.
    /// Requires: gepa_function_name, evaluation_name, analysis_model, mutation_model.
    Gepa,
    /// Together AI Supervised Fine-Tuning.
    /// Requires: model.
    TogetherSft,
}

/// Simplified optimizer configuration for LLM interaction.
///
/// Contains common options across optimizer types. The `optimizer_type` field
/// determines which options are required/used.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LlmOptimizerConfig {
    /// The type of optimizer to use.
    #[serde(rename = "type")]
    pub optimizer_type: LlmOptimizerType,

    // --- DICL options ---
    /// Embedding model for DICL (e.g., "text-embedding-3-small").
    /// Required for: dicl
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,

    /// Number of examples to include in prompt for DICL. Default: 10.
    /// Used by: dicl
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,

    // --- Fine-tuning options (SFT) ---
    /// Model to fine-tune (e.g., "gpt-4o-mini-2024-07-18", "meta-llama/Llama-3-8b").
    /// Required for: openai_sft, fireworks_sft, gcp_vertex_gemini_sft, together_sft
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Number of training epochs.
    /// Used by: openai_sft, fireworks_sft, gcp_vertex_gemini_sft, together_sft
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,

    /// Learning rate multiplier.
    /// Used by: openai_sft, gcp_vertex_gemini_sft
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,

    /// Suffix for the fine-tuned model name.
    /// Used by: openai_sft, together_sft
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    // --- GEPA options ---
    /// Function name for GEPA optimization.
    /// Required for: gepa (note: this is different from the top-level function_name)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gepa_function_name: Option<String>,

    /// Evaluation name for GEPA.
    /// Required for: gepa
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_name: Option<String>,

    /// Analysis model for GEPA (e.g., "openai::gpt-4o").
    /// Required for: gepa
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_model: Option<String>,

    /// Mutation model for GEPA.
    /// Required for: gepa
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutation_model: Option<String>,
}

/// Simplified output source enum for LLM interaction.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LlmOutputSource {
    /// Use the original model output from the inference.
    #[serde(alias = "inference")]
    InferenceOutput,
    /// Use human-provided demonstration as the output.
    Demonstration,
    /// No output (for creating datapoints without output).
    None,
}

/// Simplified parameters for the launch_optimization_workflow tool.
///
/// This type is what the LLM sees and generates. It has a simple JSON Schema
/// that is compatible with OpenAI's function calling API.
///
/// For advanced options (filters, order_by), use the API directly.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LaunchOptimizationWorkflowLlmParams {
    /// The function name to optimize (e.g., "extract_entities", "summarize").
    pub function_name: String,

    /// The variant name to use as a template for rendering inferences.
    /// This variant's prompt template will be used to format the training data.
    pub template_variant_name: String,

    /// Optional variant name to filter inferences by.
    /// If not specified, inferences from all variants are included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_variant_name: Option<String>,

    /// Source of the output data for training.
    pub output_source: LlmOutputSource,

    /// Maximum number of inferences to use for optimization.
    /// Recommended: 50-500 for DICL, 100-10000 for fine-tuning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// Offset for pagination (skip this many inferences).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,

    /// Fraction of data to use for validation (0.0 to 1.0, exclusive).
    /// E.g., 0.1 means 10% validation, 90% training.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val_fraction: Option<f64>,

    /// The optimizer configuration.
    pub optimizer_config: LlmOptimizerConfig,
}

// =============================================================================
// Conversion: LLM Types -> Internal Types
// =============================================================================

impl From<LlmOutputSource> for InferenceOutputSource {
    fn from(source: LlmOutputSource) -> Self {
        match source {
            LlmOutputSource::InferenceOutput => InferenceOutputSource::Inference,
            LlmOutputSource::Demonstration => InferenceOutputSource::Demonstration,
            LlmOutputSource::None => InferenceOutputSource::None,
        }
    }
}

/// Context needed for converting LLM optimizer config to internal types.
/// DICL requires function_name and variant_name from the top-level params.
struct OptimizerConversionContext<'a> {
    function_name: &'a str,
    variant_name: &'a str,
}

impl LlmOptimizerConfig {
    /// Convert to internal UninitializedOptimizerInfo with validation.
    fn into_optimizer_info(
        self,
        ctx: &OptimizerConversionContext<'_>,
    ) -> Result<UninitializedOptimizerInfo, ToolError> {
        let inner = match self.optimizer_type {
            LlmOptimizerType::Dicl => {
                let embedding_model =
                    self.embedding_model.ok_or_else(|| ToolError::Validation {
                        message: "DICL optimizer requires `embedding_model` (e.g., `text-embedding-3-small`)".into(),
                    })?;
                UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
                    embedding_model,
                    variant_name: ctx.variant_name.to_string(),
                    function_name: ctx.function_name.to_string(),
                    k: self.k.unwrap_or(10),
                    ..Default::default()
                })
            }

            LlmOptimizerType::OpenaiSft => {
                let model = self.model.ok_or_else(|| ToolError::Validation {
                    message:
                        "OpenAI SFT optimizer requires `model` (e.g., `gpt-4o-mini-2024-07-18`)"
                            .into(),
                })?;
                UninitializedOptimizerConfig::OpenAISFT(UninitializedOpenAISFTConfig {
                    model,
                    n_epochs: self.n_epochs.map(|n| n as usize),
                    learning_rate_multiplier: self.learning_rate_multiplier,
                    suffix: self.suffix,
                    ..Default::default()
                })
            }

            LlmOptimizerType::FireworksSft => {
                let model = self.model.ok_or_else(|| ToolError::Validation {
                    message: "Fireworks SFT optimizer requires `model`".into(),
                })?;
                UninitializedOptimizerConfig::FireworksSFT(UninitializedFireworksSFTConfig {
                    model,
                    epochs: self.n_epochs.map(|n| n as usize),
                    ..Default::default()
                })
            }

            LlmOptimizerType::GcpVertexGeminiSft => {
                let model = self.model.ok_or_else(|| ToolError::Validation {
                    message: "GCP Vertex Gemini SFT optimizer requires `model`".into(),
                })?;
                UninitializedOptimizerConfig::GCPVertexGeminiSFT(
                    UninitializedGCPVertexGeminiSFTConfig {
                        model,
                        n_epochs: self.n_epochs.map(|n| n as usize),
                        learning_rate_multiplier: self.learning_rate_multiplier,
                        ..Default::default()
                    },
                )
            }

            LlmOptimizerType::Gepa => {
                let function_name =
                    self.gepa_function_name
                        .ok_or_else(|| ToolError::Validation {
                            message: "GEPA optimizer requires `gepa_function_name`".into(),
                        })?;
                let evaluation_name =
                    self.evaluation_name.ok_or_else(|| ToolError::Validation {
                        message: "GEPA optimizer requires `evaluation_name`".into(),
                    })?;
                let analysis_model = self.analysis_model.ok_or_else(|| ToolError::Validation {
                    message: "GEPA optimizer requires `analysis_model`".into(),
                })?;
                let mutation_model = self.mutation_model.ok_or_else(|| ToolError::Validation {
                    message: "GEPA optimizer requires `mutation_model`".into(),
                })?;
                UninitializedOptimizerConfig::GEPA(UninitializedGEPAConfig {
                    function_name,
                    evaluation_name,
                    analysis_model,
                    mutation_model,
                    initial_variants: None,
                    variant_prefix: None,
                    batch_size: 5,       // default_batch_size()
                    max_iterations: 1,   // default_max_iterations()
                    max_concurrency: 10, // default_max_concurrency()
                    seed: None,
                    timeout: 300,                         // default_timeout()
                    include_inference_for_mutation: true, // default
                    retries: Default::default(),
                    max_tokens: None,
                })
            }

            LlmOptimizerType::TogetherSft => {
                let model = self.model.ok_or_else(|| ToolError::Validation {
                    message: "Together SFT optimizer requires `model`".into(),
                })?;
                UninitializedOptimizerConfig::TogetherSFT(Box::new(
                    UninitializedTogetherSFTConfig {
                        model,
                        n_epochs: self.n_epochs.unwrap_or(1),
                        suffix: self.suffix,
                        ..Default::default()
                    },
                ))
            }
        };

        Ok(UninitializedOptimizerInfo { inner })
    }
}

impl TryFrom<LaunchOptimizationWorkflowLlmParams> for LaunchOptimizationWorkflowToolParams {
    type Error = ToolError;

    fn try_from(llm: LaunchOptimizationWorkflowLlmParams) -> Result<Self, Self::Error> {
        // Convert optimizer config first while we still have references to the field values
        let ctx = OptimizerConversionContext {
            function_name: &llm.function_name,
            variant_name: &llm.template_variant_name,
        };
        let optimizer_config = llm.optimizer_config.into_optimizer_info(&ctx)?;

        Ok(LaunchOptimizationWorkflowToolParams {
            function_name: llm.function_name,
            template_variant_name: llm.template_variant_name,
            query_variant_name: llm.query_variant_name,
            filters: None, // Not exposed to LLM - use API directly for advanced filtering
            output_source: llm.output_source.into(),
            order_by: None, // Not exposed to LLM
            limit: llm.limit,
            offset: llm.offset,
            val_fraction: llm.val_fraction,
            optimizer_config,
        })
    }
}

// =============================================================================
// Internal Types (Full Feature Set)
// =============================================================================

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
    type LlmParams = LaunchOptimizationWorkflowLlmParams;

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
        // Use schemars settings that inline subschemas to avoid $ref/$defs
        // which OpenAI's function calling API doesn't support
        let settings = schemars::generate::SchemaSettings::default().with(|s| {
            s.inline_subschemas = true;
        });
        let generator = settings.into_generator();
        Ok(generator.into_root_schema_for::<LaunchOptimizationWorkflowLlmParams>())
    }
}

#[async_trait]
impl TaskTool for LaunchOptimizationWorkflowTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Convert LLM params to internal params with validation
        let params: LaunchOptimizationWorkflowToolParams = llm_params.try_into()?;

        // Step 1: Launch the optimization workflow
        let job_handle: OptimizationJobHandle = ctx
            .step("launch", params.clone(), |params, state| async move {
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
                        return Err(ToolError::Validation {
                            message: format!(
                                "Optimization timed out after {max_wait_secs} seconds"
                            ),
                        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_is_openai_compatible() {
        // Use the same schema generation as parameters_schema()
        let settings = schemars::generate::SchemaSettings::default().with(|s| {
            s.inline_subschemas = true;
        });
        let generator = settings.into_generator();
        let schema = generator.into_root_schema_for::<LaunchOptimizationWorkflowLlmParams>();
        let schema_json =
            serde_json::to_string_pretty(&schema).expect("Schema should serialize to JSON");

        // OpenAI doesn't support $ref or $defs
        assert!(
            !schema_json.contains("\"$ref\""),
            "Schema should not contain $ref for OpenAI compatibility. Schema:\n{schema_json}"
        );
        assert!(
            !schema_json.contains("\"$defs\""),
            "Schema should not contain $defs for OpenAI compatibility. Schema:\n{schema_json}"
        );
    }

    #[test]
    fn test_dicl_conversion_success() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::Dicl,
            embedding_model: Some("text-embedding-3-small".into()),
            k: Some(5),
            model: None,
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: None,
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        };

        let ctx = OptimizerConversionContext {
            function_name: "my_function",
            variant_name: "my_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(result.is_ok(), "DICL conversion should succeed");

        let info = result.expect("Should have optimizer info");
        match info.inner {
            UninitializedOptimizerConfig::Dicl(cfg) => {
                assert_eq!(
                    cfg.embedding_model, "text-embedding-3-small",
                    "Embedding model should match"
                );
                assert_eq!(cfg.k, 5, "k should match");
                assert_eq!(
                    cfg.function_name, "my_function",
                    "Function name should be derived from context"
                );
                assert_eq!(
                    cfg.variant_name, "my_variant",
                    "Variant name should be derived from context"
                );
            }
            _ => panic!("Expected Dicl variant"),
        }
    }

    #[test]
    fn test_dicl_missing_embedding_model() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::Dicl,
            embedding_model: None, // Missing!
            k: None,
            model: None,
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: None,
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        };

        let ctx = OptimizerConversionContext {
            function_name: "my_function",
            variant_name: "my_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(
            result.is_err(),
            "DICL conversion should fail without embedding_model"
        );

        let err = result.expect_err("Should have error");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("embedding_model"),
            "Error message should mention embedding_model: {err_msg}"
        );
    }

    #[test]
    fn test_openai_sft_conversion_success() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::OpenaiSft,
            embedding_model: None,
            k: None,
            model: Some("gpt-4o-mini-2024-07-18".into()),
            n_epochs: Some(3),
            learning_rate_multiplier: Some(1.5),
            suffix: Some("my-model".into()),
            gepa_function_name: None,
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        };

        let ctx = OptimizerConversionContext {
            function_name: "my_function",
            variant_name: "my_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(result.is_ok(), "OpenAI SFT conversion should succeed");

        let info = result.expect("Should have optimizer info");
        match info.inner {
            UninitializedOptimizerConfig::OpenAISFT(cfg) => {
                assert_eq!(cfg.model, "gpt-4o-mini-2024-07-18", "Model should match");
                assert_eq!(cfg.n_epochs, Some(3), "n_epochs should match");
                assert_eq!(
                    cfg.learning_rate_multiplier,
                    Some(1.5),
                    "Learning rate multiplier should match"
                );
                assert_eq!(cfg.suffix, Some("my-model".into()), "Suffix should match");
            }
            _ => panic!("Expected OpenAISFT variant"),
        }
    }

    #[test]
    fn test_openai_sft_missing_model() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::OpenaiSft,
            embedding_model: None,
            k: None,
            model: None, // Missing!
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: None,
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        };

        let ctx = OptimizerConversionContext {
            function_name: "my_function",
            variant_name: "my_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(
            result.is_err(),
            "OpenAI SFT conversion should fail without model"
        );

        let err = result.expect_err("Should have error");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("model"),
            "Error message should mention model: {err_msg}"
        );
    }

    #[test]
    fn test_full_params_conversion() {
        let llm_params = LaunchOptimizationWorkflowLlmParams {
            function_name: "extract_entities".into(),
            template_variant_name: "baseline".into(),
            query_variant_name: Some("v1".into()),
            output_source: LlmOutputSource::InferenceOutput,
            limit: Some(100),
            offset: Some(10),
            val_fraction: Some(0.1),
            optimizer_config: LlmOptimizerConfig {
                optimizer_type: LlmOptimizerType::Dicl,
                embedding_model: Some("text-embedding-3-small".into()),
                k: Some(5),
                model: None,
                n_epochs: None,
                learning_rate_multiplier: None,
                suffix: None,
                gepa_function_name: None,
                evaluation_name: None,
                analysis_model: None,
                mutation_model: None,
            },
        };

        let result: Result<LaunchOptimizationWorkflowToolParams, _> = llm_params.try_into();
        assert!(result.is_ok(), "Full params conversion should succeed");

        let params = result.expect("Should have params");
        assert_eq!(
            params.function_name, "extract_entities",
            "Function name should match"
        );
        assert_eq!(
            params.template_variant_name, "baseline",
            "Template variant name should match"
        );
        assert_eq!(
            params.query_variant_name,
            Some("v1".into()),
            "Query variant name should match"
        );
        assert_eq!(
            params.output_source,
            InferenceOutputSource::Inference,
            "Output source should match"
        );
        assert_eq!(params.limit, Some(100), "Limit should match");
        assert_eq!(params.offset, Some(10), "Offset should match");
        assert_eq!(params.val_fraction, Some(0.1), "Val fraction should match");
        assert!(
            params.filters.is_none(),
            "Filters should be None (not exposed to LLM)"
        );
        assert!(
            params.order_by.is_none(),
            "Order by should be None (not exposed to LLM)"
        );
    }

    #[test]
    fn test_output_source_conversion() {
        assert_eq!(
            InferenceOutputSource::from(LlmOutputSource::InferenceOutput),
            InferenceOutputSource::Inference,
            "InferenceOutput should convert to Inference"
        );
        assert_eq!(
            InferenceOutputSource::from(LlmOutputSource::Demonstration),
            InferenceOutputSource::Demonstration,
            "Demonstration should convert to Demonstration"
        );
        assert_eq!(
            InferenceOutputSource::from(LlmOutputSource::None),
            InferenceOutputSource::None,
            "None should convert to None"
        );
    }

    #[test]
    fn test_gepa_conversion_success() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::Gepa,
            embedding_model: None,
            k: None,
            model: None,
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: Some("my_function".into()),
            evaluation_name: Some("my_eval".into()),
            analysis_model: Some("openai::gpt-4o".into()),
            mutation_model: Some("openai::gpt-4o".into()),
        };

        let ctx = OptimizerConversionContext {
            function_name: "top_level_function",
            variant_name: "top_level_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(result.is_ok(), "GEPA conversion should succeed");

        let info = result.expect("Should have optimizer info");
        match info.inner {
            UninitializedOptimizerConfig::GEPA(cfg) => {
                assert_eq!(
                    cfg.function_name, "my_function",
                    "Function name should match gepa_function_name"
                );
                assert_eq!(
                    cfg.evaluation_name, "my_eval",
                    "Evaluation name should match"
                );
                assert_eq!(
                    cfg.analysis_model, "openai::gpt-4o",
                    "Analysis model should match"
                );
                assert_eq!(
                    cfg.mutation_model, "openai::gpt-4o",
                    "Mutation model should match"
                );
            }
            _ => panic!("Expected GEPA variant"),
        }
    }

    #[test]
    fn test_gepa_missing_required_fields() {
        let llm_config = LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::Gepa,
            embedding_model: None,
            k: None,
            model: None,
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: None, // Missing!
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        };

        let ctx = OptimizerConversionContext {
            function_name: "my_function",
            variant_name: "my_variant",
        };

        let result = llm_config.into_optimizer_info(&ctx);
        assert!(
            result.is_err(),
            "GEPA conversion should fail without required fields"
        );

        let err = result.expect_err("Should have error");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("gepa_function_name"),
            "Error message should mention gepa_function_name: {err_msg}"
        );
    }
}
