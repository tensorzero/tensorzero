//! Inference analysis functions for GEPA optimization
//!
//! This module provides functions for:
//! - Analyzing inference outputs to identify errors, improvements, and optimal patterns
//! - Building inputs for the built-in `tensorzero::optimization::gepa::analyze` function
//! - Parsing XML feedback from analysis responses

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPath, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::{datasets::StoredDatapoint, inference::InferenceResponse},
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{Arguments, ContentBlockChatOutput, Role, Template},
    optimization::gepa::GEPAConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use evaluations::stats::EvaluationInfo;

use crate::gepa::validate::FunctionConfigAndTools;

/// Serialize values to JSON with contextual error messages
fn serialize_to_value<T: serde::Serialize>(
    value: &T,
    context: &str,
) -> Result<serde_json::Value, Error> {
    serde_json::to_value(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {context}: {e}"),
        })
    })
}

/// Represents an inference output paired with its analysis feedback
#[derive(Debug, Clone, Serialize)]
pub struct InferenceWithAnalysis {
    pub inference_output: InferenceResponse,
    /// Content blocks from the analyze function response
    pub analysis: Vec<ContentBlockChatOutput>,
    /// Optional datapoint input (only included if include_datapoint_input_for_mutation is true)
    /// Skipped during serialization if None to avoid bloating the mutate function input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datapoint_input: Option<serde_json::Value>,
}

/// Build the input JSON for the analyze function
///
/// # Arguments
/// * `eval_info` - Evaluation information containing the datapoint and inference response
/// * `config_and_tools` - Function configuration with static tools
/// * `variant_config` - Variant configuration (templates, model name)
///
/// # Returns
/// * Template arguments containing function metadata, templates, schemas, and the input/output pair
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    config_and_tools: &FunctionConfigAndTools,
    variant_config: &UninitializedChatCompletionConfig,
) -> Result<Arguments, Error> {
    // Extract all templates from templates.inner HashMap using idiomatic iterators
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| config.path.read().map(|content| (name.clone(), content)))
        .collect::<Result<_, _>>()?;

    // Extract all schemas from function config using the new schemas.inner pattern
    let schemas_map: HashMap<String, serde_json::Value> = config_and_tools
        .function_config
        .schemas()
        .inner
        .iter()
        .map(|(name, schema_with_metadata)| {
            (name.clone(), schema_with_metadata.schema.value.clone())
        })
        .collect();

    // Extract output schema for JSON functions
    let output_schema = match &*config_and_tools.function_config {
        FunctionConfig::Json(params) => Some(params.output_schema.value.clone()),
        FunctionConfig::Chat(_) => None,
    };

    // Extract and serialize tool configuration using into_tool_call_config
    let tools = match &eval_info.datapoint {
        StoredDatapoint::Chat(dp) => {
            match &dp.tool_params {
                Some(tool_params) => {
                    // Clone because into_tool_call_config consumes self
                    let empty_tools = HashMap::new();
                    let tool_config = tool_params.clone().into_tool_call_config(
                        &config_and_tools.function_config,
                        config_and_tools
                            .static_tools
                            .as_ref()
                            .unwrap_or(&empty_tools),
                    )?;
                    // tool_config is Option<ToolCallConfig>
                    // Convert Option<ToolCallConfig> to Option<serde_json::Value>
                    tool_config
                        .map(|config| serialize_to_value(&config, "tool config"))
                        .transpose()?
                }
                None => None,
            }
        }
        StoredDatapoint::Json(_) => None,
    };

    // Serialize the inference output
    let output = serialize_to_value(&eval_info.response, "inference response")?;

    // Get tags from datapoint (if any)
    let tags = match &eval_info.datapoint {
        StoredDatapoint::Chat(dp) => dp.tags.clone(),
        StoredDatapoint::Json(dp) => dp.tags.clone(),
    };

    // Get function name from datapoint
    let function_name = match &eval_info.datapoint {
        StoredDatapoint::Chat(dp) => &dp.function_name,
        StoredDatapoint::Json(dp) => &dp.function_name,
    };

    // Build the input object directly as a Map
    let mut map = serde_json::Map::new();
    map.insert("function_name".to_string(), json!(function_name));
    map.insert("model".to_string(), json!(variant_config.model.as_ref()));
    map.insert("templates".to_string(), json!(templates_map));
    map.insert("schemas".to_string(), json!(schemas_map));
    map.insert("output_schema".to_string(), json!(output_schema));
    map.insert("tools".to_string(), json!(tools));
    map.insert("tags".to_string(), json!(tags));
    map.insert("input".to_string(), json!(eval_info.datapoint.input()));
    map.insert("output".to_string(), json!(output));

    // Extract evaluator scores from evaluations
    let mut evaluations = serde_json::Map::new();
    for (evaluator_name, result_opt) in &eval_info.evaluations {
        let score = result_opt.as_ref().and_then(|value| match value {
            serde_json::Value::Number(n) => n.as_f64().map(|f| f as f32),
            serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        });
        evaluations.insert(evaluator_name.clone(), json!(score));
    }
    map.insert("evaluations".to_string(), json!(evaluations));

    Ok(Arguments(map))
}

/// Analyze inference outputs using the GEPA analyze function
///
/// Calls the built-in `tensorzero::optimization::gepa::analyze` function in parallel
/// with controlled concurrency (up to `gepa_config.max_concurrency` concurrent API calls).
///
/// # Arguments
/// * `gateway_client` - TensorZero gateway client for making inference requests
/// * `evaluation_infos` - Evaluation results containing datapoints and their inference responses
/// * `function_config` - Configuration of the function being optimized
/// * `variant_config` - Configuration of the variant being analyzed
/// * `gepa_config` - GEPA configuration containing analysis_model and max_concurrency settings
///
/// # Returns
/// * Vector of [`InferenceWithAnalysis`] containing each inference paired with its XML analysis feedback
pub async fn analyze_inferences(
    gateway_client: &Client,
    evaluation_infos: &[EvaluationInfo],
    config_and_tools: &FunctionConfigAndTools,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
) -> Result<Vec<InferenceWithAnalysis>, Error> {
    // Early return for empty input - nothing to analyze
    if evaluation_infos.is_empty() {
        return Ok(Vec::new());
    }

    // Extract parameters from GEPA config
    let analysis_model = &gepa_config.analysis_model;
    let max_concurrency = gepa_config.max_concurrency as usize;

    tracing::info!(
        "Analyzing {} inferences using model '{}' with max concurrency {}",
        evaluation_infos.len(),
        analysis_model,
        max_concurrency
    );

    // Create dynamic variant config for the analyze function using new template format
    let mut analyze_config = UninitializedChatCompletionConfig {
        model: analysis_model.clone().into(),
        weight: None,
        retries: gepa_config.retries,
        ..Default::default()
    };

    // Populate templates.inner with the analyze function's templates
    analyze_config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
                "gepa/analyze/system.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/system_template.minijinja")
                    .to_string(),
            ),
        },
    );
    analyze_config.templates.inner.insert(
        "user".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
                "gepa/analyze/user.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/user_template.minijinja")
                    .to_string(),
            ),
        },
    );

    let analyze_variant_config = Arc::new(UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(analyze_config),
        timeouts: None,
    });

    // Create semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(max_concurrency));

    // Create futures for parallel execution
    let analysis_futures: Vec<_> = evaluation_infos
        .iter()
        .enumerate()
        .map(|(index, eval_info)| {
            let semaphore = Arc::clone(&semaphore);
            let analyze_variant_config = Arc::clone(&analyze_variant_config);
            let gateway_client = gateway_client.clone();

            async move {
                // Acquire semaphore permit for concurrency control
                let _permit = semaphore.acquire().await.map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to acquire semaphore: {e}"),
                    })
                })?;

                // Build input for the analyze function (returns Arguments directly)
                let arguments = build_analyze_input(eval_info, config_and_tools, variant_config)?;

                // Create ClientInferenceParams for the analyze function
                let params = ClientInferenceParams {
                    function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
                    model_name: None,
                    episode_id: None,
                    input: ClientInput {
                        messages: vec![ClientInputMessage {
                            role: Role::User,
                            content: vec![ClientInputMessageContent::Template(Template {
                                name: "user".to_string(),
                                arguments,
                            })],
                        }],
                        system: None,
                    },
                    stream: None,
                    params: Default::default(),
                    variant_name: None,
                    dryrun: Some(true), // Required when using internal_dynamic_variant_config
                    internal: true,
                    tags: HashMap::new(),
                    dynamic_tool_params: Default::default(),
                    output_schema: None,
                    credentials: HashMap::new(),
                    cache_options: Default::default(),
                    include_original_response: false,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    internal_dynamic_variant_config: Some((*analyze_variant_config).clone()),
                    otlp_traces_extra_headers: HashMap::new(),
                };

                // Call the inference API
                let inference_output = gateway_client.inference(params).await.map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to call analyze function: {e}"),
                    })
                })?;

                // Extract the response
                let response = match inference_output {
                    InferenceOutput::NonStreaming(response) => response,
                    InferenceOutput::Streaming(_) => {
                        return Err(Error::new(ErrorDetails::Inference {
                            message: "Unexpected streaming response from analyze function"
                                .to_string(),
                        }))
                    }
                };

                // Extract content blocks from the response
                let analysis = match &response {
                    InferenceResponse::Chat(chat_response) => chat_response.content.clone(),
                    InferenceResponse::Json(_) => {
                        return Err(Error::new(ErrorDetails::Inference {
                            message: "Expected Chat response from analyze function, got Json"
                                .into(),
                        }))
                    }
                };

                // Log progress every 10 analyses
                if (index + 1) % 10 == 0 {
                    tracing::info!(
                        "Completed {}/{} analyses",
                        index + 1,
                        evaluation_infos.len()
                    );
                }

                // Conditionally include datapoint input based on config flag
                let datapoint_input = if gepa_config.include_datapoint_input_for_mutation {
                    Some(serialize_to_value(
                        eval_info.datapoint.input(),
                        "datapoint input",
                    )?)
                } else {
                    None
                };

                Ok(InferenceWithAnalysis {
                    inference_output: eval_info.response.clone(),
                    analysis,
                    datapoint_input,
                })
            }
        })
        .collect();

    // Execute all analyses in parallel (graceful degradation on failures)
    let results = join_all(analysis_futures).await;

    // Partition into successes and failures
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(analysis) => successes.push(analysis),
            Err(e) => {
                tracing::warn!(
                    "Analysis failed for inference {}/{}: {}",
                    index + 1,
                    evaluation_infos.len(),
                    e
                );
                failures.push(e);
            }
        }
    }

    // Check if all analyses failed (empty input already handled by early return)
    if successes.is_empty() {
        return Err(Error::new(ErrorDetails::Inference {
            message: format!(
                "All {} analyses failed. First error: {}",
                evaluation_infos.len(),
                failures
                    .first()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "Unknown error".to_string())
            ),
        }));
    }

    // Log summary
    if failures.is_empty() {
        tracing::info!(
            "Successfully completed all {} analyses",
            evaluation_infos.len()
        );
    } else {
        tracing::warn!(
            "Completed {}/{} analyses successfully ({} failed)",
            successes.len(),
            evaluation_infos.len(),
            failures.len()
        );
    }

    Ok(successes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evaluations::stats::EvaluationInfo;
    use serde_json::json;
    use std::collections::HashMap;
    use tensorzero_core::{
        config::{path::ResolvedTomlPath, SchemaData},
        endpoints::{
            datasets::{StoredChatInferenceDatapoint, StoredDatapoint},
            inference::{ChatInferenceResponse, InferenceResponse},
        },
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{ContentBlockChatOutput, Input, Text, Usage},
        jsonschema_util::{SchemaWithMetadata, StaticJSONSchema},
    };
    use uuid::Uuid;

    // ============================================================================
    // Test Helper Functions
    // ============================================================================

    /// Create a minimal Chat FunctionConfig for testing
    fn create_test_function_config() -> FunctionConfig {
        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        })
    }

    /// Create a Chat FunctionConfig with schemas
    fn create_test_function_config_with_schemas() -> FunctionConfig {
        let system_schema = StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            }
        }))
        .unwrap();

        let user_schema = StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }))
        .unwrap();

        let mut schema_inner = HashMap::new();
        schema_inner.insert(
            "system".to_string(),
            SchemaWithMetadata {
                schema: system_schema,
                legacy_definition: false,
            },
        );
        schema_inner.insert(
            "user".to_string(),
            SchemaWithMetadata {
                schema: user_schema,
                legacy_definition: false,
            },
        );

        let schemas = SchemaData {
            inner: schema_inner,
        };

        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas,
            tools: vec![],
            tool_choice: tensorzero_core::tool::ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("Test function with schemas".to_string()),
            all_explicit_templates_names: std::collections::HashSet::new(),
            experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
        })
    }

    /// Create a minimal FunctionConfigAndTools for testing (no tools)
    fn create_test_config_and_tools() -> FunctionConfigAndTools {
        FunctionConfigAndTools {
            function_config: Arc::new(create_test_function_config()),
            static_tools: None,
        }
    }

    /// Create a FunctionConfigAndTools with schemas for testing (no tools)
    fn create_test_config_and_tools_with_schemas() -> FunctionConfigAndTools {
        FunctionConfigAndTools {
            function_config: Arc::new(create_test_function_config_with_schemas()),
            static_tools: None,
        }
    }

    /// Create a test Chat InferenceResponse
    fn create_test_chat_inference_response(text: &str) -> InferenceResponse {
        InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: text.to_string(),
            })],
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        })
    }

    /// Create a test UninitializedChatCompletionConfig
    fn create_test_variant_config() -> UninitializedChatCompletionConfig {
        UninitializedChatCompletionConfig {
            model: "test-model".into(),
            weight: None,
            system_template: Some(ResolvedTomlPath::new_fake_path(
                "system.minijinja".to_string(),
                "You are a helpful assistant.".to_string(),
            )),
            user_template: Some(ResolvedTomlPath::new_fake_path(
                "user.minijinja".to_string(),
                "User: {{input}}".to_string(),
            )),
            assistant_template: None,
            ..Default::default()
        }
    }

    /// Create a test EvaluationInfo
    fn create_test_evaluation_info() -> EvaluationInfo {
        let input = Input {
            messages: vec![],
            system: None,
        };

        // Convert Input to StoredInput via JSON round-trip
        let stored_input = serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

        let datapoint = StoredDatapoint::Chat(StoredChatInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: stored_input,
            output: None,
            tool_params: None,
            tags: Some(HashMap::new()),
            auxiliary: String::new(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            name: None,
        });

        EvaluationInfo {
            datapoint,
            response: create_test_chat_inference_response("Test response"),
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
        }
    }

    // ============================================================================
    // Unit Tests for Helper Functions
    // ============================================================================

    #[test]
    fn test_serialize_to_value_success() {
        let data = HashMap::from([("key".to_string(), "value".to_string())]);
        let result = serialize_to_value(&data, "test data");

        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["key"], "value");
    }

    #[test]
    fn test_serialize_to_value_preserves_structure() {
        let data = json!({
            "nested": {
                "field": "value"
            },
            "array": [1, 2, 3]
        });

        let result = serialize_to_value(&data, "nested data");
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["nested"]["field"], "value");
        assert_eq!(value["array"][0], 1);
    }

    // ============================================================================
    // Unit Tests for build_analyze_input
    // ============================================================================

    #[test]
    fn test_build_analyze_input_basic() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check required fields
        assert!(input.0.get("function_name").is_some());
        assert!(input.0.get("model").is_some());
        assert!(input.0.get("templates").is_some());
        assert!(input.0.get("schemas").is_some());
        assert!(input.0.get("input").is_some());
        assert!(input.0.get("output").is_some());
    }

    #[test]
    fn test_build_analyze_input_with_schemas() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools_with_schemas();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check schemas map exists and has entries
        let schemas = input.0.get("schemas").unwrap();
        assert!(schemas.is_object());
        let schemas_obj = schemas.as_object().unwrap();
        assert!(schemas_obj.contains_key("system"));
        assert!(schemas_obj.contains_key("user"));
    }

    #[test]
    fn test_build_analyze_input_empty_schemas() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools(); // No schemas
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // schemas should be an empty object
        let schemas = input.0.get("schemas").unwrap();
        assert!(schemas.is_object());
        assert!(schemas.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_build_analyze_input_templates_extracted() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check templates field exists
        let templates = input.0.get("templates").unwrap();
        assert!(templates.is_object());
        // Note: The test variant config uses legacy templates (system_template, user_template)
        // which are converted to the new templates.inner format during variant initialization.
        // For unit tests, we're just verifying that templates extraction works,
        // not the specific keys (which depend on the variant config implementation details).
    }

    #[test]
    fn test_build_analyze_input_model_name() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        let model = input.0.get("model").unwrap();
        assert_eq!(model.as_str().unwrap(), "test-model");
    }

    #[test]
    fn test_build_analyze_input_function_name() {
        let eval_info = create_test_evaluation_info();
        let config_and_tools = create_test_config_and_tools();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &config_and_tools, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        let function_name = input.0.get("function_name").unwrap();
        assert_eq!(function_name.as_str().unwrap(), "test_function");
    }
}
