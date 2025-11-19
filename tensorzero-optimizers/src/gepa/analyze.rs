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
    config::{path::ResolvedTomlPathData, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    inference::types::{Arguments, ContentBlockChatOutput, Role, StoredInput, Template},
    optimization::gepa::GEPAConfig,
    tool::StaticToolConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use evaluations::stats::EvaluationInfo;

/// Represents an inference input and output pair
#[derive(Debug, Clone, Serialize)]
pub struct Inference {
    pub input: StoredInput,
    pub output: serde_json::Value,
}

/// Represents an analysis result with optional inference context
#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    /// Optional inference context (only included if include_inference_for_mutation is true)
    /// Flattened during serialization so input/output appear at top level
    /// Skipped during serialization if None to avoid bloating the mutate function input
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub inference: Option<Inference>,
    pub analysis: Vec<ContentBlockChatOutput>,
}

/// Build the input JSON for the analyze function
///
/// Passes high-level objects to the template for serialization rather than extracting individual fields.
///
/// # Arguments
/// * `eval_info` - Evaluation information containing the datapoint and inference response
/// * `function_config` - Function configuration (serialized as function_context in template)
/// * `static_tools` - Static tools from Config.tools (serialized as tool_schemas in template)
/// * `variant_config` - Variant configuration used to extract templates
/// * `evaluation_config` - Evaluation config (serialized as evaluation_context in template)
///
/// # Returns
/// * Template arguments containing: function_config, static_tools, evaluation_config,
///   templates_map, datapoint, output, and evaluation_scores
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    function_config: &FunctionConfig,
    static_tools: &Option<HashMap<String, Arc<StaticToolConfig>>>,
    variant_config: &UninitializedChatCompletionConfig,
    evaluation_config: &EvaluationConfig,
) -> Result<Arguments, Error> {
    // Extract templates map from variant config
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| (name.clone(), config.path.data().to_string()))
        .collect();

    let output = match &eval_info.response {
        InferenceResponse::Chat(chat_response) => {
            serialize_to_value(&chat_response.content, "chat response content")?
        }
        InferenceResponse::Json(json_response) => {
            serialize_to_value(&json_response.output, "json response output")?
        }
    };

    // Build evaluation_scores map with just the scores
    let mut evaluation_scores = serde_json::Map::new();
    for (evaluator_name, result_opt) in &eval_info.evaluations {
        // Preserve the score type (number, boolean, or null)
        let score = result_opt
            .as_ref()
            .map(|value| match value {
                serde_json::Value::Number(n) => json!(n),
                serde_json::Value::Bool(b) => json!(b),
                _ => json!(null),
            })
            .unwrap_or(json!(null));
        evaluation_scores.insert(evaluator_name.clone(), score);
    }

    // Build the input with high-level objects that will be serialized in the template
    let mut map = serde_json::Map::new();
    map.insert(
        "function_config".to_string(),
        serialize_to_value(function_config, "function_config")?,
    );
    map.insert("static_tools".to_string(), json!(static_tools));
    map.insert(
        "evaluation_config".to_string(),
        serialize_to_value(evaluation_config, "evaluation_config")?,
    );
    map.insert("templates_map".to_string(), json!(templates_map));
    map.insert(
        "datapoint".to_string(),
        serialize_to_value(&eval_info.datapoint, "datapoint")?,
    );
    map.insert("output".to_string(), json!(output));
    map.insert("evaluation_scores".to_string(), json!(evaluation_scores));

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
/// * `static_tools` - Static tools from Config.tools referenced by the function
/// * `variant_config` - Configuration of the variant being analyzed
/// * `gepa_config` - GEPA configuration containing analysis_model and max_concurrency settings
/// * `evaluation_config` - Evaluation config enriched with loaded system templates
///
/// # Returns
/// * Vector of [`Analysis`] containing each inference paired with its XML analysis feedback
pub async fn analyze_inferences(
    gateway_client: &Client,
    evaluation_infos: &[EvaluationInfo],
    function_config: &FunctionConfig,
    static_tools: &Option<HashMap<String, Arc<StaticToolConfig>>>,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
    evaluation_config: &EvaluationConfig,
) -> Result<Vec<Analysis>, Error> {
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

    let mut analyze_config = UninitializedChatCompletionConfig {
        model: analysis_model.clone().into(),
        weight: None,
        retries: gepa_config.retries,
        max_tokens: gepa_config.max_tokens,
        ..Default::default()
    };

    analyze_config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPathData::new_fake_path(
                "gepa/analyze/system.minijinja".to_string(),
                include_str!("functions/analyze/system_template.minijinja").to_string(),
            ),
        },
    );
    analyze_config.templates.inner.insert(
        "user".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPathData::new_fake_path(
                "gepa/analyze/user.minijinja".to_string(),
                include_str!("functions/analyze/user_template.minijinja").to_string(),
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

                let arguments = build_analyze_input(
                    eval_info,
                    function_config,
                    static_tools,
                    variant_config,
                    evaluation_config,
                )?;

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
                let InferenceOutput::NonStreaming(response) = inference_output else {
                    return Err(Error::new(ErrorDetails::Inference {
                        message: "Expected NonStreaming response but got Streaming".to_string(),
                    }))
                };

                // Extract text content from the response
                let InferenceResponse::Chat(chat_response) = &response else {
                    return Err(Error::new(ErrorDetails::Inference {
                        message: "analyze function is defined as Chat, cannot return JSON".to_string(),
                    }))
                };

                // Warn if response has more than 1 content block
                if chat_response.content.len() > 1 {
                    tracing::warn!(
                        "Analyze function returned {} content blocks, expected 1. Using first Text block.",
                        chat_response.content.len()
                    );
                }

                // Find the first Text content block
                let text_block = chat_response.content.iter().find_map(|block| {
                    if let ContentBlockChatOutput::Text(text) = block {
                        Some(text.clone())
                    } else {
                        None
                    }
                });

                let analysis = match text_block {
                    Some(text) => vec![ContentBlockChatOutput::Text(text)],
                    None => {
                        return Err(Error::new(ErrorDetails::Inference {
                            message: "Expected at least one Text content block from analyze function, found none"
                                .to_string(),
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

                // Conditionally include inference context based on config flag
                let inference = if gepa_config.include_inference_for_mutation {
                    let output = match &eval_info.response {
                        InferenceResponse::Chat(chat_response) => {
                            serialize_to_value(&chat_response.content, "chat response content")?
                        }
                        InferenceResponse::Json(json_response) => {
                            serialize_to_value(&json_response.output, "json response output")?
                        }
                    };
                    Some(Inference {
                        input: eval_info.datapoint.input().clone(),
                        output,
                    })
                } else {
                    None
                };

                Ok(Analysis {
                    inference,
                    analysis,
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

/// Helper function to serialize a value to JSON with consistent error handling
pub fn serialize_to_value<T: serde::Serialize>(
    value: &T,
    context: &str,
) -> Result<serde_json::Value, Error> {
    serde_json::to_value(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {context}: {e}"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evaluations::stats::EvaluationInfo;
    use serde_json::json;
    use std::collections::HashMap;
    use tensorzero_core::{
        config::{path::ResolvedTomlPathData, SchemaData},
        endpoints::{
            datasets::{StoredChatInferenceDatapoint, StoredDatapoint},
            inference::{ChatInferenceResponse, InferenceResponse},
        },
        evaluations::{EvaluationConfig, InferenceEvaluationConfig},
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{ContentBlockChatOutput, Input, Text, Usage},
        jsonschema_util::{SchemaWithMetadata, StaticJSONSchema},
        tool::StaticToolConfig,
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
            system_template: Some(ResolvedTomlPathData::new_fake_path(
                "system.minijinja".to_string(),
                "You are a helpful assistant.".to_string(),
            )),
            user_template: Some(ResolvedTomlPathData::new_fake_path(
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

    /// Create a test EvaluationConfig with empty evaluators
    fn create_test_evaluation_config() -> EvaluationConfig {
        EvaluationConfig::Inference(InferenceEvaluationConfig {
            evaluators: HashMap::new(),
            function_name: "test_function".to_string(),
        })
    }

    // ============================================================================
    // Unit Tests for build_analyze_input
    // ============================================================================

    #[test]
    fn test_build_analyze_input_comprehensive() {
        // Test with minimal config (no schemas)
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let static_tools = None;
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let result = build_analyze_input(
            &eval_info,
            &function_config,
            &static_tools,
            &variant_config,
            &eval_config,
        );

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check all 7 required high-level fields are present
        assert!(input.0.get("function_config").is_some());
        assert!(input.0.get("static_tools").is_some());
        assert!(input.0.get("evaluation_config").is_some());
        assert!(input.0.get("templates_map").is_some());
        assert!(input.0.get("datapoint").is_some());
        assert!(input.0.get("output").is_some());
        assert!(input.0.get("evaluation_scores").is_some());

        // Verify function_config is properly serialized and preserves structure
        let func_config = input.0.get("function_config").unwrap();
        assert!(func_config.is_object());
        let func_config_obj = func_config.as_object().unwrap();
        assert!(func_config_obj.contains_key("description"));
        assert_eq!(
            func_config_obj.get("description").unwrap().as_str(),
            Some("Test function")
        );

        // Verify evaluation_config is properly serialized and preserves structure
        let eval_config_value = input.0.get("evaluation_config").unwrap();
        assert!(eval_config_value.is_object());
        let eval_config_obj = eval_config_value.as_object().unwrap();
        assert!(eval_config_obj.contains_key("function_name"));
        assert_eq!(
            eval_config_obj.get("function_name").unwrap().as_str(),
            Some("test_function")
        );

        // Verify datapoint is properly serialized and preserves structure
        let datapoint = input.0.get("datapoint").unwrap();
        assert!(datapoint.is_object());
        let datapoint_obj = datapoint.as_object().unwrap();
        assert!(datapoint_obj.contains_key("function_name"));
        assert_eq!(
            datapoint_obj.get("function_name").unwrap().as_str(),
            Some("test_function")
        );

        let output = input.0.get("output").unwrap();
        assert!(
            output.is_array(),
            "Chat response content should be serialized as an array"
        );
        let output_arr = output.as_array().unwrap();
        assert!(!output_arr.is_empty(), "Content array should not be empty");

        // Test with schemas
        let function_config_with_schemas = create_test_function_config_with_schemas();
        let result_with_schemas = build_analyze_input(
            &eval_info,
            &function_config_with_schemas,
            &static_tools,
            &variant_config,
            &eval_config,
        );

        assert!(result_with_schemas.is_ok());
        let input_with_schemas = result_with_schemas.unwrap();

        // Verify schemas are included in serialized function_config
        let func_config_with_schemas = input_with_schemas.0.get("function_config").unwrap();
        let func_config_obj = func_config_with_schemas.as_object().unwrap();
        assert!(func_config_obj.contains_key("schemas"));
    }

    #[test]
    fn test_build_analyze_input_evaluation_scores_types() {
        // Create evaluation info with different score types
        let mut eval_info = create_test_evaluation_info();

        // Add numeric, boolean, and null scores
        eval_info
            .evaluations
            .insert("numeric_score".to_string(), Some(json!(0.85)));
        eval_info
            .evaluations
            .insert("boolean_score".to_string(), Some(json!(true)));
        eval_info.evaluations.insert("null_score".to_string(), None);

        let result = build_analyze_input(
            &eval_info,
            &create_test_function_config(),
            &None,
            &create_test_variant_config(),
            &create_test_evaluation_config(),
        );

        assert!(result.is_ok());
        let input = result.unwrap();

        // Get evaluation_scores object
        let eval_scores = input.0.get("evaluation_scores").unwrap();
        let eval_scores_obj = eval_scores.as_object().unwrap();

        // Verify numeric score is preserved (note: f32 conversion causes precision loss)
        let numeric = eval_scores_obj.get("numeric_score").unwrap();
        assert!(numeric.is_number());
        // Use approximate comparison due to f32 precision
        let numeric_val = numeric.as_f64().unwrap();
        assert!(
            (numeric_val - 0.85).abs() < 0.001,
            "Expected ~0.85, got {numeric_val}"
        );

        // Verify boolean score is preserved
        let boolean = eval_scores_obj.get("boolean_score").unwrap();
        assert!(boolean.is_boolean());
        assert!(boolean.as_bool().unwrap());

        // Verify null score is preserved
        let null = eval_scores_obj.get("null_score").unwrap();
        assert!(null.is_null());
    }

    #[test]
    fn test_build_analyze_input_templates_content() {
        let eval_info = create_test_evaluation_info();

        // Create a variant config with templates using the new format (templates.inner)
        let mut variant_config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            weight: None,
            ..Default::default()
        };

        // Directly populate templates.inner
        variant_config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "system.minijinja".to_string(),
                    "You are a helpful assistant.".to_string(),
                ),
            },
        );
        variant_config.templates.inner.insert(
            "user".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "user.minijinja".to_string(),
                    "User: {{input}}".to_string(),
                ),
            },
        );

        let result = build_analyze_input(
            &eval_info,
            &create_test_function_config(),
            &None,
            &variant_config,
            &create_test_evaluation_config(),
        );

        assert!(result.is_ok());
        let input = result.unwrap();

        // Get templates_map
        let templates_map = input.0.get("templates_map").unwrap();
        let templates_obj = templates_map.as_object().unwrap();

        // Verify template content is extracted (not just empty map)
        assert!(
            !templates_obj.is_empty(),
            "templates_map should not be empty"
        );

        // Verify both templates are present
        assert_eq!(templates_obj.len(), 2, "Should have 2 templates");

        // Verify system template content
        let system_template = templates_obj.get("system").unwrap();
        assert_eq!(
            system_template.as_str().unwrap(),
            "You are a helpful assistant."
        );

        // Verify user template content
        let user_template = templates_obj.get("user").unwrap();
        assert_eq!(user_template.as_str().unwrap(), "User: {{input}}");
    }

    #[test]
    fn test_build_analyze_input_static_tools() {
        // Create a static tool config
        let tool_config = Arc::new(StaticToolConfig {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: StaticJSONSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }))
            .unwrap(),
            strict: false,
        });

        let mut static_tools = HashMap::new();
        static_tools.insert("test_tool".to_string(), tool_config);

        let result = build_analyze_input(
            &create_test_evaluation_info(),
            &create_test_function_config(),
            &Some(static_tools),
            &create_test_variant_config(),
            &create_test_evaluation_config(),
        );

        assert!(result.is_ok());
        let input = result.unwrap();

        // Verify static_tools is serialized
        let tools = input.0.get("static_tools").unwrap();
        assert!(!tools.is_null());

        // Verify the tool is present in the serialized data
        let tools_obj = tools.as_object().unwrap();
        assert!(tools_obj.contains_key("test_tool"));

        // Verify tool has description
        let test_tool = tools_obj.get("test_tool").unwrap();
        let tool_obj = test_tool.as_object().unwrap();
        assert!(tool_obj.contains_key("description"));
        assert_eq!(
            tool_obj.get("description").unwrap().as_str().unwrap(),
            "Test tool"
        );
    }
}
