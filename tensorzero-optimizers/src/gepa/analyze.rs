//! Inference analysis functions for GEPA optimization
//!
//! This module provides functions for analyzing LLM inference outputs to identify
//! errors, improvements, and optimal patterns. It interfaces with the built-in
//! `tensorzero::optimization::gepa::analyze` function to generate structured feedback
//! in XML format.
//!
//! # Key Functions
//!
//! - [`build_analyze_input`] - Constructs input JSON for the analyze function
//! - [`parse_analysis_response`] - Extracts raw XML feedback from analyze responses
//! - [`analyze_inferences`] - Main orchestration function for batch analysis
//!
//! # Analysis Output Formats
//!
//! The analyze function returns XML feedback in one of three mutually exclusive formats:
//!
//! - `<report_error>` - Critical failures or errors in the inference output
//! - `<report_improvement>` - Suboptimal but technically correct outputs
//! - `<report_optimal>` - High-quality aspects worth preserving
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorzero_optimizers::gepa::analyze::{analyze_inferences, build_analyze_input};
//!
//! // Analyze a batch of inference results
//! let analyses = analyze_inferences(
//!     &gateway_client,
//!     &evaluation_infos,
//!     &function_config,
//!     &variant_config,
//!     &gepa_config,
//! ).await?;
//!
//! // Each analysis contains the original inference paired with XML feedback
//! for analysis in &analyses {
//!     println!("Inference: {:?}", analysis.inference_output);
//!     println!("Analysis: {}", analysis.analysis);
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::try_join_all;
use serde::Serialize;
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
    inference::types::{ContentBlockChatOutput, Role, TextKind},
    optimization::gepa::GEPAConfig,
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::stats::EvaluationInfo;

/// Helper function to serialize values to JSON with consistent error handling
///
/// Wraps `serde_json::to_value` with contextual error messages for better debugging.
///
/// # Arguments
///
/// * `value` - The value to serialize
/// * `context` - Description of what's being serialized (e.g., "inference response", "templates map")
///
/// # Errors
///
/// Returns a `tensorzero_core::error::Error` with `ErrorDetails::Serialization` if
/// serialization fails.
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

/// Helper function to serialize values to JSON strings with consistent error handling
///
/// Wraps `serde_json::to_string` with contextual error messages for better debugging.
///
/// # Arguments
///
/// * `value` - The value to serialize
/// * `context` - Description of what's being serialized (e.g., "analyze input", "datapoint")
///
/// # Errors
///
/// Returns a `tensorzero_core::error::Error` with `ErrorDetails::Serialization` if
/// serialization fails.
fn serialize_to_string<T: serde::Serialize>(value: &T, context: &str) -> Result<String, Error> {
    serde_json::to_string(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {context}: {e}"),
        })
    })
}

/// Represents an inference output paired with its analysis feedback
#[derive(Debug, Clone, Serialize)]
pub struct InferenceWithAnalysis {
    pub inference_output: InferenceResponse,
    /// Raw XML text output from the analyze function
    pub analysis: String,
}

/// Build the input JSON for the analyze function
///
/// Constructs a structured JSON object containing all context needed for analyzing
/// an inference output, including function metadata, templates, schemas, and the
/// actual input/output pair.
///
/// # Arguments
///
/// * `eval_info` - Evaluation information containing the datapoint and inference response
/// * `function_config` - Function configuration (schemas, tools, output_schema)
/// * `variant_config` - Variant configuration (templates, model name)
///
/// # Returns
///
/// A JSON object with the following structure:
/// ```json
/// {
///   "function_name": "...",
///   "model": "...",
///   "templates": {"system": "...", "user": "...", ...},
///   "schemas": {"system": {...}, "user": {...}, ...},
///   "output_schema": {...},  // Optional, for JSON functions
///   "tools": {...},          // Optional, for Chat functions with tools
///   "tags": {...},           // Optional metadata
///   "input": {...},          // The inference input
///   "output": {...}          // The inference output to analyze
/// }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Template reading fails
/// - Serialization of any component fails
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    function_config: &FunctionConfig,
    variant_config: &UninitializedChatCompletionConfig,
) -> Result<serde_json::Value, Error> {
    // Extract all templates from templates.inner HashMap using idiomatic iterators
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| config.path.read().map(|content| (name.clone(), content)))
        .collect::<Result<_, _>>()?;

    // Extract all schemas from function config using the new schemas.inner pattern
    let schemas_map: HashMap<String, serde_json::Value> = function_config
        .schemas()
        .inner
        .iter()
        .map(|(name, schema_with_metadata)| {
            (name.clone(), schema_with_metadata.schema.value.clone())
        })
        .collect();

    // Extract output schema for JSON functions
    let output_schema = match function_config {
        FunctionConfig::Json(params) => Some(params.output_schema.value.clone()),
        FunctionConfig::Chat(_) => None,
    };

    // Extract tools for Chat functions
    let tools = match function_config {
        FunctionConfig::Chat(params) => {
            if params.tools.is_empty() {
                None
            } else {
                Some(serde_json::json!(params.tools))
            }
        }
        FunctionConfig::Json(_) => None,
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

    // Build the input object
    let input = serde_json::json!({
        "function_name": function_name,
        "model": variant_config.model.as_ref(),
        "templates": templates_map,
        "schemas": schemas_map,
        "output_schema": output_schema,
        "tools": tools,
        "tags": tags,
        "input": eval_info.datapoint.input(),
        "output": output,
    });

    Ok(input)
}

/// Parse the analysis response and extract the raw XML text
///
/// Extracts the raw XML text content from the analyze function response.
/// The analyze function returns Chat responses containing XML feedback in one
/// of three formats: `<report_error>`, `<report_improvement>`, or `<report_optimal>`.
///
/// # Arguments
///
/// * `response` - The inference response from the analyze function call
///
/// # Returns
///
/// The raw XML text content as a string. This will contain one of:
/// - `<report_error>...</report_error>`
/// - `<report_improvement>...</report_improvement>`
/// - `<report_optimal>...</report_optimal>`
///
/// # Errors
///
/// Returns an error if:
/// - The response is not a Chat response (expects Chat, not Json)
/// - The response contains no text content blocks
pub fn parse_analysis_response(response: &InferenceResponse) -> Result<String, Error> {
    let InferenceResponse::Chat(chat_response) = response else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "Expected Chat response from analyze function, got Json".into(),
        }));
    };

    // Extract text content using idiomatic iterator approach
    chat_response
        .content
        .iter()
        .find_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text.text.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: "No text content found in analyze response".into(),
            })
        })
}

/// Analyze inference outputs using the GEPA analyze function
///
/// Takes evaluation results (datapoint + inference pairs) and calls the built-in
/// `tensorzero::optimization::gepa::analyze` function to get structured feedback.
///
/// The analyses are executed in parallel with controlled concurrency (up to `gepa_config.max_concurrency`
/// concurrent API calls). Uses a semaphore for concurrency control and fail-fast error handling
/// with `try_join_all`. Progress is logged every 10 completed analyses.
///
/// # Arguments
///
/// * `gateway_client` - TensorZero gateway client for making inference requests
/// * `evaluation_infos` - Evaluation results containing datapoints and their inference responses
/// * `function_config` - Configuration of the function being optimized (for schemas, tools, etc.)
/// * `variant_config` - Configuration of the variant being analyzed (for templates, model, etc.)
/// * `gepa_config` - GEPA configuration containing analysis_model and max_concurrency settings
///
/// # Returns
///
/// A vector of [`InferenceWithAnalysis`] containing each inference paired with its XML analysis feedback.
///
/// # Errors
///
/// Returns an error if:
/// - Building the analyze input fails (e.g., template read errors, serialization failures)
/// - The analyze function call fails (e.g., network errors, model errors)
/// - Parsing the analysis response fails (e.g., unexpected response format, missing text content)
/// - Acquiring a semaphore permit fails
///
/// Note: This function is currently unused but will be called by the main GEPA optimization loop.
#[expect(dead_code)]
pub async fn analyze_inferences(
    gateway_client: &Client,
    evaluation_infos: &[EvaluationInfo],
    function_config: &FunctionConfig,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
) -> Result<Vec<InferenceWithAnalysis>, Error> {
    // Extract parameters from GEPA config
    let analysis_model = &gepa_config.analysis_model;
    let max_concurrency = gepa_config.max_concurrency as usize;

    tracing::info!(
        "Analyzing {} inferences using model '{}' with max concurrency {}",
        evaluation_infos.len(),
        analysis_model,
        max_concurrency
    );

    // Create dynamic variant config for the analyze function (wrapped in Arc for efficient sharing)
    let analyze_variant_config = Arc::new(UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
            model: analysis_model.clone().into(),
            weight: None,
            system_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/analyze/system.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/system_template.minijinja")
                    .to_string(),
            )),
            user_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/analyze/user.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/user_template.minijinja")
                    .to_string(),
            )),
            assistant_template: None,
            ..Default::default()
        }),
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

                // Build input JSON for the analyze function
                let input_data = build_analyze_input(eval_info, function_config, variant_config)?;

                // Create ClientInferenceParams for the analyze function
                let params = ClientInferenceParams {
                    function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
                    model_name: None,
                    episode_id: None,
                    input: ClientInput {
                        messages: vec![ClientInputMessage {
                            role: Role::User,
                            content: vec![ClientInputMessageContent::Text(TextKind::Text {
                                text: serialize_to_string(&input_data, "analyze input")?,
                            })],
                        }],
                        system: None,
                    },
                    stream: None,
                    params: Default::default(),
                    variant_name: None,
                    dryrun: None,
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

                // Parse the analysis from the response
                let analysis = parse_analysis_response(&response)?;

                // Log progress every 10 analyses
                if (index + 1) % 10 == 0 {
                    tracing::info!(
                        "Completed {}/{} analyses",
                        index + 1,
                        evaluation_infos.len()
                    );
                }

                Ok(InferenceWithAnalysis {
                    inference_output: eval_info.response.clone(),
                    analysis,
                })
            }
        })
        .collect();

    // Execute all analyses in parallel with fail-fast error handling
    let results = try_join_all(analysis_futures).await?;

    tracing::info!(
        "Successfully completed all {} analyses",
        evaluation_infos.len()
    );

    Ok(results)
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
            inference::{ChatInferenceResponse, InferenceResponse, JsonInferenceResponse},
        },
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{ContentBlockChatOutput, Input, JsonInferenceOutput, Text, Usage},
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

    /// Create a test Chat InferenceResponse with no text content
    fn create_test_chat_inference_response_empty() -> InferenceResponse {
        InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        })
    }

    /// Create a test Json InferenceResponse
    fn create_test_json_inference_response() -> InferenceResponse {
        InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            output: JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(json!({"result": "test"})),
            },
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

    #[test]
    fn test_serialize_to_string_success() {
        let data = HashMap::from([("key".to_string(), "value".to_string())]);
        let result = serialize_to_string(&data, "test data");

        assert!(result.is_ok());
        let string = result.unwrap();
        assert!(string.contains("key"));
        assert!(string.contains("value"));
    }

    #[test]
    fn test_serialize_to_string_format() {
        let data = json!({"test": "value"});
        let result = serialize_to_string(&data, "json data");

        assert!(result.is_ok());
        let string = result.unwrap();
        // Should be valid JSON
        assert!(serde_json::from_str::<serde_json::Value>(&string).is_ok());
    }

    // ============================================================================
    // Unit Tests for parse_analysis_response
    // ============================================================================

    #[test]
    fn test_parse_analysis_response_success() {
        let response = create_test_chat_inference_response(
            "Analysis: <report_error>Test error</report_error>",
        );
        let result = parse_analysis_response(&response);

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.contains("Test error"));
        assert!(analysis.contains("report_error"));
    }

    #[test]
    fn test_parse_analysis_response_multiple_blocks() {
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![
                ContentBlockChatOutput::Text(Text {
                    text: "First text block".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Second text block".to_string(),
                }),
            ],
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        });

        let result = parse_analysis_response(&response);
        assert!(result.is_ok());
        // Should return the first text block
        assert_eq!(result.unwrap(), "First text block");
    }

    #[test]
    fn test_parse_analysis_response_json_error() {
        let response = create_test_json_inference_response();
        let result = parse_analysis_response(&response);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Expected Chat response"));
    }

    #[test]
    fn test_parse_analysis_response_no_text_content() {
        let response = create_test_chat_inference_response_empty();
        let result = parse_analysis_response(&response);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("No text content found"));
    }

    #[test]
    fn test_parse_analysis_response_empty_content() {
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Usage::default(),
            original_response: None,
            finish_reason: None,
        });

        let result = parse_analysis_response(&response);
        assert!(result.is_err());
    }

    // ============================================================================
    // Unit Tests for build_analyze_input
    // ============================================================================

    #[test]
    fn test_build_analyze_input_basic() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check required fields
        assert!(input.get("function_name").is_some());
        assert!(input.get("model").is_some());
        assert!(input.get("templates").is_some());
        assert!(input.get("schemas").is_some());
        assert!(input.get("input").is_some());
        assert!(input.get("output").is_some());
    }

    #[test]
    fn test_build_analyze_input_with_schemas() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config_with_schemas();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check schemas map exists and has entries
        let schemas = input.get("schemas").unwrap();
        assert!(schemas.is_object());
        let schemas_obj = schemas.as_object().unwrap();
        assert!(schemas_obj.contains_key("system"));
        assert!(schemas_obj.contains_key("user"));
    }

    #[test]
    fn test_build_analyze_input_empty_schemas() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config(); // No schemas
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // schemas should be an empty object
        let schemas = input.get("schemas").unwrap();
        assert!(schemas.is_object());
        assert!(schemas.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_build_analyze_input_templates_extracted() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check templates field exists
        let templates = input.get("templates").unwrap();
        assert!(templates.is_object());
        // Note: The test variant config uses legacy templates (system_template, user_template)
        // which are converted to the new templates.inner format during variant initialization.
        // For unit tests, we're just verifying that templates extraction works,
        // not the specific keys (which depend on the variant config implementation details).
    }

    #[test]
    fn test_build_analyze_input_model_name() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        let model = input.get("model").unwrap();
        assert_eq!(model.as_str().unwrap(), "test-model");
    }

    #[test]
    fn test_build_analyze_input_function_name() {
        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();

        let result = build_analyze_input(&eval_info, &function_config, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        let function_name = input.get("function_name").unwrap();
        assert_eq!(function_name.as_str().unwrap(), "test_function");
    }
}
