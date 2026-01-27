//! Inference analysis for GEPA optimization.
//!
//! Analyzes inference outputs to identify errors, improvements, and optimal patterns.
//! Builds inputs for the analyze function and handles results with optional inference context.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde::Serialize;
use serde_json::{Map, Value, json, to_value};
use tokio::sync::Semaphore;

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent,
    },
    config::{UninitializedVariantConfig, UninitializedVariantInfo, path::ResolvedTomlPathData},
    endpoints::{
        datasets::{ChatInferenceDatapoint, Datapoint},
        inference::InferenceResponse,
    },
    error::{Error, ErrorDetails},
    inference::types::{
        Arguments, ContentBlockChatOutput, InputExt, Role, StoredInput, StoredInputMessageContent,
        Template,
    },
    optimization::gepa::GEPAConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};

use evaluations::stats::EvaluationInfo;

use crate::gepa::validate::FunctionContext;

/// Fields to exclude when serializing datapoints for GEPA functions.
/// These metadata fields are not relevant for prompt optimization and waste tokens.
const DATAPOINT_FIELDS_TO_DROP: &[&str] = &[
    "dataset_name",
    "id",
    "episode_id",
    "auxiliary",
    "is_deleted",
    "is_custom",
    "source_inference_id",
    "staled_at",
    "updated_at",
    "name",
];

/// Serialize a datapoint for GEPA functions, excluding superfluous metadata fields.
///
/// This reduces token usage by removing fields like IDs, timestamps, and flags
/// that are not relevant for prompt optimization analysis.
fn serialize_filtered_datapoint(datapoint: &Datapoint) -> Result<Value, Error> {
    let mut value = to_value(datapoint)?;

    if let Some(obj) = value.as_object_mut() {
        for field in DATAPOINT_FIELDS_TO_DROP {
            obj.remove(*field);
        }
    }

    Ok(value)
}

// ============================================================================
// Type-safe thought signature stripping
// ============================================================================
//
// These functions remove the `signature` field from Thought content blocks at
// the Rust type level, before serialization. This is more type-safe than JSON
// manipulation and ensures compile-time errors if type structures change.
//
// Thought signatures are Anthropic-specific encrypted blobs that have no semantic
// value and waste tokens when passed to GEPA analysis/mutation models.

/// Strip signatures from input message content blocks.
fn strip_signatures_from_input(input: &mut Input) {
    for message in &mut input.messages {
        for content in &mut message.content {
            if let InputMessageContent::Thought(thought) = content {
                thought.signature = None;
            }
        }
    }
}

/// Strip signatures from stored input message content blocks.
fn strip_signatures_from_stored_input(input: &mut StoredInput) {
    for message in &mut input.messages {
        for content in &mut message.content {
            if let StoredInputMessageContent::Thought(thought) = content {
                thought.signature = None;
            }
        }
    }
}

/// Strip signatures from chat output content blocks.
fn strip_signatures_from_chat_output(blocks: &mut [ContentBlockChatOutput]) {
    for block in blocks {
        if let ContentBlockChatOutput::Thought(thought) = block {
            thought.signature = None;
        }
    }
}

/// Strip signatures from a Chat datapoint's input and output.
fn strip_signatures_from_chat_datapoint(datapoint: &mut ChatInferenceDatapoint) {
    strip_signatures_from_input(&mut datapoint.input);
    if let Some(ref mut output) = datapoint.output {
        strip_signatures_from_chat_output(output);
    }
}

/// Inference input/output pair for GEPA mutation phase.
///
/// Conditionally included in Analysis via include_inference_for_mutation config flag.
#[derive(Debug, Clone, Serialize)]
pub struct Inference {
    /// Inference input (messages, system prompt, etc.)
    pub input: StoredInput,
    /// Inference output as JSON (chat content blocks or JSON output)
    pub output: Value,
}

/// Analysis result with optional inference context.
///
/// Contains analysis feedback from the analyze function and optional inference context for mutation.
#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    /// Optional inference context (included if include_inference_for_mutation is true).
    /// Flattened during serialization so input/output appear at top level.
    /// Skipped if None to avoid bloating mutate function input.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub inference: Option<Inference>,
    /// Analysis feedback text from the analyze function.
    /// Typically XML-formatted reports (error, improvement, or optimal).
    pub analysis: String,
}

/// Creates variant configuration for the analyze function.
///
/// Builds uninitialized chat completion config with embedded templates using GEPAConfig settings.
///
/// Returns configured UninitializedChatCompletionConfig with system and user templates.
fn create_analyze_variant_config(gepa_config: &GEPAConfig) -> UninitializedChatCompletionConfig {
    let mut analyze_config = UninitializedChatCompletionConfig {
        model: gepa_config.analysis_model.clone().into(),
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

    analyze_config
}

/// Builds input JSON for the analyze function.
///
/// Passes high-level objects to the template for serialization.
///
/// Returns Arguments with template variables: function_config, static_tools, evaluation_config,
/// templates_map, datapoint, output, and evaluation_scores.
///
/// Returns error if serialization fails.
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
) -> Result<Arguments, Error> {
    // Extract fields from function context
    let FunctionContext {
        function_config,
        static_tools,
        evaluation_config,
    } = function_context;

    // Extract templates map from variant config
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| (name.clone(), config.path.data().to_string()))
        .collect();

    // Build evaluation_scores map with just the scores
    let mut evaluation_scores = Map::new();
    for (evaluator_name, result_opt) in &eval_info.evaluations {
        // Preserve the score type (number, boolean, or null)
        let score = result_opt
            .as_ref()
            .map(|value| match value {
                Value::Number(n) => json!(n),
                Value::Bool(b) => json!(b),
                _ => json!(null),
            })
            .unwrap_or(json!(null));
        evaluation_scores.insert(evaluator_name.clone(), score);
    }

    // Build the input with high-level objects that will be serialized in the template
    let mut map = Map::new();
    map.insert("function_config".to_string(), to_value(function_config)?);
    if let Some(tools) = static_tools {
        map.insert("static_tools".to_string(), json!(tools));
    }
    map.insert(
        "evaluation_config".to_string(),
        to_value(evaluation_config)?,
    );
    map.insert("templates_map".to_string(), json!(templates_map));

    // Strip thought signatures at the type level before serialization.
    // Only Chat types can contain Thought blocks; Json outputs are left intact.
    let mut datapoint_for_serialization = eval_info.datapoint.clone();
    match &mut datapoint_for_serialization {
        Datapoint::Chat(chat_datapoint) => {
            strip_signatures_from_chat_datapoint(chat_datapoint);
        }
        Datapoint::Json(_) => {} // No signatures to strip in JSON
    }

    // Serialize with filtered fields
    let datapoint_value = serialize_filtered_datapoint(&datapoint_for_serialization)?;
    map.insert("datapoint".to_string(), datapoint_value);

    let output_value = match &eval_info.response {
        InferenceResponse::Chat(chat_response) => {
            let mut content = chat_response.content.clone();
            strip_signatures_from_chat_output(&mut content);
            to_value(&content)?
        }
        InferenceResponse::Json(json_response) => to_value(&json_response.output)?,
    };
    map.insert("output".to_string(), output_value);

    map.insert("evaluation_scores".to_string(), json!(evaluation_scores));

    Ok(Arguments(map))
}

/// Analyzes a single inference using the analyze function.
///
/// Returns Analysis with feedback and optional inference context.
///
/// Returns error if semaphore acquisition, input building, API call, or response parsing fails.
async fn analyze_inference(
    semaphore: Arc<Semaphore>,
    gateway_client: &Client,
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
    eval_info: &EvaluationInfo,
) -> Result<Analysis, Error> {
    // Acquire semaphore permit for concurrency control
    let _permit = semaphore.acquire().await.map_err(|e| {
        Error::new(ErrorDetails::Inference {
            message: format!("Failed to acquire semaphore: {e}"),
        })
    })?;

    // Create analyze variant configuration
    let analyze_config = create_analyze_variant_config(gepa_config);
    let analyze_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(analyze_config),
        timeouts: None,
    };

    let arguments = build_analyze_input(eval_info, function_context, variant_config)?;

    // Create ClientInferenceParams for the analyze function
    let params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
        input: Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments,
                })],
            }],
            system: None,
        },
        dryrun: Some(true), // Required when using internal_dynamic_variant_config
        internal: true,
        internal_dynamic_variant_config: Some(analyze_variant_config.clone()),
        ..Default::default()
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
        }));
    };

    // Extract text content from the response
    let InferenceResponse::Chat(chat_response) = &response else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "analyze function is defined as Chat, cannot return JSON".to_string(),
        }));
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
        Some(text) => text.text,
        None => {
            return Err(Error::new(ErrorDetails::Inference {
                message:
                    "Expected at least one Text content block from analyze function, found none"
                        .to_string(),
            }));
        }
    };

    tracing::debug!("Generated analysis: {}", analysis);

    // Conditionally include inference context based on config flag
    let inference = if gepa_config.include_inference_for_mutation {
        // Strip signatures from input at the type level
        let mut stored_input = eval_info
            .datapoint
            .input()
            .clone()
            .into_stored_input_without_file_handling()?;
        strip_signatures_from_stored_input(&mut stored_input);

        // Strip signatures from output at the type level (Chat only)
        let output_value = match &eval_info.response {
            InferenceResponse::Chat(chat_response) => {
                let mut content = chat_response.content.clone();
                strip_signatures_from_chat_output(&mut content);
                to_value(&content)?
            }
            InferenceResponse::Json(json_response) => to_value(&json_response.output)?,
        };

        Some(Inference {
            input: stored_input,
            output: output_value,
        })
    } else {
        None
    };

    Ok(Analysis {
        inference,
        analysis,
    })
}

/// Analyzes inference outputs using the analyze function in parallel.
///
/// Executes analyses with controlled concurrency and graceful degradation for failures.
///
/// Returns successful analyses with feedback and optional inference context.
///
/// Returns error only if all analyses fail.
pub async fn analyze_inferences(
    gateway_client: &Client,
    evaluation_infos: &[EvaluationInfo],
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
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

    // Create semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(max_concurrency));

    // Create futures for parallel execution
    let analysis_futures: Vec<_> = evaluation_infos
        .iter()
        .map(|eval_info| {
            let semaphore = Arc::clone(&semaphore);

            analyze_inference(
                semaphore,
                gateway_client,
                function_context,
                variant_config,
                gepa_config,
                eval_info,
            )
        })
        .collect();

    // Execute all analyses in parallel (graceful degradation on failures)
    let results = join_all(analysis_futures).await;

    // Partition into successes and failures
    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .enumerate()
        .partition(|(_, result)| result.is_ok());

    let successes = successes
        .into_iter()
        .filter_map(|(_, result)| result.ok())
        .collect::<Vec<_>>();

    let failures = failures
        .into_iter()
        .filter_map(|(index, result)| {
            result.err().map(|e| {
                tracing::warn!(
                    "Analysis failed for inference {}/{}: {}",
                    index + 1,
                    evaluation_infos.len(),
                    e
                );
                e
            })
        })
        .collect::<Vec<_>>();

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
        config::{SchemaData, path::ResolvedTomlPathData},
        db::stored_datapoint::StoredChatInferenceDatapoint,
        endpoints::{
            datasets::Datapoint,
            inference::{ChatInferenceResponse, InferenceResponse},
        },
        evaluations::{EvaluationConfig, InferenceEvaluationConfig},
        function::{FunctionConfig, FunctionConfigChat},
        inference::types::{ContentBlockChatOutput, Input, Text, Usage},
        jsonschema_util::{JSONSchema, SchemaWithMetadata},
        optimization::gepa::GEPAConfig,
        tool::StaticToolConfig,
        utils::retries::RetryConfig,
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
        let system_schema = JSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            }
        }))
        .unwrap();

        let user_schema = JSONSchema::from_value(json!({
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
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        })
    }

    /// Create a test UninitializedChatCompletionConfig using templates.inner
    fn create_test_variant_config() -> UninitializedChatCompletionConfig {
        let mut config = UninitializedChatCompletionConfig {
            model: "test-model".into(),
            weight: None,
            ..Default::default()
        };

        config.templates.inner.insert(
            "system".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "system.minijinja".to_string(),
                    "You are a helpful assistant.".to_string(),
                ),
            },
        );
        config.templates.inner.insert(
            "user".to_string(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    "user.minijinja".to_string(),
                    "User: {{input}}".to_string(),
                ),
            },
        );

        config
    }

    /// Create a test EvaluationInfo
    fn create_test_evaluation_info() -> EvaluationInfo {
        let input = Input {
            messages: vec![],
            system: None,
        };

        // Convert Input to StoredInput via JSON round-trip
        let stored_input = serde_json::from_value(to_value(&input).unwrap()).unwrap();

        let stored_datapoint = StoredChatInferenceDatapoint {
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
            snapshot_hash: None,
        };

        let datapoint = Datapoint::Chat(stored_datapoint.into_datapoint());

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
            description: Some("evaluation".to_string()),
        })
    }

    // ============================================================================
    // Unit Tests for create_analyze_variant_config
    // ============================================================================

    #[test]
    fn test_create_analyze_variant_config() {
        // Create a test GEPAConfig with specific values
        let gepa_config = GEPAConfig {
            function_name: "test_function".to_string(),
            evaluation_name: "test_evaluation".to_string(),
            initial_variants: None,
            variant_prefix: None,
            batch_size: 1,
            max_iterations: 1,
            max_concurrency: 10,
            analysis_model: "test-analysis-model".to_string(),
            mutation_model: "test-mutation-model".to_string(),
            seed: None,
            timeout: 300,
            include_inference_for_mutation: true,
            retries: RetryConfig {
                num_retries: 5,
                max_delay_s: 30.0,
            },
            max_tokens: Some(1000),
        };

        let config = create_analyze_variant_config(&gepa_config);

        // Verify model is set correctly
        assert_eq!(&*config.model, "test-analysis-model");

        // Verify retries is set correctly
        assert_eq!(config.retries.num_retries, 5);
        assert_eq!(config.retries.max_delay_s, 30.0);

        // Verify max_tokens is set correctly
        assert_eq!(config.max_tokens, Some(1000));

        // Verify weight is None (default)
        assert_eq!(config.weight, None);

        // Verify templates are created
        assert_eq!(config.templates.inner.len(), 2);
        assert!(config.templates.inner.contains_key("system"));
        assert!(config.templates.inner.contains_key("user"));

        // Verify system template path and content
        let system_template = config.templates.inner.get("system").unwrap();
        assert!(
            system_template
                .path
                .get_template_key()
                .ends_with("system.minijinja")
        );
        let system_content = system_template.path.data();
        assert!(system_content.contains("You are an expert in diagnosing quality issues"));
        assert!(system_content.contains("## Context"));
        assert!(system_content.contains("## Your Task"));
        assert!(system_content.contains("These response types are mutually exclusive"));
        assert!(system_content.contains("<report_error>"));
        assert!(system_content.contains("<report_improvement>"));
        assert!(system_content.contains("<report_optimal>"));

        // Verify user template path and content
        let user_template = config.templates.inner.get("user").unwrap();
        assert!(
            user_template
                .path
                .get_template_key()
                .ends_with("user.minijinja")
        );
        let user_content = user_template.path.data();
        assert!(user_content.contains("<function_context>"));
        assert!(user_content.contains("<inference_context>"));
        assert!(user_content.contains("{{function_config"));
        assert!(user_content.contains("{{evaluation_scores"));
        assert!(user_content.contains("tojson(indent=2)"));
        assert!(user_content.contains("Analyze the inference_output given the"));
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

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools,
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_analyze_input(&eval_info, &function_context, &variant_config);

        assert!(result.is_ok());
        let input = result.unwrap();

        // Check all required high-level fields are present
        assert!(input.0.get("function_config").is_some());
        assert!(input.0.get("static_tools").is_none()); // static_tools is None, so should not be present
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
        let static_tools_schemas = None;
        let eval_config_schemas = create_test_evaluation_config();
        let function_context_with_schemas = FunctionContext {
            function_config: Arc::new(function_config_with_schemas),
            static_tools: static_tools_schemas,
            evaluation_config: Arc::new(eval_config_schemas),
        };
        let result_with_schemas =
            build_analyze_input(&eval_info, &function_context_with_schemas, &variant_config);

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

        let function_config = create_test_function_config();
        let static_tools = None;
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools,
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_analyze_input(&eval_info, &function_context, &variant_config);

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

        let function_config = create_test_function_config();
        let static_tools = None;
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools,
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_analyze_input(&eval_info, &function_context, &variant_config);

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
            key: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: JSONSchema::from_value(json!({
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

        let eval_info = create_test_evaluation_info();
        let function_config = create_test_function_config();
        let variant_config = create_test_variant_config();
        let eval_config = create_test_evaluation_config();

        let function_context = FunctionContext {
            function_config: Arc::new(function_config),
            static_tools: Some(static_tools),
            evaluation_config: Arc::new(eval_config),
        };

        let result = build_analyze_input(&eval_info, &function_context, &variant_config);

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

    // ============================================================================
    // Unit Tests for type-safe thought signature stripping
    // ============================================================================

    use tensorzero_core::inference::types::{Thought, ThoughtSummaryBlock};

    #[test]
    fn test_strip_signatures_from_chat_output_removes_signature() {
        let mut blocks = vec![ContentBlockChatOutput::Thought(Thought {
            text: Some("Let me think...".to_string()),
            signature: Some("encrypted-blob".to_string()),
            summary: None,
            provider_type: Some("anthropic".to_string()),
            extra_data: None,
        })];

        strip_signatures_from_chat_output(&mut blocks);

        let ContentBlockChatOutput::Thought(thought) = &blocks[0] else {
            panic!("Expected Thought block");
        };
        assert!(
            thought.signature.is_none(),
            "signature should be removed from thought block"
        );
        assert_eq!(
            thought.text,
            Some("Let me think...".to_string()),
            "text should be preserved"
        );
        assert_eq!(
            thought.provider_type,
            Some("anthropic".to_string()),
            "provider_type should be preserved"
        );
    }

    #[test]
    fn test_strip_signatures_from_chat_output_mixed_blocks() {
        let mut blocks = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello".to_string(),
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Thinking...".to_string()),
                signature: Some("sig123".to_string()),
                summary: Some(vec![ThoughtSummaryBlock::SummaryText {
                    text: "Summary".to_string(),
                }]),
                provider_type: None,
                extra_data: None,
            }),
            ContentBlockChatOutput::Text(Text {
                text: "World".to_string(),
            }),
        ];

        strip_signatures_from_chat_output(&mut blocks);

        // First text block unchanged
        let ContentBlockChatOutput::Text(text1) = &blocks[0] else {
            panic!("Expected Text block");
        };
        assert_eq!(text1.text, "Hello", "first text block should be preserved");

        // Thought block has signature removed
        let ContentBlockChatOutput::Thought(thought) = &blocks[1] else {
            panic!("Expected Thought block");
        };
        assert!(
            thought.signature.is_none(),
            "signature should be removed from thought block"
        );
        assert_eq!(
            thought.text,
            Some("Thinking...".to_string()),
            "thought text should be preserved"
        );
        assert_eq!(
            thought.summary,
            Some(vec![ThoughtSummaryBlock::SummaryText {
                text: "Summary".to_string(),
            }]),
            "thought summary should be preserved"
        );

        // Second text block unchanged
        let ContentBlockChatOutput::Text(text2) = &blocks[2] else {
            panic!("Expected Text block");
        };
        assert_eq!(text2.text, "World", "second text block should be preserved");
    }

    #[test]
    fn test_strip_signatures_from_chat_output_multiple_thoughts() {
        let mut blocks = vec![
            ContentBlockChatOutput::Thought(Thought {
                text: Some("First thought".to_string()),
                signature: Some("sig1".to_string()),
                summary: None,
                provider_type: None,
                extra_data: None,
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Second thought".to_string()),
                signature: Some("sig2".to_string()),
                summary: Some(vec![ThoughtSummaryBlock::SummaryText {
                    text: "A summary".to_string(),
                }]),
                provider_type: Some("anthropic".to_string()),
                extra_data: None,
            }),
        ];

        strip_signatures_from_chat_output(&mut blocks);

        for (i, block) in blocks.iter().enumerate() {
            let ContentBlockChatOutput::Thought(thought) = block else {
                panic!("Expected Thought block at index {i}");
            };
            assert!(
                thought.signature.is_none(),
                "signature should be removed from thought block at index {i}"
            );
        }

        // Verify other fields preserved
        let ContentBlockChatOutput::Thought(first) = &blocks[0] else {
            panic!("Expected Thought");
        };
        assert_eq!(first.text, Some("First thought".to_string()));

        let ContentBlockChatOutput::Thought(second) = &blocks[1] else {
            panic!("Expected Thought");
        };
        assert_eq!(second.text, Some("Second thought".to_string()));
        assert_eq!(
            second.summary,
            Some(vec![ThoughtSummaryBlock::SummaryText {
                text: "A summary".to_string(),
            }])
        );
        assert_eq!(second.provider_type, Some("anthropic".to_string()));
    }

    #[test]
    fn test_strip_signatures_from_chat_output_no_signature() {
        let mut blocks = vec![ContentBlockChatOutput::Thought(Thought {
            text: Some("Already clean".to_string()),
            signature: None,
            summary: None,
            provider_type: None,
            extra_data: None,
        })];

        strip_signatures_from_chat_output(&mut blocks);

        let ContentBlockChatOutput::Thought(thought) = &blocks[0] else {
            panic!("Expected Thought block");
        };
        assert!(thought.signature.is_none(), "signature should remain None");
        assert_eq!(
            thought.text,
            Some("Already clean".to_string()),
            "text should be preserved"
        );
    }

    #[test]
    fn test_strip_signatures_from_chat_output_empty() {
        let mut blocks: Vec<ContentBlockChatOutput> = vec![];
        strip_signatures_from_chat_output(&mut blocks);
        assert!(blocks.is_empty(), "empty vector should remain empty");
    }

    #[test]
    fn test_strip_signatures_from_input_removes_signature() {
        let mut input = Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Thought(Thought {
                    text: Some("User thought".to_string()),
                    signature: Some("user-sig".to_string()),
                    summary: None,
                    provider_type: None,
                    extra_data: None,
                })],
            }],
            system: None,
        };

        strip_signatures_from_input(&mut input);

        let InputMessageContent::Thought(thought) = &input.messages[0].content[0] else {
            panic!("Expected Thought content");
        };
        assert!(
            thought.signature.is_none(),
            "signature should be removed from input thought"
        );
        assert_eq!(
            thought.text,
            Some("User thought".to_string()),
            "text should be preserved"
        );
    }

    #[test]
    fn test_strip_signatures_from_stored_input_removes_signature() {
        // Create StoredInput with a Thought block
        let stored_input_json = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "thought",
                    "text": "Stored thought",
                    "signature": "stored-sig"
                }]
            }]
        });

        let mut stored_input: StoredInput =
            serde_json::from_value(stored_input_json).expect("Valid StoredInput JSON");

        strip_signatures_from_stored_input(&mut stored_input);

        let StoredInputMessageContent::Thought(thought) = &stored_input.messages[0].content[0]
        else {
            panic!("Expected Thought content");
        };
        assert!(
            thought.signature.is_none(),
            "signature should be removed from stored input thought"
        );
        assert_eq!(
            thought.text,
            Some("Stored thought".to_string()),
            "text should be preserved"
        );
    }

    // ============================================================================
    // Unit Tests for datapoint field filtering
    // ============================================================================

    #[test]
    fn test_serialize_filtered_datapoint_chat() {
        let eval_info = create_test_evaluation_info();
        let datapoint = match &eval_info.datapoint {
            Datapoint::Chat(dp) => Datapoint::Chat(dp.clone()),
            Datapoint::Json(_) => panic!("Expected Chat datapoint"),
        };

        let filtered = serialize_filtered_datapoint(&datapoint)
            .expect("Serialization should succeed, expected Chat datapoint with valid fields");
        let obj = filtered
            .as_object()
            .expect("Filtered result should be a JSON object");

        // Verify kept fields are present
        assert!(
            obj.contains_key("function_name"),
            "Expected function_name to be kept for context"
        );
        assert!(
            obj.contains_key("input"),
            "Expected input to be kept as core data for analysis"
        );
        // Note: output might be None in test data, so we check it's either present or absent as an Option
        // tool_params is flattened so its individual fields would be in the object
        // tags is optional but if present should be kept

        // Verify dropped fields are absent
        assert!(
            !obj.contains_key("dataset_name"),
            "Expected dataset_name to be dropped as not relevant for optimization"
        );
        assert!(
            !obj.contains_key("id"),
            "Expected id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("episode_id"),
            "Expected episode_id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("auxiliary"),
            "Expected auxiliary to be dropped as extra metadata"
        );
        assert!(
            !obj.contains_key("is_deleted"),
            "Expected is_deleted to be dropped as internal state flag"
        );
        assert!(
            !obj.contains_key("is_custom"),
            "Expected is_custom to be dropped as internal flag"
        );
        assert!(
            !obj.contains_key("source_inference_id"),
            "Expected source_inference_id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("staled_at"),
            "Expected staled_at to be dropped as timestamp not relevant"
        );
        assert!(
            !obj.contains_key("updated_at"),
            "Expected updated_at to be dropped as timestamp not relevant"
        );
        assert!(
            !obj.contains_key("name"),
            "Expected name to be dropped as optional name field typically not relevant"
        );
    }

    #[test]
    fn test_serialize_filtered_datapoint_json() {
        use tensorzero_core::{
            db::stored_datapoint::StoredJsonInferenceDatapoint,
            inference::types::JsonInferenceOutput,
        };

        // Create a JSON datapoint
        let input = Input {
            messages: vec![],
            system: None,
        };

        let stored_input = serde_json::from_value(to_value(&input).unwrap()).unwrap();

        let output_schema = json!({
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        });

        let stored_datapoint = StoredJsonInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: stored_input,
            output: Some(JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(json!({"result": "test"})),
            }),
            output_schema: output_schema.clone(),
            tags: Some(HashMap::new()),
            auxiliary: String::new(),
            is_deleted: false,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            name: Some("test_name".to_string()),
            snapshot_hash: None,
        };

        let datapoint = Datapoint::Json(stored_datapoint.into_datapoint());

        let filtered = serialize_filtered_datapoint(&datapoint)
            .expect("Serialization should succeed, expected JSON datapoint with valid fields");
        let obj = filtered
            .as_object()
            .expect("Filtered result should be a JSON object");

        // Verify kept fields are present
        assert!(
            obj.contains_key("function_name"),
            "Expected function_name to be kept for context"
        );
        assert!(
            obj.contains_key("input"),
            "Expected input to be kept as core data for analysis"
        );
        assert!(
            obj.contains_key("output_schema"),
            "Expected output_schema to be kept for understanding output format"
        );

        // Verify dropped fields are absent (same as Chat datapoint)
        assert!(
            !obj.contains_key("dataset_name"),
            "Expected dataset_name to be dropped as not relevant for optimization"
        );
        assert!(
            !obj.contains_key("id"),
            "Expected id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("episode_id"),
            "Expected episode_id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("auxiliary"),
            "Expected auxiliary to be dropped as extra metadata"
        );
        assert!(
            !obj.contains_key("is_deleted"),
            "Expected is_deleted to be dropped as internal state flag"
        );
        assert!(
            !obj.contains_key("is_custom"),
            "Expected is_custom to be dropped as internal flag"
        );
        assert!(
            !obj.contains_key("source_inference_id"),
            "Expected source_inference_id to be dropped as internal identifier"
        );
        assert!(
            !obj.contains_key("staled_at"),
            "Expected staled_at to be dropped as timestamp not relevant"
        );
        assert!(
            !obj.contains_key("updated_at"),
            "Expected updated_at to be dropped as timestamp not relevant"
        );
        assert!(
            !obj.contains_key("name"),
            "Expected name to be dropped as optional name field typically not relevant"
        );
    }
}
