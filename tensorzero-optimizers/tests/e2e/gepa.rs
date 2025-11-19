//! Shared test helpers for GEPA e2e tests

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::sync::Arc;

use evaluations::EvaluationInfo;
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::{
    config::{path::ResolvedTomlPathData, SchemaData},
    endpoints::{
        datasets::{StoredChatInferenceDatapoint, StoredDatapoint},
        inference::{ChatInferenceResponse, InferenceResponse},
    },
    evaluations::{
        EvaluationConfig, EvaluatorConfig, ExactMatchConfig, InferenceEvaluationConfig,
        LLMJudgeConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
        LLMJudgeOutputType,
    },
    function::{FunctionConfig, FunctionConfigChat},
    inference::types::{ContentBlockChatOutput, FinishReason, Input, Text, Usage},
    jsonschema_util::{SchemaWithMetadata, StaticJSONSchema},
    optimization::gepa::GEPAConfig,
    tool::StaticToolConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};
use tensorzero_optimizers::gepa::analyze_inferences;
use uuid::Uuid;

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Create a minimal Chat FunctionConfig for testing
pub fn create_test_function_config() -> FunctionConfig {
    FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        tools: vec![],
        tool_choice: tensorzero_core::tool::ToolChoice::None,
        parallel_tool_calls: None,
        description: Some("Test function for GEPA e2e tests".to_string()),
        all_explicit_templates_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    })
}

/// Create a minimal JSON FunctionConfig for testing
pub fn create_test_json_function_config() -> FunctionConfig {
    use tensorzero_core::{function::FunctionConfigJson, tool::create_implicit_tool_call_config};

    let output_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        },
        "required": ["result"]
    }))
    .expect("Failed to create JSON output schema");

    let implicit_tool_call_config = create_implicit_tool_call_config(output_schema.clone());

    FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        output_schema,
        implicit_tool_call_config,
        description: Some("Test JSON function for GEPA e2e tests".to_string()),
        all_explicit_template_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    })
}

/// Create a Chat FunctionConfig with schemas for validation tests
pub fn create_test_function_config_with_schemas() -> FunctionConfig {
    let system_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "greeting": {"type": "string"}
        }
    }))
    .expect("Failed to create system schema");

    let user_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    }))
    .expect("Failed to create user schema");

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
        description: Some("Test function with schemas for GEPA e2e tests".to_string()),
        all_explicit_templates_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    })
}

/// Create a FunctionConfig with static tools (calculator and weather) for testing
/// Returns tuple of (FunctionConfig, Option<HashMap<tools>>) for use with analyze_inferences
pub fn create_test_function_config_with_static_tools() -> (
    FunctionConfig,
    Option<HashMap<String, Arc<StaticToolConfig>>>,
) {
    use tensorzero_core::tool::StaticToolConfig;

    // Create calculator tool
    let calculator_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }))
    .expect("Failed to create calculator schema");

    let calculator_tool = StaticToolConfig {
        name: "calculator".to_string(),
        description: "Evaluates mathematical expressions".to_string(),
        parameters: calculator_schema,
        strict: true,
    };

    // Create weather tool
    let weather_schema = StaticJSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location to get weather for"
            }
        },
        "required": ["location"]
    }))
    .expect("Failed to create weather schema");

    let weather_tool = StaticToolConfig {
        name: "weather".to_string(),
        description: "Gets weather information".to_string(),
        parameters: weather_schema,
        strict: true,
    };

    // Create FunctionConfig with tools
    let function_config = FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        tools: vec!["calculator".to_string(), "weather".to_string()],
        tool_choice: tensorzero_core::tool::ToolChoice::Auto,
        parallel_tool_calls: None,
        description: Some("Test function with static tools".to_string()),
        all_explicit_templates_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    });

    // Create static_tools HashMap
    let mut static_tools_map = HashMap::new();
    static_tools_map.insert("calculator".to_string(), Arc::new(calculator_tool));
    static_tools_map.insert("weather".to_string(), Arc::new(weather_tool));

    (function_config, Some(static_tools_map))
}

/// Create a test UninitializedChatCompletionConfig with simple templates (new format)
pub fn create_test_variant_config() -> UninitializedChatCompletionConfig {
    let mut config = UninitializedChatCompletionConfig {
        model: "dummy::echo_request_messages".into(),
        weight: None,
        ..Default::default()
    };

    config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPathData::new_fake_path(
                "system.minijinja".to_string(),
                "You are a helpful assistant for testing.".to_string(),
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

/// Create a test UninitializedChatCompletionConfig with new template format (templates.inner)
pub fn create_test_variant_config_with_templates_inner(
    template_map: HashMap<String, String>,
) -> UninitializedChatCompletionConfig {
    let mut config = UninitializedChatCompletionConfig {
        model: "dummy::echo_request_messages".into(),
        weight: None,
        ..Default::default()
    };

    for (template_name, content) in template_map {
        config.templates.inner.insert(
            template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    format!("{template_name}.minijinja"),
                    content,
                ),
            },
        );
    }

    config
}

/// Create a simple template map with basic system and user templates
pub fn create_simple_template_map() -> HashMap<String, String> {
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    template_map.insert("user".to_string(), "User: {{input}}".to_string());
    template_map
}

/// Create a custom template map from a list of (name, content) pairs
pub fn create_custom_template_map(templates: Vec<(&str, &str)>) -> HashMap<String, String> {
    templates
        .into_iter()
        .map(|(name, content)| (name.to_string(), content.to_string()))
        .collect()
}

/// Create a test GEPAConfig with reasonable defaults
pub fn create_test_gepa_config() -> GEPAConfig {
    GEPAConfig {
        function_name: "test_function".to_string(),
        evaluation_name: "test_eval".to_string(),
        initial_variants: None,
        variant_prefix: Some("test".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 10,
        analysis_model: "openai::gpt-4.1-nano".to_string(),
        mutation_model: "openai::gpt-4.1-nano".to_string(),
        seed: Some(42),
        timeout: 300,
        include_inference_for_mutation: false,
        retries: tensorzero_core::utils::retries::RetryConfig::default(),
        max_tokens: Some(16_384),
    }
}

/// Create a test GEPAConfig using echo model for input validation tests
pub fn create_test_gepa_config_echo() -> GEPAConfig {
    GEPAConfig {
        function_name: "test_function".to_string(),
        evaluation_name: "test_eval".to_string(),
        initial_variants: None,
        variant_prefix: Some("test".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 10,
        analysis_model: "dummy::echo_request_messages".to_string(),
        mutation_model: "dummy::echo_request_messages".to_string(),
        seed: Some(42),
        timeout: 300,
        include_inference_for_mutation: false,
        retries: tensorzero_core::utils::retries::RetryConfig::default(),
        max_tokens: Some(16_384),
    }
}

/// Create a test EvaluationInfo with simple text input/output
pub fn create_test_evaluation_info(
    function_name: &str,
    input_text: &str,
    output_text: &str,
) -> EvaluationInfo {
    let input = Input {
        messages: vec![],
        system: None,
    };

    // Convert Input to StoredInput via JSON round-trip
    let stored_input =
        serde_json::from_value(serde_json::to_value(&input).expect("Failed to serialize input"))
            .expect("Failed to deserialize stored input");

    let datapoint = StoredDatapoint::Chat(StoredChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: function_name.to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input: stored_input,
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: output_text.to_string(),
        })]),
        tool_params: None,
        tags: Some(HashMap::from([(
            "test_input".to_string(),
            input_text.to_string(),
        )])),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        name: None,
    });

    let response = InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: output_text.to_string(),
        })],
        usage: Usage::default(),
        original_response: None,
        finish_reason: Some(FinishReason::Stop),
    });

    EvaluationInfo {
        datapoint,
        response,
        evaluations: HashMap::new(),
        evaluator_errors: HashMap::new(),
    }
}

/// Create a test EvaluationConfig with empty evaluators
pub fn create_test_evaluation_config() -> EvaluationConfig {
    EvaluationConfig::Inference(InferenceEvaluationConfig {
        evaluators: HashMap::new(),
        function_name: "test_function".to_string(),
    })
}

/// Create a test EvaluationConfig with test evaluators for tests that use scores
pub fn create_test_evaluation_config_with_evaluators() -> EvaluationConfig {
    let mut evaluators = HashMap::new();

    // Add ExactMatch evaluator
    evaluators.insert(
        "exact_match".to_string(),
        EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.8) }),
    );

    // Add fluency evaluator (Float type)
    evaluators.insert(
        "fluency".to_string(),
        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
            optimize: LLMJudgeOptimize::Max,
            cutoff: Some(0.5),
        }),
    );

    EvaluationConfig::Inference(InferenceEvaluationConfig {
        evaluators,
        function_name: "test_function".to_string(),
    })
}

/// Check if analysis contains one of the expected XML tags
fn contains_expected_xml_tag(analysis: &[ContentBlockChatOutput]) -> bool {
    let text = analysis
        .iter()
        .filter_map(|block| match block {
            ContentBlockChatOutput::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<String>();

    text.contains("<report_error>")
        || text.contains("<report_improvement>")
        || text.contains("<report_optimal>")
}

/// Extract the user message content from echo model response
/// The echo model returns a JSON representation of the request
fn extract_user_message_from_echo(analysis: &[ContentBlockChatOutput]) -> String {
    analysis
        .iter()
        .filter_map(|block| match block {
            ContentBlockChatOutput::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<String>()
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/// Helper function for basic analyze_inferences tests
async fn test_analyze_inferences_with_eval_infos(
    eval_infos: Vec<EvaluationInfo>,
    expected_count: usize,
) {
    let client = make_embedded_gateway().await;
    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    let analyses = result.expect("analyze_inferences should succeed");

    assert_eq!(
        analyses.len(),
        expected_count,
        "Should return analysis for all {expected_count} inferences"
    );

    // Verify each analysis has content and expected XML tags
    for analysis in &analyses {
        assert!(
            !analysis.analysis.is_empty(),
            "Each analysis should have non-empty XML content"
        );
        assert!(
            contains_expected_xml_tag(&analysis.analysis),
            "Analysis should contain one of: <report_error>, <report_improvement>, or <report_optimal>"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_success() {
    let eval_infos = vec![
        create_test_evaluation_info("test_function", "What is 2+2?", "The answer is 4."),
        create_test_evaluation_info(
            "test_function",
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
        create_test_evaluation_info(
            "test_function",
            "Explain gravity.",
            "Gravity is a force that attracts objects with mass.",
        ),
    ];

    test_analyze_inferences_with_eval_infos(eval_infos, 3).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_empty_input() {
    test_analyze_inferences_with_eval_infos(vec![], 0).await;
}

// ============================================================================
// Concurrency Tests
// ============================================================================

/// Helper function for concurrency tests
async fn test_analyze_inferences_with_concurrency(num_inferences: usize, max_concurrency: u32) {
    let client = make_embedded_gateway().await;

    let eval_infos: Vec<EvaluationInfo> = (0..num_inferences)
        .map(|i| {
            create_test_evaluation_info(
                "test_function",
                &format!("Test input {i}"),
                &format!("Test output {i}"),
            )
        })
        .collect();

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.max_concurrency = max_concurrency;

    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with concurrency limit"
    );
    let analyses = result.unwrap();
    assert_eq!(
        analyses.len(),
        num_inferences,
        "Should return all {num_inferences} analyses"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_concurrency_limit() {
    test_analyze_inferences_with_concurrency(5, 2).await;
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_invalid_model() {
    // Setup: Test that invalid model causes all analyses to fail with proper error
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.analysis_model = "invalid_provider::nonexistent_model".to_string();

    // Execute: Should fail when all analyses fail
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Should return error with appropriate message
    assert!(result.is_err(), "Should error when all analyses fail");
    let error = result.unwrap_err();
    let error_msg = error.to_string();
    assert!(
        error_msg.contains("All") || error_msg.contains("fail"),
        "Error message should indicate failure: {error_msg}"
    );
}

// ============================================================================
// Input Validation Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_with_schemas() {
    // Setup: Use function config with schemas
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let function_config = create_test_function_config_with_schemas();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Should handle schemas correctly
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Should succeed with schemas present
    assert!(
        result.is_ok(),
        "analyze_inferences should work with schemas"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_json_function() {
    use tensorzero_core::{
        endpoints::{
            datasets::{JsonInferenceDatapoint, StoredDatapoint},
            inference::JsonInferenceResponse,
        },
        inference::types::{Input, JsonInferenceOutput},
    };

    // Setup: Test with actual JSON function and StoredDatapoint::Json
    let client = make_embedded_gateway().await;

    // Use JSON function config
    let function_config = create_test_json_function_config();
    let static_tools = None;
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.include_inference_for_mutation = true; // Enable to verify inference structure

    // Create a proper JSON datapoint
    let input = Input {
        messages: vec![],
        system: None,
    };
    let stored_input =
        serde_json::from_value(serde_json::to_value(&input).expect("Failed to serialize input"))
            .expect("Failed to deserialize stored input");

    let output_json = serde_json::json!({
        "result": "This is a test JSON output"
    });

    let output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        },
        "required": ["result"]
    });

    let datapoint = JsonInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "test_json_function".to_string(),
        id: uuid::Uuid::now_v7(),
        episode_id: Some(uuid::Uuid::now_v7()),
        input: stored_input,
        output: Some(JsonInferenceOutput {
            raw: Some(output_json.to_string()),
            parsed: Some(output_json.clone()),
        }),
        output_schema,
        tags: Some(HashMap::from([(
            "test_type".to_string(),
            "json_function_test".to_string(),
        )])),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        name: None,
    };

    let response = InferenceResponse::Json(JsonInferenceResponse {
        inference_id: uuid::Uuid::now_v7(),
        episode_id: uuid::Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        output: JsonInferenceOutput {
            raw: Some(output_json.to_string()),
            parsed: Some(output_json),
        },
        usage: tensorzero_core::inference::types::Usage::default(),
        original_response: None,
        finish_reason: Some(tensorzero_core::inference::types::FinishReason::Stop),
    });

    let eval_info = evaluations::stats::EvaluationInfo {
        datapoint: StoredDatapoint::Json(datapoint),
        response,
        evaluations: HashMap::new(),
        evaluator_errors: HashMap::new(),
    };

    let eval_infos = vec![eval_info];

    // Execute: Should handle JSON functions correctly
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &static_tools,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Should succeed with JSON function
    assert!(
        result.is_ok(),
        "analyze_inferences should work with JSON functions"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    // Verify the inference is Some and contains Json output
    let inference = analyses[0]
        .inference
        .as_ref()
        .expect("inference should be Some when include_inference_for_mutation is true");

    assert!(
        inference.output.is_object(),
        "JSON response output should be an object"
    );

    // Verify the output has the expected structure (parsed with 'result' field)
    let output_obj = inference.output.as_object().unwrap();
    assert!(
        output_obj.get("parsed").is_some(),
        "JSON output should have 'parsed' field"
    );
    let parsed = output_obj.get("parsed").unwrap();
    assert!(
        parsed.get("result").is_some(),
        "Should have 'result' field in parsed JSON output"
    );
}

// ============================================================================
// Response Parsing Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_response_structure() {
    // Setup: Verify the InferenceWithAnalysis structure is correct
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "What is the meaning of life?",
        "42",
    )];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.include_inference_for_mutation = true; // Enable to verify inference structure

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Verify structure
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    let analysis = &analyses[0];

    // Verify inference field is populated
    let inference = analysis
        .inference
        .as_ref()
        .expect("inference should be Some when include_inference_for_mutation is true");

    assert!(
        inference.output.is_array(),
        "Chat response content should be serialized as an array"
    );
    let content_array = inference.output.as_array().unwrap();
    assert!(!content_array.is_empty(), "Should have response content");

    // Verify analysis field is populated with expected XML tags
    assert!(
        !analysis.analysis.is_empty(),
        "Analysis field should be populated"
    );
    assert!(
        contains_expected_xml_tag(&analysis.analysis),
        "Analysis should contain expected XML tags"
    );
}

// ============================================================================
// Input Formatting Validation Tests (using echo model)
// ============================================================================

/// Helper function for echo model tests that validates input format
/// Returns the user_message from the echo response for assertion
async fn test_analyze_input_echo_helper(
    eval_infos: Vec<EvaluationInfo>,
    function_config: FunctionConfig,
    static_tools: Option<HashMap<String, Arc<StaticToolConfig>>>,
    eval_config: EvaluationConfig,
) -> String {
    let client = make_embedded_gateway().await;
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo();

    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &static_tools,
        &variant_config,
        &gepa_config,
        &eval_config,
    )
    .await;

    assert!(
        result.is_ok(),
        "analyze_inferences should succeed: {:?}",
        result.as_ref().err()
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    extract_user_message_from_echo(&analyses[0].analysis)
}

/// Comprehensive test for input template with schemas
/// Tests that schemas, required fields, and evaluation config all appear correctly in template
#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_schemas() {
    let mut eval_info = create_test_evaluation_info("test_function", "Test input", "Test output");
    eval_info
        .evaluations
        .insert("exact_match".to_string(), Some(serde_json::json!(0.85)));
    eval_info
        .evaluations
        .insert("fluency".to_string(), Some(serde_json::json!(0.92)));

    let function_config = create_test_function_config_with_schemas();
    let user_message = test_analyze_input_echo_helper(
        vec![eval_info],
        function_config,
        None,
        create_test_evaluation_config_with_evaluators(),
    )
    .await;

    // Verify required template sections exist
    assert!(
        user_message.contains("<function_context>"),
        "Should have function_context"
    );
    assert!(
        user_message.contains("<function_config>"),
        "Should have function_config"
    );
    assert!(
        user_message.contains("<inference_context>"),
        "Should have inference_context"
    );
    assert!(
        user_message.contains("<inference_output>"),
        "Should have inference_output"
    );
    assert!(
        user_message.contains("<evaluation_config>"),
        "Should have evaluation_config"
    );
    assert!(
        user_message.contains("<evaluation_scores>"),
        "Should have evaluation_scores"
    );

    // Verify schemas appear in function_config JSON
    assert!(
        user_message.contains(r#"\"schemas\""#) || user_message.contains("schemas"),
        "Should contain schemas in function_config JSON"
    );
    assert!(
        user_message.contains("system") || user_message.contains("user"),
        "Should contain schema role names (system/user)"
    );

    // Verify evaluation config includes evaluator definitions
    assert!(
        user_message.contains("exact_match"),
        "Should contain exact_match evaluator"
    );
    assert!(
        user_message.contains("fluency"),
        "Should contain fluency evaluator"
    );
    assert!(
        user_message.contains(r#"\"type\": \"llm_judge\""#)
            || user_message.contains("\"type\":\"llm_judge\""),
        "Should have llm_judge type for fluency"
    );

    // Verify evaluation scores appear
    assert!(
        (user_message.contains(r#"\"exact_match\": 0.85"#)
            || user_message.contains("\"exact_match\":0.85")),
        "Should have exact_match score of 0.85"
    );
    assert!(
        (user_message.contains(r#"\"fluency\": 0.92"#)
            || user_message.contains("\"fluency\":0.92")),
        "Should have fluency score of 0.92"
    );
}

/// Comprehensive test for input template with static tools
/// Tests that tools, required fields, and evaluation config all appear correctly in template
#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_static_tools() {
    let mut eval_info =
        create_test_evaluation_info("test_function", "What is 2+2?", "The answer is 4.");
    eval_info
        .evaluations
        .insert("exact_match".to_string(), Some(serde_json::json!(0.85)));

    let (function_config, static_tools) = create_test_function_config_with_static_tools();
    let user_message = test_analyze_input_echo_helper(
        vec![eval_info],
        function_config,
        static_tools,
        create_test_evaluation_config_with_evaluators(),
    )
    .await;

    // Verify required template sections exist
    assert!(
        user_message.contains("<function_context>"),
        "Should have function_context"
    );
    assert!(
        user_message.contains("<tool_schemas>"),
        "Should have tool_schemas section"
    );
    assert!(
        user_message.contains("<evaluation_config>"),
        "Should have evaluation_config"
    );
    assert!(
        user_message.contains("<evaluation_scores>"),
        "Should have evaluation_scores"
    );

    // Verify both tools appear with their details
    assert!(
        user_message.contains("calculator"),
        "Should contain calculator tool"
    );
    assert!(
        user_message.contains("weather"),
        "Should contain weather tool"
    );
    assert!(
        user_message.contains("Evaluates mathematical expressions"),
        "Should have calculator description"
    );
    assert!(
        user_message.contains("Gets weather information"),
        "Should have weather description"
    );
    assert!(
        user_message.contains("expression"),
        "Should have calculator parameter"
    );
    assert!(
        user_message.contains("location"),
        "Should have weather parameter"
    );

    // Verify evaluation config and scores
    assert!(
        user_message.contains("exact_match"),
        "Should contain exact_match evaluator"
    );
    assert!(
        (user_message.contains(r#"\"exact_match\": 0.85"#)
            || user_message.contains("\"exact_match\":0.85")),
        "Should have exact_match score"
    );
}

// ============================================================================
// Tool Handling Tests - Custom datapoint construction
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_datapoint_tool_params() {
    use tensorzero_core::{
        endpoints::datasets::StoredChatInferenceDatapoint,
        tool::{AllowedTools, AllowedToolsChoice, ToolCallConfigDatabaseInsert, ToolChoice},
    };

    // Setup: Create gateway client
    let client = make_embedded_gateway().await;

    // Use helper to get function config with static tools (calculator and weather)
    let (function_config, static_tools) = create_test_function_config_with_static_tools();

    // Create datapoint with tool_params that restricts to only calculator
    let tool_params = ToolCallConfigDatabaseInsert::new_for_test(
        vec![], // dynamic_tools
        vec![], // dynamic_provider_tools
        AllowedTools {
            tools: vec!["calculator".to_string()],
            choice: AllowedToolsChoice::Explicit,
        },
        ToolChoice::Required,
        Some(false), // parallel_tool_calls
    );

    // Create a custom datapoint with tool_params
    let input = tensorzero_core::inference::types::Input {
        messages: vec![],
        system: None,
    };
    let stored_input = serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

    let datapoint = StoredChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "test_function".to_string(),
        id: uuid::Uuid::now_v7(),
        episode_id: Some(uuid::Uuid::now_v7()),
        input: stored_input,
        output: Some(vec![ContentBlockChatOutput::Text(
            tensorzero_core::inference::types::Text {
                text: "The answer is 4.".to_string(),
            },
        )]),
        tool_params: Some(tool_params),
        tags: Some(HashMap::new()),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        name: None,
    };

    let response = InferenceResponse::Chat(
        tensorzero_core::endpoints::inference::ChatInferenceResponse {
            inference_id: uuid::Uuid::now_v7(),
            episode_id: uuid::Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::Text(
                tensorzero_core::inference::types::Text {
                    text: "The answer is 4.".to_string(),
                },
            )],
            usage: tensorzero_core::inference::types::Usage::default(),
            original_response: None,
            finish_reason: Some(tensorzero_core::inference::types::FinishReason::Stop),
        },
    );

    let eval_info = evaluations::stats::EvaluationInfo {
        datapoint: tensorzero_core::endpoints::datasets::StoredDatapoint::Chat(datapoint),
        response,
        evaluations: HashMap::new(),
        evaluator_errors: HashMap::new(),
    };

    let eval_infos = vec![eval_info];

    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo(); // Use echo model

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &static_tools,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Should succeed
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with datapoint tool_params. Error: {:?}",
        result.as_ref().err()
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    // Extract the user message from echo response
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Verify tools section appears (renamed from <available_tools> to <tool_schemas>)
    assert!(
        user_message.contains("<tool_schemas>"),
        "User message should contain <tool_schemas> section"
    );

    // Verify calculator is included (it's in allowed_tools)
    assert!(
        user_message.contains("calculator"),
        "User message should contain calculator tool (in allowed_tools)"
    );

    // Note: The exact behavior of whether weather tool appears depends on into_tool_call_config
    // The tool_params restricts to only calculator, so weather should not appear
    // However, this depends on the implementation details of ToolCallConfig serialization
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_evaluation_score_types() {
    use tensorzero_core::{
        endpoints::datasets::{StoredChatInferenceDatapoint, StoredDatapoint},
        endpoints::inference::ChatInferenceResponse,
    };

    // Setup: Create gateway client
    let client = make_embedded_gateway().await;

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();

    // Create a datapoint with different evaluation score types
    let input = tensorzero_core::inference::types::Input {
        messages: vec![],
        system: None,
    };
    let stored_input = serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

    let datapoint = StoredChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "test_function".to_string(),
        id: uuid::Uuid::now_v7(),
        episode_id: Some(uuid::Uuid::now_v7()),
        input: stored_input,
        output: Some(vec![ContentBlockChatOutput::Text(
            tensorzero_core::inference::types::Text {
                text: "Test output".to_string(),
            },
        )]),
        tool_params: None,
        tags: Some(HashMap::new()),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        name: None,
    };

    let response = InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: uuid::Uuid::now_v7(),
        episode_id: uuid::Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        content: vec![ContentBlockChatOutput::Text(
            tensorzero_core::inference::types::Text {
                text: "Test output".to_string(),
            },
        )],
        usage: tensorzero_core::inference::types::Usage::default(),
        original_response: None,
        finish_reason: Some(tensorzero_core::inference::types::FinishReason::Stop),
    });

    // Create evaluations with different types
    let mut evaluations = HashMap::new();
    evaluations.insert(
        "numeric_score".to_string(),
        Some(serde_json::json!(0.85)), // numeric value
    );
    evaluations.insert(
        "bool_true".to_string(),
        Some(serde_json::json!(true)), // boolean value
    );
    evaluations.insert("null_value".to_string(), Some(serde_json::json!(null))); // null value

    let eval_info = evaluations::stats::EvaluationInfo {
        datapoint: StoredDatapoint::Chat(datapoint),
        response,
        evaluations,
        evaluator_errors: HashMap::new(),
    };

    let eval_infos = vec![eval_info];
    let gepa_config = create_test_gepa_config_echo(); // Use echo model

    // Create evaluation config with score type evaluators for this test
    let mut evaluators = HashMap::new();
    evaluators.insert(
        "numeric_score".to_string(),
        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
            optimize: LLMJudgeOptimize::Max,
            cutoff: None,
        }),
    );
    evaluators.insert(
        "bool_true".to_string(),
        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Boolean,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
            optimize: LLMJudgeOptimize::Max,
            cutoff: None,
        }),
    );
    evaluators.insert(
        "null_value".to_string(),
        EvaluatorConfig::LLMJudge(LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
            optimize: LLMJudgeOptimize::Max,
            cutoff: None,
        }),
    );
    let eval_config = EvaluationConfig::Inference(InferenceEvaluationConfig {
        evaluators,
        function_name: "test_function".to_string(),
    });

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &eval_config,
    )
    .await;

    // Assert: Should succeed
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with various evaluation types: {:?}",
        result.as_ref().err()
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1);

    // Extract the user message from echo response
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Verify evaluation_scores section appears
    assert!(
        user_message.contains("<evaluation_scores>"),
        "User message should contain evaluation_scores section"
    );

    // Verify numeric score appears with correct value
    assert!(
        user_message.contains("numeric_score") && user_message.contains("0.85"),
        "User message should contain numeric_score with value 0.85"
    );

    // Verify boolean value appears (as true or 1)
    assert!(
        user_message.contains("bool_true")
            && (user_message.contains("true") || user_message.contains("1")),
        "User message should contain bool_true with value true or 1"
    );

    // Verify null value is handled (appears as null or is filtered out)
    // We just check that the evaluator name appears in the message
    assert!(
        user_message.contains("null_value"),
        "User message should reference null_value evaluator"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_tags() {
    use tensorzero_core::endpoints::datasets::{StoredChatInferenceDatapoint, StoredDatapoint};

    // Setup: Create gateway client
    let client = make_embedded_gateway().await;

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();

    // Create a datapoint with tags
    let input = tensorzero_core::inference::types::Input {
        messages: vec![],
        system: None,
    };
    let stored_input = serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

    let mut tags = HashMap::new();
    tags.insert("environment".to_string(), "test".to_string());
    tags.insert("user_id".to_string(), "12345".to_string());
    tags.insert("experiment_id".to_string(), "exp-001".to_string());

    let datapoint = StoredChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "test_function".to_string(),
        id: uuid::Uuid::now_v7(),
        episode_id: Some(uuid::Uuid::now_v7()),
        input: stored_input,
        output: Some(vec![ContentBlockChatOutput::Text(
            tensorzero_core::inference::types::Text {
                text: "Test output".to_string(),
            },
        )]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        is_deleted: false,
        is_custom: false,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-01-01T00:00:00Z".to_string(),
        name: None,
    };

    let response = InferenceResponse::Chat(
        tensorzero_core::endpoints::inference::ChatInferenceResponse {
            inference_id: uuid::Uuid::now_v7(),
            episode_id: uuid::Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::Text(
                tensorzero_core::inference::types::Text {
                    text: "Test output".to_string(),
                },
            )],
            usage: tensorzero_core::inference::types::Usage::default(),
            original_response: None,
            finish_reason: Some(tensorzero_core::inference::types::FinishReason::Stop),
        },
    );

    let eval_info = evaluations::stats::EvaluationInfo {
        datapoint: StoredDatapoint::Chat(datapoint),
        response,
        evaluations: HashMap::new(),
        evaluator_errors: HashMap::new(),
    };

    let eval_infos = vec![eval_info];
    let gepa_config = create_test_gepa_config_echo(); // Use echo model

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &None,
        &variant_config,
        &gepa_config,
        &create_test_evaluation_config(),
    )
    .await;

    // Assert: Should succeed
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with tags"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1);

    // Extract the user message from echo response
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Verify tags appear inside the datapoint JSON (not in a separate metadata section)
    assert!(
        user_message.contains("<datapoint>"),
        "User message should contain datapoint section with tags inside JSON"
    );
    assert!(
        user_message.contains(r#"\"tags\""#) || user_message.contains("tags"),
        "User message should contain tags field inside datapoint JSON"
    );
    assert!(
        user_message.contains("environment") && user_message.contains("test"),
        "User message should contain environment tag value"
    );
    assert!(
        user_message.contains("user_id") && user_message.contains("12345"),
        "User message should contain user_id tag value"
    );
    assert!(
        user_message.contains("experiment_id") && user_message.contains("exp-001"),
        "User message should contain experiment_id tag value"
    );
}
