#![expect(clippy::unwrap_used, clippy::expect_used, clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::sync::Arc;

use evaluations::EvaluationInfo;
use serde_json::Value;
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::{
    config::{SchemaData, path::ResolvedTomlPathData},
    db::stored_datapoint::StoredChatInferenceDatapoint,
    endpoints::{
        datasets::Datapoint,
        inference::{ChatInferenceResponse, InferenceResponse},
    },
    evaluations::{
        EvaluationConfig, EvaluatorConfig, ExactMatchConfig, InferenceEvaluationConfig,
        LLMJudgeConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
        LLMJudgeOutputType,
    },
    function::{FunctionConfig, FunctionConfigChat},
    inference::types::{ContentBlockChatOutput, FinishReason, Input, Text, Usage},
    jsonschema_util::{JSONSchema, SchemaWithMetadata},
    optimization::gepa::GEPAConfig,
    tool::StaticToolConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};
use tensorzero_optimizers::gepa::{analyze::analyze_inferences, validate::FunctionContext};
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
    use tensorzero_core::{function::FunctionConfigJson, tool::create_json_mode_tool_call_config};

    let output_schema = JSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        },
        "required": ["result"]
    }))
    .expect("Failed to create JSON output schema");

    let json_mode_tool_call_config = create_json_mode_tool_call_config(output_schema.clone());

    FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        output_schema,
        json_mode_tool_call_config,
        description: Some("Test JSON function for GEPA e2e tests".to_string()),
        all_explicit_template_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    })
}

/// Create a Chat FunctionConfig with schemas for validation tests
pub fn create_test_function_config_with_schemas() -> FunctionConfig {
    let system_schema = JSONSchema::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "greeting": {"type": "string"}
        }
    }))
    .expect("Failed to create system schema");

    let user_schema = JSONSchema::from_value(serde_json::json!({
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
    let calculator_schema = JSONSchema::from_value(serde_json::json!({
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
        key: "calculator".to_string(),
        description: "Evaluates mathematical expressions".to_string(),
        parameters: calculator_schema,
        strict: true,
    };

    // Create weather tool
    let weather_schema = JSONSchema::from_value(serde_json::json!({
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
        key: "weather".to_string(),
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

    let stored_datapoint = StoredChatInferenceDatapoint {
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
        snapshot_hash: None,
    };

    let datapoint = Datapoint::Chat(stored_datapoint.into_datapoint());

    let response = InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: output_text.to_string(),
        })],
        usage: Usage::default(),
        raw_usage: None,
        original_response: None,
        raw_response: None,
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
        description: Some("empty evaluation".to_string()),
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
            description: Some("fluency evaluation".to_string()),
        }),
    );

    EvaluationConfig::Inference(InferenceEvaluationConfig {
        evaluators,
        function_name: "test_function".to_string(),
        description: Some("evaluation with evaluators".to_string()),
    })
}

/// Extract the user message content from echo model response
/// The echo model returns a JSON representation of the request; parse it once for reuse.
fn parse_echo_payload(analysis: &str) -> Value {
    serde_json::from_str(analysis)
        .expect("Echo response should be valid JSON with system and messages fields")
}

/// Extract concatenated text content for a given role from the echo payload
fn extract_role_text(payload: &Value, role: &str) -> String {
    payload
        .get("messages")
        .and_then(|m| m.as_array())
        .into_iter()
        .flatten()
        .filter(|msg| msg.get("role").and_then(|r| r.as_str()) == Some(role))
        .flat_map(|msg| {
            msg.get("content")
                .and_then(|c| c.as_array())
                .into_iter()
                .flatten()
        })
        .filter_map(|block| block.get("text").and_then(|t| t.as_str()))
        .collect::<String>()
}

struct InputScenario {
    eval_infos: Vec<EvaluationInfo>,
    function_config: FunctionConfig,
    static_tools: Option<HashMap<String, Arc<StaticToolConfig>>>,
    eval_config: EvaluationConfig,
    assert_fn: fn(&Value),
}

fn assert_schema_payload(payload: &Value) {
    let user_message = extract_role_text(payload, "user");

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
    assert!(
        user_message.contains("schemas"),
        "Should contain schemas in function_config JSON"
    );
    assert!(
        user_message.contains("system") || user_message.contains("user"),
        "Should contain schema role names (system/user)"
    );
    assert!(
        user_message.contains("exact_match") && user_message.contains("fluency"),
        "Should contain evaluator names"
    );
    assert!(
        user_message.contains("llm_judge"),
        "Should mention llm_judge evaluator type for fluency"
    );
    assert!(
        user_message.contains("exact_match") && user_message.contains("0.85"),
        "Should include exact_match score of 0.85"
    );
    assert!(
        user_message.contains("fluency") && user_message.contains("0.92"),
        "Should include fluency score of 0.92"
    );
}

fn assert_static_tools_payload(payload: &Value) {
    let user_message = extract_role_text(payload, "user");

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
    assert!(
        user_message.contains("calculator") && user_message.contains("weather"),
        "Should list both tools"
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
        user_message.contains("expression") && user_message.contains("location"),
        "Should include tool parameters"
    );
    assert!(
        user_message.contains("exact_match") && user_message.contains("0.85"),
        "Should include exact_match score"
    );
}

fn assert_score_types_payload(payload: &Value) {
    let user_message = extract_role_text(payload, "user");

    assert!(
        user_message.contains("<evaluation_scores>"),
        "User message should contain evaluation_scores section"
    );
    assert!(
        user_message.contains("numeric_score") && user_message.contains("0.85"),
        "Should contain numeric_score with value 0.85"
    );
    assert!(
        user_message.contains("bool_true")
            && (user_message.contains("true") || user_message.contains("1")),
        "Should contain bool_true with value true or 1"
    );
    assert!(
        user_message.contains("null_value"),
        "Should reference null_value evaluator"
    );
}

fn assert_tags_payload(payload: &Value) {
    let user_message = extract_role_text(payload, "user");

    assert!(
        user_message.contains("<datapoint>"),
        "User message should contain datapoint section with tags inside JSON"
    );
    assert!(
        user_message.contains("tags"),
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

fn assert_tool_params_payload(payload: &Value) {
    let user_message = extract_role_text(payload, "user");

    assert!(
        user_message.contains("<tool_schemas>"),
        "User message should contain <tool_schemas> section"
    );
    assert!(
        user_message.contains("calculator"),
        "User message should contain calculator tool (in allowed_tools)"
    );
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
    let static_tools = None;
    let eval_config = create_test_evaluation_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.analysis_model = "invalid_provider::nonexistent_model".to_string();

    let function_context = FunctionContext {
        function_config: Arc::new(function_config),
        static_tools,
        evaluation_config: Arc::new(eval_config),
    };

    // Execute: Should fail when all analyses fail
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_context,
        &variant_config,
        &gepa_config,
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
    let static_tools = None;
    let eval_config = create_test_evaluation_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    let function_context = FunctionContext {
        function_config: Arc::new(function_config),
        static_tools,
        evaluation_config: Arc::new(eval_config),
    };

    // Execute: Should handle schemas correctly
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_context,
        &variant_config,
        &gepa_config,
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

// ============================================================================
// Input Formatting Validation Tests (using echo model)
// ============================================================================

/// Helper function for echo model tests that validates input format
/// Returns the parsed echo payload for assertions
async fn test_analyze_input_echo_helper(
    eval_infos: Vec<EvaluationInfo>,
    function_config: FunctionConfig,
    static_tools: Option<HashMap<String, Arc<StaticToolConfig>>>,
    eval_config: EvaluationConfig,
) -> Value {
    let client = make_embedded_gateway().await;
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo();

    let function_context = FunctionContext {
        function_config: Arc::new(function_config),
        static_tools,
        evaluation_config: Arc::new(eval_config),
    };

    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_context,
        &variant_config,
        &gepa_config,
    )
    .await;

    assert!(
        result.is_ok(),
        "analyze_inferences should succeed: {:?}",
        result.as_ref().err()
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    parse_echo_payload(&analyses[0].analysis)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_includes_system_template() {
    let payload = Box::pin(test_analyze_input_echo_helper(
        vec![create_test_evaluation_info(
            "test_function",
            "Test input",
            "Test output",
        )],
        create_test_function_config(),
        None,
        create_test_evaluation_config(),
    ))
    .await;

    let system_prompt = payload
        .get("system")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        !system_prompt.is_empty(),
        "System prompt from analyze variant should be present"
    );
    assert!(
        system_prompt.contains("diagnosing quality issues in LLM-generated outputs"),
        "System prompt should include the GEPA analyze system template text, got: {system_prompt}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_format_scenarios() {
    let mut schemas_eval =
        create_test_evaluation_info("test_function", "Test input", "Test output");
    schemas_eval
        .evaluations
        .insert("exact_match".to_string(), Some(serde_json::json!(0.85)));
    schemas_eval
        .evaluations
        .insert("fluency".to_string(), Some(serde_json::json!(0.92)));

    let mut tools_eval =
        create_test_evaluation_info("test_function", "What is 2+2?", "The answer is 4.");
    tools_eval
        .evaluations
        .insert("exact_match".to_string(), Some(serde_json::json!(0.85)));
    let (static_tools_function_config, static_tools_for_static) =
        create_test_function_config_with_static_tools();

    let mut eval_info_scores =
        create_test_evaluation_info("test_function", "Test input", "Test output");
    eval_info_scores
        .evaluations
        .insert("numeric_score".to_string(), Some(serde_json::json!(0.85)));
    eval_info_scores
        .evaluations
        .insert("bool_true".to_string(), Some(serde_json::json!(true)));
    eval_info_scores
        .evaluations
        .insert("null_value".to_string(), Some(serde_json::json!(null)));

    let (tools_function_config_tool_params, static_tools_tool_params) =
        create_test_function_config_with_static_tools();

    let mut tags_eval =
        create_test_evaluation_info("test_function", "Tagged input", "Tagged output");
    if let Datapoint::Chat(datapoint) = &mut tags_eval.datapoint {
        datapoint.tags = Some(HashMap::from([
            ("environment".to_string(), "test".to_string()),
            ("user_id".to_string(), "12345".to_string()),
            ("experiment_id".to_string(), "exp-001".to_string()),
        ]));
    }

    let eval_info_tool_params = {
        use tensorzero_core::tool::{
            AllowedTools, AllowedToolsChoice, ToolCallConfigDatabaseInsert,
        };

        let input = tensorzero_core::inference::types::Input {
            messages: vec![],
            system: None,
        };
        let stored_input = serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();

        let tool_params = ToolCallConfigDatabaseInsert::new_for_test(
            vec![], // dynamic_tools
            vec![], // dynamic_provider_tools
            AllowedTools {
                tools: vec!["calculator".to_string()],
                choice: AllowedToolsChoice::Explicit,
            },
            tensorzero_core::tool::ToolChoice::Required,
            Some(false), // parallel_tool_calls
        );

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
            snapshot_hash: None,
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
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: Some(tensorzero_core::inference::types::FinishReason::Stop),
            },
        );

        evaluations::stats::EvaluationInfo {
            datapoint: Datapoint::Chat(datapoint.into_datapoint()),
            response,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
        }
    };

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
            description: Some("numeric evaluator".to_string()),
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
            description: Some("bool evaluator".to_string()),
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
            description: Some("null evaluator".to_string()),
        }),
    );
    let score_eval_config = EvaluationConfig::Inference(InferenceEvaluationConfig {
        evaluators,
        function_name: "test_function".to_string(),
        description: Some("empty evaluation".to_string()),
    });

    let scenarios = vec![
        InputScenario {
            eval_infos: vec![schemas_eval],
            function_config: create_test_function_config_with_schemas(),
            static_tools: None,
            eval_config: create_test_evaluation_config_with_evaluators(),
            assert_fn: assert_schema_payload,
        },
        InputScenario {
            eval_infos: vec![tools_eval],
            function_config: static_tools_function_config,
            static_tools: static_tools_for_static,
            eval_config: create_test_evaluation_config_with_evaluators(),
            assert_fn: assert_static_tools_payload,
        },
        InputScenario {
            eval_infos: vec![eval_info_scores],
            function_config: create_test_function_config(),
            static_tools: None,
            eval_config: score_eval_config,
            assert_fn: assert_score_types_payload,
        },
        InputScenario {
            eval_infos: vec![tags_eval],
            function_config: create_test_function_config(),
            static_tools: None,
            eval_config: create_test_evaluation_config(),
            assert_fn: assert_tags_payload,
        },
        InputScenario {
            eval_infos: vec![eval_info_tool_params],
            function_config: tools_function_config_tool_params,
            static_tools: static_tools_tool_params,
            eval_config: create_test_evaluation_config(),
            assert_fn: assert_tool_params_payload,
        },
    ];

    for scenario in scenarios {
        let payload = Box::pin(test_analyze_input_echo_helper(
            scenario.eval_infos,
            scenario.function_config,
            scenario.static_tools,
            scenario.eval_config,
        ))
        .await;

        (scenario.assert_fn)(&payload);
    }
}
