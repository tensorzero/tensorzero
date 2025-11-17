//! Shared test helpers for GEPA e2e tests

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::sync::Arc;

use evaluations::EvaluationInfo;
use tensorzero_core::{
    config::{path::ResolvedTomlPath, SchemaData},
    endpoints::{
        datasets::{StoredChatInferenceDatapoint, StoredDatapoint},
        inference::{ChatInferenceResponse, InferenceResponse},
    },
    evaluations::{
        ExactMatchConfig, LLMJudgeConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat,
        LLMJudgeOptimize, LLMJudgeOutputType,
    },
    function::{FunctionConfig, FunctionConfigChat},
    inference::types::{ContentBlockChatOutput, FinishReason, Input, Text, Usage},
    jsonschema_util::{SchemaWithMetadata, StaticJSONSchema},
    optimization::gepa::GEPAConfig,
    variant::chat_completion::{UninitializedChatCompletionConfig, UninitializedChatTemplate},
};
use tensorzero_optimizers::gepa::{
    EvaluationConfigWithInstructions, EvaluatorConfigWithInstructions, FunctionConfigAndTools,
    InferenceWithAnalysis, LLMJudgeConfigWithInstructions,
};
use uuid::Uuid;

pub mod analyze;
pub mod mutate;

// ============================================================================
// Shared Test Helper Functions
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

/// Create a minimal FunctionConfigAndTools for testing (no tools)
pub fn create_test_config_and_tools() -> FunctionConfigAndTools {
    FunctionConfigAndTools {
        function_config: Arc::new(create_test_function_config()),
        static_tools: None,
    }
}

/// Create a FunctionConfigAndTools with schemas for testing (no tools)
pub fn create_test_config_and_tools_with_schemas() -> FunctionConfigAndTools {
    FunctionConfigAndTools {
        function_config: Arc::new(create_test_function_config_with_schemas()),
        static_tools: None,
    }
}

/// Create a minimal JSON FunctionConfigAndTools for testing (no tools)
pub fn create_test_json_config_and_tools() -> FunctionConfigAndTools {
    FunctionConfigAndTools {
        function_config: Arc::new(create_test_json_function_config()),
        static_tools: None,
    }
}

/// Create a FunctionConfigAndTools with static tools (calculator and weather) for testing
pub fn create_test_config_and_tools_with_static_tools() -> FunctionConfigAndTools {
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
    let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        tools: vec!["calculator".to_string(), "weather".to_string()],
        tool_choice: tensorzero_core::tool::ToolChoice::Auto,
        parallel_tool_calls: None,
        description: Some("Test function with static tools".to_string()),
        all_explicit_templates_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    }));

    // Create static_tools HashMap
    let mut static_tools = HashMap::new();
    static_tools.insert("calculator".to_string(), Arc::new(calculator_tool));
    static_tools.insert("weather".to_string(), Arc::new(weather_tool));

    FunctionConfigAndTools {
        function_config,
        static_tools: Some(static_tools),
    }
}

/// Create a test UninitializedChatCompletionConfig with simple templates (new format)
pub fn create_test_variant_config() -> UninitializedChatCompletionConfig {
    let mut config = UninitializedChatCompletionConfig {
        model: "dummy::echo_request_messages".into(),
        weight: None,
        ..Default::default()
    };

    // Use new format: populate templates.inner
    config.templates.inner.insert(
        "system".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
                "system.minijinja".to_string(),
                "You are a helpful assistant for testing.".to_string(),
            ),
        },
    );
    config.templates.inner.insert(
        "user".to_string(),
        UninitializedChatTemplate {
            path: ResolvedTomlPath::new_fake_path(
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

    // Populate templates.inner with provided template map
    for (template_name, content) in template_map {
        config.templates.inner.insert(
            template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPath::new_fake_path(
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
        include_inference_input_for_mutation: false,
        retries: tensorzero_core::utils::retries::RetryConfig::default(),
        max_tokens: 16_384,
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
        include_inference_input_for_mutation: false,
        retries: tensorzero_core::utils::retries::RetryConfig::default(),
        max_tokens: 16_384,
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

/// Create mock InferenceWithAnalysis for testing mutate
pub fn create_test_inference_with_analysis(
    variant_name: &str,
    output_text: &str,
    analysis_text: &str,
) -> InferenceWithAnalysis {
    let inference_output = InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: variant_name.to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: output_text.to_string(),
        })],
        usage: Usage::default(),
        original_response: None,
        finish_reason: Some(FinishReason::Stop),
    });

    InferenceWithAnalysis {
        inference_output,
        analysis: vec![ContentBlockChatOutput::Text(Text {
            text: analysis_text.to_string(),
        })],
        inference_input: None,
    }
}

/// Create a test EvaluationConfigWithInstructions with empty evaluators
pub fn create_test_evaluation_config(
) -> tensorzero_optimizers::gepa::EvaluationConfigWithInstructions {
    tensorzero_optimizers::gepa::EvaluationConfigWithInstructions {
        evaluators: HashMap::new(),
        function_name: "test_function".to_string(),
    }
}

/// Create a test EvaluationConfigWithInstructions with test evaluators for tests that use scores
pub fn create_test_evaluation_config_with_evaluators() -> EvaluationConfigWithInstructions {
    let mut evaluators = HashMap::new();

    // Add ExactMatch evaluator
    evaluators.insert(
        "exact_match".to_string(),
        EvaluatorConfigWithInstructions::ExactMatch(ExactMatchConfig { cutoff: Some(0.8) }),
    );

    // Add fluency evaluator (Float type)
    evaluators.insert(
        "fluency".to_string(),
        EvaluatorConfigWithInstructions::LLMJudge(LLMJudgeConfigWithInstructions {
            config: LLMJudgeConfig {
                input_format: LLMJudgeInputFormat::Serialized,
                output_type: LLMJudgeOutputType::Float,
                include: LLMJudgeIncludeConfig {
                    reference_output: false,
                },
                optimize: LLMJudgeOptimize::Max,
                cutoff: Some(0.5),
            },
            system_instructions: "Evaluate fluency of the response.".to_string(),
        }),
    );

    EvaluationConfigWithInstructions {
        evaluators,
        function_name: "test_function".to_string(),
    }
}
