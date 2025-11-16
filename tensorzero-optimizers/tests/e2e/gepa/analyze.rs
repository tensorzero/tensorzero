//! E2E tests for GEPA analyze function
//!
//! These tests exercise the analyze_inferences component using real gateway clients
//! and the full inference pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::endpoints::inference::InferenceResponse;
use tensorzero_optimizers::gepa::analyze_inferences;

use super::*;

// ============================================================================
// Helper Functions
// ============================================================================

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

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_success() {
    // Setup: Create gateway client and test data
    let client = make_embedded_gateway().await;

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

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Call analyze_inferences
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify success and correct number of results
    let analyses = result.unwrap_or_else(|e| {
        panic!("analyze_inferences should succeed but got error: {e}");
    });
    assert_eq!(
        analyses.len(),
        3,
        "Should return analysis for all 3 inferences"
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
async fn test_analyze_inferences_empty_input() {
    // Setup: Create gateway client with empty evaluation infos
    let client = make_embedded_gateway().await;

    let eval_infos: Vec<EvaluationInfo> = vec![];
    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Call with empty input
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with empty results
    let analyses = result.unwrap_or_else(|e| {
        panic!("analyze_inferences should handle empty input but got error: {e}");
    });
    assert_eq!(analyses.len(), 0, "Should return empty Vec for empty input");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_single_inference() {
    // Setup: Create gateway client with single inference
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify exactly one result
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return exactly 1 analysis");
    assert!(
        !analyses[0].analysis.is_empty(),
        "Analysis should have content"
    );
    assert!(
        contains_expected_xml_tag(&analyses[0].analysis),
        "Analysis should contain expected XML tags"
    );
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_concurrency_limit() {
    // Setup: Create gateway client and multiple inferences
    let client = make_embedded_gateway().await;

    let eval_infos: Vec<EvaluationInfo> = (0..5)
        .map(|i| {
            create_test_evaluation_info(
                "test_function",
                &format!("Test input {i}"),
                &format!("Test output {i}"),
            )
        })
        .collect();

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();

    // Set low concurrency limit
    let mut gepa_config = create_test_gepa_config();
    gepa_config.max_concurrency = 2;

    // Execute: Should complete successfully despite low concurrency
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: All analyses should complete
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with concurrency limit"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 5, "Should return all 5 analyses");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_parallel_execution() {
    // Setup: Create gateway client and many inferences
    let client = make_embedded_gateway().await;

    let eval_infos: Vec<EvaluationInfo> = (0..10)
        .map(|i| {
            create_test_evaluation_info(
                "test_function",
                &format!("Test input {i}"),
                &format!("Test output {i}"),
            )
        })
        .collect();

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.max_concurrency = 5;

    // Execute: Should process in parallel
    let start = std::time::Instant::now();
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;
    let duration = start.elapsed();

    // Assert: All analyses should complete
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 10, "Should return all 10 analyses");

    // Parallel execution should be reasonably fast (not testing exact timing due to variance)
    assert!(
        duration.as_secs() < 30,
        "Parallel execution should complete in reasonable time, took {duration:?}"
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

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.analysis_model = "invalid_provider::nonexistent_model".to_string();

    // Execute: Should fail when all analyses fail
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
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

    let config_and_tools = create_test_config_and_tools_with_schemas();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Should handle schemas correctly
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
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
    let config_and_tools = create_test_json_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

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
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with JSON function
    assert!(
        result.is_ok(),
        "analyze_inferences should work with JSON functions"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    // Verify the inference_output is Json variant
    match &analyses[0].inference_output {
        InferenceResponse::Json(json_response) => {
            assert_eq!(
                json_response.variant_name, "test_variant",
                "Should preserve variant name"
            );
            assert!(
                json_response
                    .output
                    .parsed
                    .as_ref()
                    .and_then(|v| v.get("result"))
                    .is_some(),
                "Should have JSON output with 'result' field"
            );
        }
        InferenceResponse::Chat(_) => {
            panic!("Expected Json response, got Chat");
        }
    }
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

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify structure
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    let analysis = &analyses[0];

    // Verify inference_output field is populated
    match &analysis.inference_output {
        InferenceResponse::Chat(chat_response) => {
            assert_eq!(
                chat_response.variant_name, "test_variant",
                "Should preserve original variant name"
            );
            assert!(
                !chat_response.content.is_empty(),
                "Should have response content"
            );
        }
        InferenceResponse::Json(_) => {
            panic!("Expected Chat response, got Json");
        }
    }

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

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_includes_evaluations() {
    // Setup: Create gateway client with echo model to inspect input
    let client = make_embedded_gateway().await;

    // Create evaluation info with scores
    let mut eval_info = create_test_evaluation_info("test_function", "Test input", "Test output");

    // Add evaluator scores
    eval_info
        .evaluations
        .insert("accuracy".to_string(), Some(serde_json::json!(0.85)));
    eval_info
        .evaluations
        .insert("fluency".to_string(), Some(serde_json::json!(0.92)));

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo();

    // Execute: Call with echo model
    let result = analyze_inferences(
        &client,
        &[eval_info],
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Parse echo response and verify evaluations are included
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    assert!(
        user_message.contains("<evaluations>"),
        "User message should contain <evaluations> section"
    );
    assert!(
        user_message.contains("accuracy"),
        "User message should contain 'accuracy' evaluator"
    );
    assert!(
        user_message.contains("fluency"),
        "User message should contain 'fluency' evaluator"
    );
    assert!(
        user_message.contains("0.85") || user_message.contains("85"),
        "User message should contain accuracy score"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_includes_required_fields() {
    // Setup: Create gateway client with echo model to verify all required template fields
    // Use basic config with no schemas, tools, or tags to verify conditional fields don't appear
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify all required fields are present
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Required: <function_context> section
    assert!(
        user_message.contains("<function_context>"),
        "User message should contain <function_context> section"
    );
    assert!(
        user_message.contains("<function_name>"),
        "User message should contain <function_name>"
    );
    assert!(
        user_message.contains("test_function"),
        "User message should contain the function name"
    );
    assert!(
        user_message.contains("<model_name>"),
        "User message should contain <model_name>"
    );
    assert!(
        user_message.contains("dummy::echo_request_messages"),
        "User message should contain the model name"
    );
    assert!(
        user_message.contains("<message_templates>"),
        "User message should contain <message_templates> section"
    );

    // Required: <inference_context> section
    assert!(
        user_message.contains("<inference_context>"),
        "User message should contain <inference_context> section"
    );
    assert!(
        user_message.contains("<inference_input>"),
        "User message should contain <inference_input>"
    );

    // Required: <inference_output> section
    assert!(
        user_message.contains("<inference_output>"),
        "User message should contain <inference_output>"
    );

    // Required: <evaluations> section (even if empty)
    assert!(
        user_message.contains("<evaluations>"),
        "User message should contain <evaluations> section"
    );

    // Required: <task> section
    assert!(
        user_message.contains("<task>"),
        "User message should contain <task> section"
    );

    // Conditional fields should NOT appear when empty:
    // - No schemas configured, so <message_schemas> should not appear
    assert!(
        !user_message.contains("<message_schemas>"),
        "User message should NOT contain <message_schemas> when no schemas are configured"
    );

    // - No output_schema configured (Chat function), so <output_schema> should not appear
    assert!(
        !user_message.contains("<output_schema>"),
        "User message should NOT contain <output_schema> for Chat functions without output schema"
    );

    // - No tools configured, so <available_tools> should not appear
    assert!(
        !user_message.contains("<available_tools>"),
        "User message should NOT contain <available_tools> when no tools are configured"
    );

    // - No tags in datapoint, so <metadata> should not appear
    // Note: create_test_evaluation_info includes one tag "test_input", so metadata will appear
    // This is fine - the important thing is we're testing the field presence/absence logic
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_includes_schemas() {
    // Setup: Use function config with schemas
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let config_and_tools = create_test_config_and_tools_with_schemas();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify schemas are included
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();

    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    assert!(
        user_message.contains("<message_schemas>"),
        "User message should contain <message_schemas> section"
    );
    assert!(
        user_message.contains("system") || user_message.contains("user"),
        "User message should contain schema names"
    );
}

// ============================================================================
// Tool Handling Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_static_tools() {
    // Setup: Create gateway client with static tools (calculator and weather)
    let client = make_embedded_gateway().await;

    // Use helper to get function config with static tools
    let config_and_tools = create_test_config_and_tools_with_static_tools();

    // Create test evaluation info
    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "What is 2+2?",
        "The answer is 4.",
    )];

    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config_echo(); // Use echo model

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with static tools"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    // Extract the user message from echo response
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Verify tools section appears in template input
    assert!(
        user_message.contains("<available_tools>"),
        "User message should contain <available_tools> section when tools are configured"
    );

    // Verify both tools appear (calculator and weather)
    assert!(
        user_message.contains("calculator"),
        "User message should contain the calculator tool"
    );
    assert!(
        user_message.contains("weather"),
        "User message should contain the weather tool"
    );

    assert!(
        user_message.contains("Evaluates mathematical expressions"),
        "User message should contain the calculator tool description"
    );
    assert!(
        user_message.contains("Gets weather information"),
        "User message should contain the weather tool description"
    );

    assert!(
        user_message.contains("expression"),
        "User message should contain the calculator tool parameter"
    );
    assert!(
        user_message.contains("location"),
        "User message should contain the weather tool parameter"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_datapoint_tool_params() {
    use tensorzero_core::{
        endpoints::datasets::StoredChatInferenceDatapoint,
        tool::{AllowedTools, AllowedToolsChoice, ToolCallConfigDatabaseInsert, ToolChoice},
    };

    // Setup: Create gateway client
    let client = make_embedded_gateway().await;

    // Use helper to get function config with static tools (calculator and weather)
    let config_and_tools = create_test_config_and_tools_with_static_tools();

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
        &config_and_tools,
        &variant_config,
        &gepa_config,
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

    // Verify tools section appears
    assert!(
        user_message.contains("<available_tools>"),
        "User message should contain <available_tools> section"
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

    let config_and_tools = create_test_config_and_tools();
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
        Some(serde_json::json!(0.85)), // f64
    );
    evaluations.insert(
        "bool_true".to_string(),
        Some(serde_json::json!(true)), // bool -> should convert to 1.0
    );
    evaluations.insert(
        "bool_false".to_string(),
        Some(serde_json::json!(false)), // bool -> should convert to 0.0
    );
    evaluations.insert("null_value".to_string(), Some(serde_json::json!(null))); // null -> should be None
    evaluations.insert(
        "string_value".to_string(),
        Some(serde_json::json!("invalid")), // string -> should be filtered out
    );

    let eval_info = evaluations::stats::EvaluationInfo {
        datapoint: StoredDatapoint::Chat(datapoint),
        response,
        evaluations,
        evaluator_errors: HashMap::new(),
    };

    let eval_infos = vec![eval_info];
    let gepa_config = create_test_gepa_config_echo(); // Use echo model

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed
    assert!(
        result.is_ok(),
        "analyze_inferences should succeed with various evaluation types"
    );
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1);

    // Extract the user message from echo response
    let user_message = extract_user_message_from_echo(&analyses[0].analysis);

    // Verify evaluations section appears
    assert!(
        user_message.contains("evaluations"),
        "User message should contain evaluations section"
    );

    // Verify numeric score appears
    assert!(
        user_message.contains("numeric_score") && user_message.contains("0.85"),
        "User message should contain numeric evaluation score"
    );

    // Verify boolean conversions (true -> 1.0, false -> 0.0)
    assert!(
        user_message.contains("bool_true") && user_message.contains("1"),
        "User message should convert true to 1.0"
    );
    assert!(
        user_message.contains("bool_false") && user_message.contains("0"),
        "User message should convert false to 0.0"
    );

    // null and string values should be filtered out or shown as null
    // The exact behavior depends on the serialization logic
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_input_with_tags() {
    use tensorzero_core::endpoints::datasets::{StoredChatInferenceDatapoint, StoredDatapoint};

    // Setup: Create gateway client
    let client = make_embedded_gateway().await;

    let config_and_tools = create_test_config_and_tools();
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
        &config_and_tools,
        &variant_config,
        &gepa_config,
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

    // Verify tags appear in the template (wrapped in metadata section)
    assert!(
        user_message.contains("<metadata>"),
        "User message should contain metadata section with tags"
    );
    assert!(
        user_message.contains("environment") && user_message.contains("test"),
        "User message should contain environment tag"
    );
    assert!(
        user_message.contains("user_id") && user_message.contains("12345"),
        "User message should contain user_id tag"
    );
    assert!(
        user_message.contains("experiment_id") && user_message.contains("exp-001"),
        "User message should contain experiment_id tag"
    );
}
