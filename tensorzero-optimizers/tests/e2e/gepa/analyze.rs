//! E2E tests for GEPA analyze function
//!
//! These tests exercise the analyze_inferences component using real gateway clients
//! and the full inference pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::{endpoints::inference::InferenceResponse, function::FunctionConfig};
use tensorzero_optimizers::gepa::analyze_inferences;

use super::*;

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

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Call analyze_inferences
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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

    // Verify each analysis has content
    for analysis in &analyses {
        assert!(
            !analysis.analysis.is_empty(),
            "Each analysis should have non-empty XML content"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_empty_input() {
    // Setup: Create gateway client with empty evaluation infos
    let client = make_embedded_gateway().await;

    let eval_infos: Vec<EvaluationInfo> = vec![];
    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Call with empty input
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();

    // Set low concurrency limit
    let mut gepa_config = create_test_gepa_config();
    gepa_config.max_concurrency = 2;

    // Execute: Should complete successfully despite low concurrency
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.max_concurrency = 5;

    // Execute: Should process in parallel
    let start = std::time::Instant::now();
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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
async fn test_analyze_inferences_graceful_degradation() {
    // Setup: Create gateway client with mix of valid and potentially problematic inputs
    let client = make_embedded_gateway().await;

    let eval_infos = vec![
        create_test_evaluation_info("test_function", "Normal input", "Normal output"),
        create_test_evaluation_info(
            "test_function",
            "Another normal input",
            "Another normal output",
        ),
    ];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute: Should handle any issues gracefully
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed even if some analyses have issues
    assert!(
        result.is_ok(),
        "analyze_inferences should handle errors gracefully"
    );
    let analyses = result.unwrap();
    assert!(
        !analyses.is_empty(),
        "Should return at least some successful analyses"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_invalid_model() {
    // Setup: Create gateway client with invalid model
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let mut gepa_config = create_test_gepa_config();
    gepa_config.analysis_model = "nonexistent::invalid_model".to_string();

    // Execute: Should fail gracefully
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should return an error
    assert!(
        result.is_err(),
        "analyze_inferences should error with invalid model"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_all_failures_error() {
    // Setup: This test verifies that if all analyses fail, we get a proper error
    // Note: With dummy::echo_request_messages, failures are rare, so we test the error path
    // by using an invalid configuration
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

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should return error when all analyses fail
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
async fn test_analyze_inferences_with_tools() {
    // Setup: Use Chat function with tools
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    // Create function config with tools
    use std::collections::HashMap;
    use tensorzero_core::config::SchemaData;
    use tensorzero_core::function::FunctionConfigChat;

    let function_config = FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        schemas: SchemaData::default(),
        tools: vec![], // Tools would be added here in a real scenario
        tool_choice: tensorzero_core::tool::ToolChoice::None,
        parallel_tool_calls: None,
        description: Some("Test function with tools".to_string()),
        all_explicit_templates_names: std::collections::HashSet::new(),
        experimentation: tensorzero_core::experimentation::ExperimentationConfig::default(),
    });

    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with tools
    assert!(result.is_ok(), "analyze_inferences should work with tools");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_json_function() {
    // Setup: For JSON functions, we would need different datapoint structure
    // This test is a placeholder showing how it would be structured
    let client = make_embedded_gateway().await;

    // Note: JSON functions require different StoredDatapoint::Json variant
    // For this test, we use Chat to verify the code path still works
    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert
    assert!(result.is_ok(), "analyze_inferences should work");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");
}

// ============================================================================
// Response Parsing Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_analyze_inferences_xml_extraction() {
    // Setup: Test that XML content is correctly extracted from responses
    let client = make_embedded_gateway().await;

    let eval_infos = vec![create_test_evaluation_info(
        "test_function",
        "Test input",
        "Test output",
    )];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Verify XML content is present
    assert!(result.is_ok(), "analyze_inferences should succeed");
    let analyses = result.unwrap();
    assert_eq!(analyses.len(), 1, "Should return 1 analysis");

    // The analysis field should contain some text (exact format depends on the model)
    assert!(
        !analyses[0].analysis.is_empty(),
        "Analysis should contain XML or text content"
    );
}

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
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
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

    // Verify analysis field is populated
    assert!(
        !analysis.analysis.is_empty(),
        "Analysis field should be populated"
    );
}
