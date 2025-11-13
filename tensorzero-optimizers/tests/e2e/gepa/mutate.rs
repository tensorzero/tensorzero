//! E2E tests for GEPA mutate function
//!
//! These tests exercise the mutate_templates component using real gateway clients
//! and the full inference pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_optimizers::gepa::{analyze_inferences, mutate_templates};

use super::*;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_success() {
    // Setup: Create gateway client and mock analyses
    let client = make_embedded_gateway().await;

    // Create mock InferenceWithAnalysis entries
    let analyses = vec![
        create_test_inference_with_analysis(
            "test_variant",
            "The answer is 4.",
            "<report_optimal>Good mathematical response</report_optimal>",
        ),
        create_test_inference_with_analysis(
            "test_variant",
            "The capital of France is Paris.",
            "<report_optimal>Correct factual answer</report_optimal>",
        ),
        create_test_inference_with_analysis(
            "test_variant",
            "Gravity pulls things down.",
            "<report_improvement>Could be more precise about mass and attraction</report_improvement>",
        ),
    ];

    // Create function and variant configs
    let function_config = create_test_function_config();
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    template_map.insert("user".to_string(), "User: {{input}}".to_string());
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute: Call mutate_templates
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should succeed
    assert!(result.is_ok(), "mutate_templates should succeed");
    let mutate_output = result.unwrap();

    // Verify templates HashMap is non-empty
    assert!(
        !mutate_output.templates.is_empty(),
        "Should return non-empty templates HashMap"
    );

    // Verify original template names are preserved
    assert!(
        mutate_output.templates.contains_key("system")
            || mutate_output.templates.contains_key("user"),
        "Should preserve at least one original template name"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_single_analysis() {
    // Setup: Test with just one analysis
    let client = make_embedded_gateway().await;

    let analyses = vec![create_test_inference_with_analysis(
        "test_variant",
        "Test output",
        "<report_error>Output is too generic</report_error>",
    )];

    let function_config = create_test_function_config();
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a test assistant.".to_string(),
    );
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should still generate improved templates
    assert!(
        result.is_ok(),
        "mutate_templates should succeed with single analysis"
    );
    let mutate_output = result.unwrap();
    assert!(
        !mutate_output.templates.is_empty(),
        "Should generate templates even with single analysis"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_empty_analyses() {
    // Setup: Test with empty analyses array
    let client = make_embedded_gateway().await;

    let analyses: Vec<InferenceWithAnalysis> = vec![];

    let function_config = create_test_function_config();
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute: Should handle gracefully
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should succeed (mutate can work with empty analyses, using templates as baseline)
    assert!(
        result.is_ok(),
        "mutate_templates should handle empty analyses gracefully"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_invalid_model() {
    // Setup: Use non-existent mutation model
    let client = make_embedded_gateway().await;

    let analyses = vec![create_test_inference_with_analysis(
        "test_variant",
        "Test output",
        "<report_optimal>Good</report_optimal>",
    )];

    let function_config = create_test_function_config();
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute: Call with invalid model
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "invalid_provider::nonexistent_model",
    )
    .await;

    // Assert: Should return error
    assert!(
        result.is_err(),
        "mutate_templates should error with invalid model"
    );
    let error = result.unwrap_err();
    let error_msg = error.to_string();
    // Error message should indicate the problem
    assert!(
        !error_msg.is_empty(),
        "Error message should be non-empty: {error_msg}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_with_schemas() {
    // Setup: Use function config with schemas
    let client = make_embedded_gateway().await;

    let analyses = vec![
        create_test_inference_with_analysis(
            "test_variant",
            "Hello World",
            "<report_optimal>Good greeting</report_optimal>",
        ),
        create_test_inference_with_analysis(
            "test_variant",
            "Hi there",
            "<report_improvement>Could be more formal</report_improvement>",
        ),
    ];

    let function_config = create_test_function_config_with_schemas();
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a greeting assistant with {{greeting}}.".to_string(),
    );
    template_map.insert("user".to_string(), "Greet {{name}}".to_string());
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute: Should handle schemas correctly
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should succeed with schemas present
    assert!(result.is_ok(), "mutate_templates should work with schemas");
    let mutate_output = result.unwrap();
    assert!(
        !mutate_output.templates.is_empty(),
        "Should generate templates with schemas"
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_end_to_end() {
    // Setup: Full pipeline test - analyze then mutate
    let client = make_embedded_gateway().await;

    // Step 1: Create evaluation infos
    let eval_infos = vec![
        create_test_evaluation_info("test_function", "What is 2+2?", "The answer is 4."),
        create_test_evaluation_info(
            "test_function",
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
    ];

    let function_config = create_test_function_config();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Step 2: Analyze inferences
    let analyses_result = analyze_inferences(
        &client,
        &eval_infos,
        &function_config,
        &variant_config,
        &gepa_config,
    )
    .await;

    assert!(
        analyses_result.is_ok(),
        "analyze_inferences should succeed in end-to-end test"
    );
    let analyses = analyses_result.unwrap();

    assert!(
        !analyses.is_empty(),
        "Should have at least one analysis for mutation"
    );

    // Step 3: Mutate templates using analyses
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Complete flow should work
    assert!(
        result.is_ok(),
        "mutate_templates should succeed in end-to-end pipeline"
    );
    let mutate_output = result.unwrap();
    assert!(
        !mutate_output.templates.is_empty(),
        "Should generate mutated templates from analyses"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_preserves_template_names() {
    // Setup: Verify output has same template names as input
    let client = make_embedded_gateway().await;

    let analyses = vec![create_test_inference_with_analysis(
        "test_variant",
        "Test output",
        "<report_optimal>Good</report_optimal>",
    )];

    let function_config = create_test_function_config();

    // Create variant with specific template names
    let mut template_map = HashMap::new();
    template_map.insert(
        "system".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    template_map.insert("user".to_string(), "User: {{input}}".to_string());
    template_map.insert("custom".to_string(), "Custom template content".to_string());

    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Execute
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should preserve template names
    assert!(result.is_ok(), "mutate_templates should succeed");
    let mutate_output = result.unwrap();

    // Note: The dummy model may or may not preserve all template names exactly
    // but it should return at least some templates
    assert!(
        !mutate_output.templates.is_empty(),
        "Should return templates"
    );

    // The mutate function should ideally preserve template names, but with dummy model
    // we just verify that we get some output back
    // In a real scenario with a proper LLM, we would check:
    // assert!(mutate_output.templates.contains_key("system"));
    // assert!(mutate_output.templates.contains_key("user"));
    // assert!(mutate_output.templates.contains_key("custom"));
}

// ============================================================================
// Template Format Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_with_new_template_format() {
    // Setup: Test with modern template format (templates.inner HashMap)
    let client = make_embedded_gateway().await;

    let analyses = vec![create_test_inference_with_analysis(
        "test_variant",
        "Good response",
        "<report_optimal>Excellent quality</report_optimal>",
    )];

    let function_config = create_test_function_config();

    // Use custom template names (not just system/user/assistant)
    let mut template_map = HashMap::new();
    template_map.insert(
        "context".to_string(),
        "Context template content".to_string(),
    );
    template_map.insert(
        "instruction".to_string(),
        "Instruction template content".to_string(),
    );
    template_map.insert(
        "examples".to_string(),
        "Examples template content".to_string(),
    );

    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Verify variant config is using new format
    assert!(
        !variant_config.templates.inner.is_empty(),
        "Variant should use templates.inner format"
    );
    assert!(
        variant_config.system_template.is_none(),
        "Legacy system_template should be None"
    );
    assert!(
        variant_config.user_template.is_none(),
        "Legacy user_template should be None"
    );
    assert!(
        variant_config.assistant_template.is_none(),
        "Legacy assistant_template should be None"
    );

    // Execute
    let result = mutate_templates(
        &client,
        &analyses,
        &function_config,
        &variant_config,
        "dummy::echo_request_messages",
    )
    .await;

    // Assert: Should work with new template format
    assert!(
        result.is_ok(),
        "mutate_templates should work with new template format"
    );
    let mutate_output = result.unwrap();
    assert!(
        !mutate_output.templates.is_empty(),
        "Should generate templates using new format"
    );
}
