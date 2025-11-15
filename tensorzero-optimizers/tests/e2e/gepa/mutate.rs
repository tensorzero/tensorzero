//! E2E tests for GEPA mutate function
//!
//! These tests exercise the mutate_templates component using real gateway clients
//! and the full inference pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used)]

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
    let config_and_tools = create_test_config_and_tools();
    let template_map = create_simple_template_map();
    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

    // Execute: Call mutate_templates
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should succeed but got error: {e}");
    });

    // Verify both template names are preserved
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        mutate_output.templates.contains_key("user"),
        "Should preserve 'user' template name"
    );

    // Verify templates have non-empty content
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content"
    );
    assert!(
        !mutate_output.templates["user"].is_empty(),
        "User template should have content"
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

    let config_and_tools = create_test_config_and_tools();
    let template_map = create_custom_template_map(vec![("system", "You are a test assistant.")]);
    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should still generate improved templates
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should succeed with single analysis but got error: {e}");
    });

    // Verify system template is preserved and has content
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content even with single analysis"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_empty_analyses() {
    // Setup: Test with empty analyses array
    let client = make_embedded_gateway().await;

    let analyses: Vec<InferenceWithAnalysis> = vec![];

    let config_and_tools = create_test_config_and_tools();
    let template_map = create_simple_template_map();
    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

    // Execute: Should handle gracefully
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed (mutate can work with empty analyses, using templates as baseline)
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should handle empty analyses gracefully but got error: {e}");
    });

    // Verify template names are preserved even with empty analyses
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content even with empty analyses"
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

    let config_and_tools = create_test_config_and_tools();
    let template_map = create_simple_template_map();
    let variant_config = create_test_variant_config_with_templates_inner(template_map);

    // Create config with invalid model
    let mut gepa_config = create_test_gepa_config();
    gepa_config.mutation_model = "invalid_provider::nonexistent_model".to_string();

    // Execute: Call with invalid model
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
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

    let config_and_tools = create_test_config_and_tools_with_schemas();
    let template_map = create_custom_template_map(vec![
        ("system", "You are a greeting assistant with {{greeting}}."),
        ("user", "Greet {{name}}"),
    ]);
    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

    // Execute: Should handle schemas correctly
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with schemas present
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should work with schemas but got error: {e}");
    });

    // Verify template names are preserved
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        mutate_output.templates.contains_key("user"),
        "Should preserve 'user' template name"
    );

    // Verify schema variables are preserved in mutated templates
    let system_template = &mutate_output.templates["system"];
    let user_template = &mutate_output.templates["user"];

    assert!(
        system_template.contains("{{greeting}}") || system_template.contains("greeting"),
        "System template should preserve {{greeting}} variable or reference 'greeting'"
    );
    assert!(
        user_template.contains("{{name}}") || user_template.contains("name"),
        "User template should preserve {{name}} variable or reference 'name'"
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

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();
    let gepa_config = create_test_gepa_config();

    // Step 2: Analyze inferences
    let analyses_result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
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
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Complete flow should work
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should succeed in end-to-end pipeline but got error: {e}");
    });

    // Verify template names from variant_config are preserved
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        mutate_output.templates.contains_key("user"),
        "Should preserve 'user' template name"
    );

    // Verify templates have non-empty content
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content"
    );
    assert!(
        !mutate_output.templates["user"].is_empty(),
        "User template should have content"
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

    let config_and_tools = create_test_config_and_tools();

    // Create variant with specific template names
    let template_map = create_custom_template_map(vec![
        ("system", "You are a helpful assistant."),
        ("user", "User: {{input}}"),
        ("custom", "Custom template content"),
    ]);

    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

    // Execute
    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should preserve template names
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should succeed but got error: {e}");
    });

    // Verify all template names are preserved
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        mutate_output.templates.contains_key("user"),
        "Should preserve 'user' template name"
    );
    assert!(
        mutate_output.templates.contains_key("custom"),
        "Should preserve 'custom' template name"
    );

    // Verify templates have non-empty content
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content"
    );
    assert!(
        !mutate_output.templates["user"].is_empty(),
        "User template should have content"
    );
    assert!(
        !mutate_output.templates["custom"].is_empty(),
        "Custom template should have content"
    );
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

    let config_and_tools = create_test_config_and_tools();

    // Use custom template names (not just system/user/assistant)
    let template_map = create_custom_template_map(vec![
        ("context", "Context template content"),
        ("instruction", "Instruction template content"),
        ("examples", "Examples template content"),
    ]);

    let variant_config = create_test_variant_config_with_templates_inner(template_map);
    let gepa_config = create_test_gepa_config();

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
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should work with new template format
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should work with new template format but got error: {e}");
    });

    // Verify all custom template names are preserved
    assert!(
        mutate_output.templates.contains_key("context"),
        "Should preserve 'context' template name"
    );
    assert!(
        mutate_output.templates.contains_key("instruction"),
        "Should preserve 'instruction' template name"
    );
    assert!(
        mutate_output.templates.contains_key("examples"),
        "Should preserve 'examples' template name"
    );

    // Verify templates have non-empty content
    assert!(
        !mutate_output.templates["context"].is_empty(),
        "Context template should have content"
    );
    assert!(
        !mutate_output.templates["instruction"].is_empty(),
        "Instruction template should have content"
    );
    assert!(
        !mutate_output.templates["examples"].is_empty(),
        "Examples template should have content"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_templates_with_inference_input_integration() {
    // Setup: Integration test for full analyze → mutate pipeline with inference_input
    let client = make_embedded_gateway().await;

    // Create evaluation infos
    let eval_infos = vec![
        create_test_evaluation_info("test_function", "What is 2+2?", "The answer is 4."),
        create_test_evaluation_info(
            "test_function",
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
    ];

    let config_and_tools = create_test_config_and_tools();
    let variant_config = create_test_variant_config();

    // Configure GEPA to include inference_input
    let mut gepa_config = create_test_gepa_config();
    gepa_config.include_inference_input_for_mutation = true;

    // Step 1: Analyze inferences with inference_input enabled
    let analyses_result = analyze_inferences(
        &client,
        &eval_infos,
        &config_and_tools,
        &variant_config,
        &gepa_config,
    )
    .await;

    assert!(
        analyses_result.is_ok(),
        "analyze_inferences should succeed with include_inference_input_for_mutation = true"
    );
    let analyses = analyses_result.unwrap();
    assert_eq!(analyses.len(), 2, "Should return 2 analyses");

    // Verify inference_input is populated
    assert!(
        analyses[0].inference_input.is_some(),
        "inference_input should be Some when include_inference_input_for_mutation is true"
    );
    assert!(
        analyses[1].inference_input.is_some(),
        "All analyses should have inference_input when flag is true"
    );

    // Verify that serialization includes inference_input field
    let serialized =
        serde_json::to_string(&analyses[0]).expect("Should serialize InferenceWithAnalysis");
    assert!(
        serialized.contains("inference_input"),
        "Serialized JSON should contain inference_input field when it's Some"
    );

    // Step 2: Call mutate_templates with analyses containing inference_input
    let template_map = create_simple_template_map();
    let variant_config_mutate = create_test_variant_config_with_templates_inner(template_map);

    let result = mutate_templates(
        &client,
        &analyses,
        &config_and_tools,
        &variant_config_mutate,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with inference_input
    let mutate_output = result.unwrap_or_else(|e| {
        panic!("mutate_templates should succeed with inference_input but got error: {e}");
    });

    // Verify templates are generated
    assert!(
        mutate_output.templates.contains_key("system"),
        "Should preserve 'system' template name"
    );
    assert!(
        mutate_output.templates.contains_key("user"),
        "Should preserve 'user' template name"
    );

    // Verify templates have non-empty content
    assert!(
        !mutate_output.templates["system"].is_empty(),
        "System template should have content"
    );
    assert!(
        !mutate_output.templates["user"].is_empty(),
        "User template should have content"
    );
}
