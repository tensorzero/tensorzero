#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::DynamicToolParams;
use tensorzero_core::client::{ClientBuilder, ClientBuilderMode};
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_dataset_clickhouse, select_json_dataset_clickhouse,
};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::endpoints::datasets::v1::delete_dataset;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedContentBlock,
    ResolvedRequestMessage, Role, StoredInput, StoredInputMessage, StoredInputMessageContent,
    System, Template, Text,
};
use tensorzero_core::optimization::gepa::GEPAConfig;
use tensorzero_core::stored_inference::{RenderedSample, StoredOutput};

use tensorzero_core::utils::retries::RetryConfig;
use tensorzero_optimizers::gepa::{
    analyze::analyze_inferences,
    evaluate::{
        create_evaluation_dataset, evaluate_variant, EvaluateVariantParams, EvaluationResults,
    },
    mutate::mutate_variant,
    validate::{initialize_pareto_frontier, validate_gepa_config},
};
use uuid::Uuid;

pub mod analyze;

/// Helper function to load the e2e test config
async fn get_e2e_config() -> Arc<Config> {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/config/tensorzero.*.toml");
    Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(&config_path)
                .expect("Failed to create config glob from path"),
            false,
        )
        .await
        .expect("Failed to load e2e config")
        .config,
    )
}

/// Helper function to create a test RenderedSample for Chat functions
fn create_test_chat_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let output_vec = vec![ContentBlockChatOutput::Text(Text {
        text: output.to_string(),
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: None,
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        output: Some(output_vec.clone()),
        stored_output: Some(StoredOutput::Chat(output_vec)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("test_key".to_string(), "test_value".to_string());
            tags
        },
    }
}

/// Helper function to create a test RenderedSample for JSON functions
fn create_test_json_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let json_output = JsonInferenceOutput {
        raw: Some(output.to_string()),
        parsed: Some(serde_json::json!({"answer": output})),
    };

    RenderedSample {
        function_name: "json_success".to_string(),
        input: ModelInput {
            system: Some("JSON system prompt".to_string()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter(vec![(
                        "country".to_string(),
                        json!(input),
                    )])),
                })],
            }],
        },
        output: None, // JSON functions don't have chat output
        stored_output: Some(StoredOutput::Json(json_output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        })),
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("json_key".to_string(), "json_value".to_string());
            tags
        },
    }
}

/// Check if analysis contains one of the expected XML tags
fn contains_expected_xml_tag(analysis: &str) -> bool {
    analysis.contains("<report_error>")
        || analysis.contains("<report_improvement>")
        || analysis.contains("<report_optimal>")
}

/// Helper function to verify evaluation results have expected structure and valid scores
fn assert_evaluation_results_valid(evaluation_results: &EvaluationResults, expected_count: usize) {
    let expected_evaluators = ["happy_bool", "sad_bool", "zero", "one"];

    // Verify we have stats for all 4 evaluators
    for evaluator_name in &expected_evaluators {
        assert!(
            evaluation_results
                .evaluation_stats
                .contains_key(*evaluator_name),
            "Expected {evaluator_name} evaluator stats"
        );
    }

    // Verify each evaluator has valid stats
    for (evaluator_name, stats) in &evaluation_results.evaluation_stats {
        assert_eq!(
            stats.count, expected_count,
            "Expected count of {} for {}, got {}",
            expected_count, evaluator_name, stats.count
        );
        assert!(
            stats.mean.is_finite(),
            "Expected mean to be finite for {}, got {}",
            evaluator_name,
            stats.mean
        );
    }

    // Test per_datapoint_scores extraction
    let per_datapoint_scores = evaluation_results.per_datapoint_scores();
    assert_eq!(
        per_datapoint_scores.len(),
        expected_count,
        "Expected {} datapoints in per_datapoint_scores, got {}",
        expected_count,
        per_datapoint_scores.len()
    );

    // Verify each datapoint has scores for all evaluators
    for (datapoint_id, scores) in &per_datapoint_scores {
        for evaluator_name in &expected_evaluators {
            assert!(
                scores.contains_key(*evaluator_name),
                "Datapoint {datapoint_id} missing {evaluator_name} score"
            );
        }

        // Verify scores are valid (either None or finite)
        for (evaluator_name, score_opt) in scores {
            if let Some(score) = score_opt {
                assert!(
                    score.is_finite(),
                    "Score for evaluator {evaluator_name} on datapoint {datapoint_id} is not finite: {score}"
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_gepa_step_chat() {
    let gepa_config = GEPAConfig {
        function_name: "basic_test".to_string(),
        evaluation_name: "test_evaluation".to_string(),
        initial_variants: Some(vec!["openai".to_string()]),
        variant_prefix: Some("gepa_test_chat".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 5,
        analysis_model: "openai::gpt-4.1-nano".to_string(),
        mutation_model: "openai::gpt-4.1-nano".to_string(),
        seed: None,
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    };

    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;

    let function_context = validate_gepa_config(&gepa_config, &config)
        .expect("validate_gepa_config should succeed in test");

    // Initialize baseline variants using the same function as the GEPA optimizer
    let initial_variants = initialize_pareto_frontier(&gepa_config, &function_context)
        .expect("initialize_pareto_frontier should succeed in test");

    // Get the first variant to use for testing
    let (internal_dynamic_variant_name, internal_dynamic_variant_config) = initial_variants
        .iter()
        .next()
        .expect("Should have at least one variant");

    // Generate unique dataset name to ensure test isolation
    let dataset_name = format!("test_eval_dataset_chat_{}", Uuid::now_v7());

    // Create test samples
    let samples = vec![
        create_test_chat_rendered_sample("test input 1", "test output 1"),
        create_test_chat_rendered_sample("test input 2", "test output 2"),
        create_test_chat_rendered_sample("test input 3", "test output 3"),
    ];

    // Call create_evaluation_dataset
    let result =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name).await;

    assert!(
        result.is_ok(),
        "Failed to create evaluation dataset: {:?}",
        result.err()
    );

    // Give ClickHouse a moment to process
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the datapoints were created in ClickHouse
    let datapoints = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();

    assert_eq!(
        datapoints.len(),
        3,
        "Expected 3 datapoints to be created, but found {}",
        datapoints.len()
    );

    // Verify the structure of the first datapoint
    let first_datapoint = &datapoints[0];
    assert_eq!(first_datapoint.dataset_name, dataset_name);
    assert_eq!(first_datapoint.function_name, "basic_test");
    assert!(!first_datapoint.is_deleted);
    assert!(first_datapoint.episode_id.is_some());

    // Verify tags are preserved
    assert!(first_datapoint.tags.is_some());
    let tags = first_datapoint.tags.as_ref().unwrap();
    assert_eq!(tags.get("test_key"), Some(&"test_value".to_string()));

    // Verify output is present
    assert!(first_datapoint.output.is_some());

    // Test evaluate_variant function
    let gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: config.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        http_client: http_client.clone(),
        timeout: Some(Duration::from_secs(gepa_config.timeout)),
    })
    .build()
    .await
    .expect("Failed to build gateway client");

    // Call evaluate_variant
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        tensorzero_config: config.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: internal_dynamic_variant_name.to_string(),
        variant_config: internal_dynamic_variant_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let evaluation_result = evaluate_variant(evaluation_params).await;
    assert!(
        evaluation_result.is_ok(),
        "Failed to evaluate variant: {:?}",
        evaluation_result.err()
    );

    let evaluation_results = evaluation_result.unwrap();

    // Assert evaluation results
    assert_eq!(
        evaluation_results.evaluation_infos.len(),
        3,
        "Expected 3 evaluation results, got {}",
        evaluation_results.evaluation_infos.len()
    );

    // Verify evaluation results have expected structure and valid scores
    assert_evaluation_results_valid(&evaluation_results, 3);

    // Delete the dataset
    let delete_result = delete_dataset(&clickhouse, &dataset_name).await;
    assert!(
        delete_result.is_ok(),
        "Failed to delete dataset: {:?}",
        delete_result.err()
    );
    assert_eq!(delete_result.unwrap().num_deleted_datapoints, 3);

    // Give ClickHouse a moment to process the deletion
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the dataset is empty after deletion
    let datapoints_after_delete = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(
        datapoints_after_delete.is_empty(),
        "Expected dataset to be empty after deletion, but found {} datapoints",
        datapoints_after_delete.len()
    );

    // Run Analyses
    let analysis_result = analyze_inferences(
        &gateway_client,
        &evaluation_results.evaluation_infos,
        &function_context,
        internal_dynamic_variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with CHAT function
    assert!(
        analysis_result.is_ok(),
        "analyze_inferences should work with CHAT functions"
    );
    let analyses = analysis_result.unwrap();
    assert_eq!(analyses.len(), 3, "Should return 3 analyses");

    // Verify each analysis has content and expected XML tags
    for analysis in &analyses {
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

        assert!(
            !analysis.analysis.is_empty(),
            "Each analysis should have non-empty XML content"
        );
        assert!(
            contains_expected_xml_tag(&analysis.analysis),
            "Analysis should contain one of: <report_error>, <report_improvement>, or <report_optimal>"
        );
    }

    // Test mutate_variant function
    let parent_name = internal_dynamic_variant_name.to_string();
    let mut parent = HashMap::new();
    parent.insert(&parent_name, internal_dynamic_variant_config);

    let mutate_result = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        parent,
        &gepa_config,
        0, // iteration
    )
    .await;

    // Assert mutate_variant succeeded
    assert!(
        mutate_result.is_ok(),
        "mutate_variant should succeed: {:?}",
        mutate_result.err()
    );

    let child_variants = mutate_result.unwrap();

    // Validate returned HashMap has exactly 1 entry
    assert_eq!(
        child_variants.len(),
        1,
        "Expected exactly 1 child variant, got {}",
        child_variants.len()
    );

    // Get the child variant
    let (child_name, child_config) = child_variants.iter().next().unwrap();

    // Verify child variant name format uses the configured prefix
    let expected_prefix = format!(
        "{}-iter-0-",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa")
    );
    assert!(
        child_name.starts_with(&expected_prefix),
        "Child variant name '{child_name}' should start with '{expected_prefix}'"
    );

    // Verify templates are non-empty
    assert!(
        !child_config.templates.inner.is_empty(),
        "Child variant should have non-empty templates"
    );

    // Verify child has the same template keys as parent
    let parent_template_keys: std::collections::HashSet<_> = internal_dynamic_variant_config
        .templates
        .inner
        .keys()
        .collect();
    let child_template_keys: std::collections::HashSet<_> =
        child_config.templates.inner.keys().collect();
    assert_eq!(
        parent_template_keys,
        child_template_keys,
        "Child variant should have the same template keys as parent. Parent keys: {parent_template_keys:?}, Child keys: {child_template_keys:?}"
    );

    // Verify all templates have non-empty content
    for (template_name, template_config) in &child_config.templates.inner {
        let content = template_config.path.data();
        assert!(
            !content.is_empty(),
            "Template '{template_name}' should have non-empty content"
        );
    }

    // Verify system template contains the expected variable
    let system_template = child_config
        .templates
        .inner
        .get("system")
        .expect("Child should have system template");
    assert!(
        system_template.path.data().contains("{{ assistant_name }}"),
        "Child system template should preserve {{ assistant_name }} variable"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_gepa_step_json() {
    let gepa_config = GEPAConfig {
        function_name: "json_success".to_string(),
        evaluation_name: "json_evaluation".to_string(),
        initial_variants: Some(vec!["openai".to_string()]),
        variant_prefix: Some("gepa_test_json".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 5,
        analysis_model: "openai::gpt-4.1-nano".to_string(),
        mutation_model: "openai::gpt-4.1-nano".to_string(),
        seed: None,
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    };

    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;

    let function_context = validate_gepa_config(&gepa_config, &config)
        .expect("validate_gepa_config should succeed in test");

    // Initialize baseline variants using the same function as the GEPA optimizer
    let initial_variants = initialize_pareto_frontier(&gepa_config, &function_context)
        .expect("initialize_pareto_frontier should succeed in test");

    // Get the first variant to use for testing
    let (internal_dynamic_variant_name, internal_dynamic_variant_config) = initial_variants
        .iter()
        .next()
        .expect("Should have at least one variant");

    // Generate unique dataset name to ensure test isolation
    let dataset_name = format!("test_eval_dataset_json_{}", Uuid::now_v7());

    // Create test samples for JSON function
    let samples = vec![
        create_test_json_rendered_sample("input 1", r#"{"answer": "output 1"}"#),
        create_test_json_rendered_sample("input 2", r#"{"answer": "output 2"}"#),
    ];

    // Call create_evaluation_dataset
    let result =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name).await;

    assert!(
        result.is_ok(),
        "Failed to create evaluation dataset: {:?}",
        result.err()
    );

    // Give ClickHouse a moment to process
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the datapoints were created in ClickHouse
    let datapoints = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();

    assert_eq!(
        datapoints.len(),
        2,
        "Expected 2 datapoints to be created, but found {}",
        datapoints.len()
    );

    // Verify the structure of the first datapoint
    let first_datapoint = &datapoints[0];
    assert_eq!(first_datapoint.dataset_name, dataset_name);
    assert_eq!(first_datapoint.function_name, "json_success");
    assert!(!first_datapoint.is_deleted);
    assert!(first_datapoint.episode_id.is_some());

    // Verify tags are preserved
    assert!(first_datapoint.tags.is_some());
    let tags = first_datapoint.tags.as_ref().unwrap();
    assert_eq!(tags.get("json_key"), Some(&"json_value".to_string()));

    // Verify output is present and structured correctly
    assert!(first_datapoint.output.is_some());
    let output = first_datapoint.output.as_ref().unwrap();
    assert!(output.raw.is_some());
    assert!(output.parsed.is_some());

    // Verify output_schema is preserved
    assert!(first_datapoint.output_schema.get("type").is_some());
    assert_eq!(first_datapoint.output_schema.get("type").unwrap(), "object");

    // Test evaluate_variant function
    let gateway_client = ClientBuilder::new(ClientBuilderMode::FromComponents {
        config: config.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        http_client: http_client.clone(),
        timeout: Some(Duration::from_secs(gepa_config.timeout)),
    })
    .build()
    .await
    .expect("Failed to build gateway client");

    // Call evaluate_variant
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        tensorzero_config: config.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: internal_dynamic_variant_name.to_string(),
        variant_config: internal_dynamic_variant_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let evaluation_result = evaluate_variant(evaluation_params).await;
    assert!(
        evaluation_result.is_ok(),
        "Failed to evaluate variant: {:?}",
        evaluation_result.err()
    );

    let evaluation_results = evaluation_result.unwrap();

    // Assert evaluation results
    assert_eq!(
        evaluation_results.evaluation_infos.len(),
        2,
        "Expected 2 evaluation results, got {}",
        evaluation_results.evaluation_infos.len()
    );

    // Verify evaluation results have expected structure and valid scores
    assert_evaluation_results_valid(&evaluation_results, 2);

    // Delete the dataset
    let delete_result = delete_dataset(&clickhouse, &dataset_name).await;
    assert!(
        delete_result.is_ok(),
        "Failed to delete dataset: {:?}",
        delete_result.err()
    );
    assert_eq!(delete_result.unwrap().num_deleted_datapoints, 2);

    // Give ClickHouse a moment to process the deletion
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the dataset is empty after deletion
    let datapoints_after_delete = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(
        datapoints_after_delete.is_empty(),
        "Expected dataset to be empty after deletion, but found {} datapoints",
        datapoints_after_delete.len()
    );

    // Run Analyses
    let analysis_result = analyze_inferences(
        &gateway_client,
        &evaluation_results.evaluation_infos,
        &function_context,
        internal_dynamic_variant_config,
        &gepa_config,
    )
    .await;

    // Assert: Should succeed with JSON function
    assert!(
        analysis_result.is_ok(),
        "analyze_inferences should work with JSON functions"
    );
    let analyses = analysis_result.unwrap();
    assert_eq!(analyses.len(), 2, "Should return 2 analyses");

    // Verify each analysis has content and expected XML tags
    for analysis in &analyses {
        // Verify the inference is Some and contains Json output
        let inference = analysis
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
            parsed.get("answer").is_some(),
            "Should have 'answer' field in parsed JSON output"
        );

        assert!(
            !analysis.analysis.is_empty(),
            "Each analysis should have non-empty XML content"
        );
        assert!(
            contains_expected_xml_tag(&analysis.analysis),
            "Analysis should contain one of: <report_error>, <report_improvement>, or <report_optimal>"
        );
    }

    // Test mutate_variant function
    let parent_name = internal_dynamic_variant_name.to_string();
    let mut parent = HashMap::new();
    parent.insert(&parent_name, internal_dynamic_variant_config);

    let mutate_result = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        parent,
        &gepa_config,
        0, // iteration
    )
    .await;

    // Assert mutate_variant succeeded
    assert!(
        mutate_result.is_ok(),
        "mutate_variant should succeed: {:?}",
        mutate_result.err()
    );

    let child_variants = mutate_result.unwrap();

    // Validate returned HashMap has exactly 1 entry
    assert_eq!(
        child_variants.len(),
        1,
        "Expected exactly 1 child variant, got {}",
        child_variants.len()
    );

    // Get the child variant
    let (child_name, child_config) = child_variants.iter().next().unwrap();

    // Verify child variant name format uses the configured prefix
    let expected_prefix = format!(
        "{}-iter-0-",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa")
    );
    assert!(
        child_name.starts_with(&expected_prefix),
        "Child variant name '{child_name}' should start with '{expected_prefix}'"
    );

    // Verify templates are non-empty
    assert!(
        !child_config.templates.inner.is_empty(),
        "Child variant should have non-empty templates"
    );

    // Verify child has the same template keys as parent
    let parent_template_keys: std::collections::HashSet<_> = internal_dynamic_variant_config
        .templates
        .inner
        .keys()
        .collect();
    let child_template_keys: std::collections::HashSet<_> =
        child_config.templates.inner.keys().collect();
    assert_eq!(
        parent_template_keys,
        child_template_keys,
        "Child variant should have the same template keys as parent. Parent keys: {parent_template_keys:?}, Child keys: {child_template_keys:?}"
    );

    // Verify all templates have non-empty content
    for (template_name, template_config) in &child_config.templates.inner {
        let content = template_config.path.data();
        assert!(
            !content.is_empty(),
            "Template '{template_name}' should have non-empty content"
        );
    }

    // Verify system template contains the expected variable
    let system_template = child_config
        .templates
        .inner
        .get("system")
        .expect("Child should have system template");
    let system_content = system_template.path.data();
    assert!(
        system_content.contains("{{ assistant_name }}"),
        "Child system template should preserve {{ assistant_name }} variable"
    );
    assert!(
        system_content.contains(r#""answer":"#) || system_content.contains("'answer':"),
        "Child system template should reference output schema field 'answer'"
    );

    // Verify user template contains the expected variable
    let user_template = child_config
        .templates
        .inner
        .get("user")
        .expect("Child should have user template");
    assert!(
        user_template.path.data().contains("{{ country }}"),
        "Child user template should preserve {{ country }} variable"
    );
}
