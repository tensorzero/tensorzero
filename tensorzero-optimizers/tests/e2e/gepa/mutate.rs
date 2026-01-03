use std::sync::Arc;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_optimizers::gepa::GEPAVariant;
use tensorzero_optimizers::gepa::analyze::analyze_inferences;
use tensorzero_optimizers::gepa::evaluate::{
    EvaluateVariantParams, create_evaluation_dataset, evaluate_variant,
};
use tensorzero_optimizers::gepa::mutate::mutate_variant;
use tensorzero_optimizers::gepa::validate::get_uninitialized_variant_configs;
use tokio::time::{Duration, sleep};

use super::{
    TEST_CLICKHOUSE_WAIT_MS, build_gateway_client, cleanup_dataset, contains_expected_xml_tag,
    create_gepa_config_chat, create_gepa_config_json, create_test_chat_rendered_sample,
    create_test_json_rendered_sample, get_e2e_config, get_function_context,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_variant_chat() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_mutate_chat".to_string();

    // Create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
    ];
    let _response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;

    // Get function context and variants
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get first variant as parent
    let (parent_name, parent_config) = initial_variants.iter().next().expect("Should have variant");

    // Evaluate parent
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent_name.to_string(),
        variant_config: parent_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let parent_results = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate parent");

    // Analyze parent results
    let analyses = analyze_inferences(
        &gateway_client,
        &parent_results.evaluation_infos,
        &function_context,
        parent_config,
        &gepa_config,
    )
    .await
    .expect("Failed to analyze inferences");

    assert_eq!(analyses.len(), 2, "Should return 2 analyses");

    // Verify analyses have expected structure
    for analysis in &analyses {
        assert!(
            !analysis.analysis.is_empty(),
            "Each analysis should have non-empty content"
        );
        assert!(
            contains_expected_xml_tag(&analysis.analysis),
            "Analysis should contain expected XML tags"
        );
    }

    // Create parent structure for mutation
    let parent = GEPAVariant {
        name: parent_name.clone(),
        config: parent_config.clone(),
    };

    // Mutate variant
    let mutate_result = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        0, // iteration
    )
    .await;

    assert!(
        mutate_result.is_ok(),
        "mutate_variant should succeed: {:?}",
        mutate_result.err()
    );

    let child = mutate_result.unwrap();

    // Verify child variant name format
    let expected_prefix = format!(
        "{}-iter-0-",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa")
    );
    assert!(
        child.name.starts_with(&expected_prefix),
        "Child variant name '{}' should start with '{}'",
        child.name,
        expected_prefix
    );

    // Verify templates are non-empty
    assert!(
        !child.config.templates.inner.is_empty(),
        "Child variant should have non-empty templates"
    );

    // Verify all templates have non-empty content
    for (template_name, template_config) in &child.config.templates.inner {
        let content = template_config.path.data();
        assert!(
            !content.is_empty(),
            "Template '{template_name}' should have non-empty content"
        );
    }

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_variant_json() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_json();
    let dataset_name = "test_mutate_json".to_string();

    // Create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![create_test_json_rendered_sample(
        "input 1",
        r#"{"answer": "output 1"}"#,
    )];
    let _response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;

    // Get function context and variants
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get first variant as parent
    let (parent_name, parent_config) = initial_variants.iter().next().expect("Should have variant");

    // Evaluate parent
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent_name.to_string(),
        variant_config: parent_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let parent_results = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate parent");

    // Analyze parent results
    let analyses = analyze_inferences(
        &gateway_client,
        &parent_results.evaluation_infos,
        &function_context,
        parent_config,
        &gepa_config,
    )
    .await
    .expect("Failed to analyze inferences");

    // Create parent structure for mutation
    let parent = GEPAVariant {
        name: parent_name.clone(),
        config: parent_config.clone(),
    };

    // Mutate variant
    let mutate_result = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        0, // iteration
    )
    .await;

    assert!(
        mutate_result.is_ok(),
        "mutate_variant should succeed: {:?}",
        mutate_result.err()
    );

    let child = mutate_result.unwrap();

    // Verify child variant name format
    let expected_prefix = format!(
        "{}-iter-0-",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa")
    );
    assert!(
        child.name.starts_with(&expected_prefix),
        "Child variant name '{}' should start with '{}'",
        child.name,
        expected_prefix
    );

    // Verify templates are non-empty
    assert!(
        !child.config.templates.inner.is_empty(),
        "Child variant should have non-empty templates"
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_variant_preserves_variables() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_mutate_preserves_vars".to_string();

    // Create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![create_test_chat_rendered_sample("input 1", "output 1")];
    let _response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;

    // Get function context and variants
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get first variant as parent
    let (parent_name, parent_config) = initial_variants.iter().next().expect("Should have variant");

    // Evaluate and analyze
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent_name.to_string(),
        variant_config: parent_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let parent_results = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate parent");

    let analyses = analyze_inferences(
        &gateway_client,
        &parent_results.evaluation_infos,
        &function_context,
        parent_config,
        &gepa_config,
    )
    .await
    .expect("Failed to analyze inferences");

    // Create parent and mutate
    let parent = GEPAVariant {
        name: parent_name.clone(),
        config: parent_config.clone(),
    };

    let child = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        0,
    )
    .await
    .expect("Failed to mutate variant");

    // Verify system template contains the expected variable
    let system_template = child
        .config
        .templates
        .inner
        .get("system")
        .expect("Child should have system template");
    assert!(
        system_template.path.data().contains("{{ assistant_name }}"),
        "Child system template should preserve {{ assistant_name }} variable"
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_variant_preserves_schema_references() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_json();
    let dataset_name = "test_mutate_schema_refs".to_string();

    // Create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![create_test_json_rendered_sample(
        "input 1",
        r#"{"answer": "output 1"}"#,
    )];
    let _response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;

    // Get function context and variants
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get first variant as parent
    let (parent_name, parent_config) = initial_variants.iter().next().expect("Should have variant");

    // Evaluate and analyze
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent_name.to_string(),
        variant_config: parent_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let parent_results = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate parent");

    let analyses = analyze_inferences(
        &gateway_client,
        &parent_results.evaluation_infos,
        &function_context,
        parent_config,
        &gepa_config,
    )
    .await
    .expect("Failed to analyze inferences");

    // Create parent and mutate
    let parent = GEPAVariant {
        name: parent_name.clone(),
        config: parent_config.clone(),
    };

    let child = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        0,
    )
    .await
    .expect("Failed to mutate variant");

    // Verify system template contains schema field references
    let system_template = child
        .config
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
    let user_template = child
        .config
        .templates
        .inner
        .get("user")
        .expect("Child should have user template");
    assert!(
        user_template.path.data().contains("{{ country }}"),
        "Child user template should preserve {{ country }} variable"
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mutate_variant_naming() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;

    // Test with custom prefix
    let mut gepa_config = create_gepa_config_chat();
    gepa_config.variant_prefix = Some("custom_prefix".to_string());

    let dataset_name = "test_mutate_naming".to_string();

    // Create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![create_test_chat_rendered_sample("input 1", "output 1")];
    let _response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;

    // Get function context and variants
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get first variant as parent
    let (parent_name, parent_config) = initial_variants.iter().next().expect("Should have variant");

    // Evaluate and analyze
    let evaluation_params = EvaluateVariantParams {
        gateway_client: gateway_client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: parent_name.to_string(),
        variant_config: parent_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let parent_results = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate parent");

    let analyses = analyze_inferences(
        &gateway_client,
        &parent_results.evaluation_infos,
        &function_context,
        parent_config,
        &gepa_config,
    )
    .await
    .expect("Failed to analyze inferences");

    // Create parent and mutate with iteration 2
    let parent = GEPAVariant {
        name: parent_name.clone(),
        config: parent_config.clone(),
    };

    let child = mutate_variant(
        &gateway_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        2, // iteration
    )
    .await
    .expect("Failed to mutate variant");

    // Verify child variant name format includes custom prefix and iteration
    assert!(
        child.name.starts_with("custom_prefix-iter-2-"),
        "Child variant name '{}' should start with 'custom_prefix-iter-2-'",
        child.name
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}
