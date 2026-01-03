use std::sync::Arc;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_optimizers::gepa::evaluate::{
    EvaluateVariantParams, create_evaluation_dataset, evaluate_variant,
};
use tensorzero_optimizers::gepa::validate::get_uninitialized_variant_configs;
use tokio::time::{Duration, sleep};

use super::{
    TEST_CLICKHOUSE_WAIT_MS, assert_evaluation_results_valid, build_gateway_client,
    cleanup_dataset, create_gepa_config_chat, create_gepa_config_json,
    create_test_chat_rendered_sample, create_test_json_rendered_sample, get_e2e_config,
    get_function_context,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_evaluate_variant_chat() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_evaluate_variant_chat".to_string();

    // Clean up and create dataset
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

    // Get first variant to evaluate
    let (variant_name, variant_config) =
        initial_variants.iter().next().expect("Should have variant");

    // Evaluate variant
    let evaluation_params = EvaluateVariantParams {
        gateway_client,
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: variant_name.to_string(),
        variant_config: variant_config.clone(),
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
    assert_eq!(
        evaluation_results.evaluation_infos.len(),
        2,
        "Expected 2 evaluation results"
    );

    // Verify evaluation results structure
    assert_evaluation_results_valid(&evaluation_results, 2);

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_evaluate_variant_json() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_json();
    let dataset_name = "test_evaluate_variant_json".to_string();

    // Clean up and create dataset
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_json_rendered_sample("input 1", r#"{"answer": "output 1"}"#),
        create_test_json_rendered_sample("input 2", r#"{"answer": "output 2"}"#),
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

    // Get first variant to evaluate
    let (variant_name, variant_config) =
        initial_variants.iter().next().expect("Should have variant");

    // Evaluate variant
    let evaluation_params = EvaluateVariantParams {
        gateway_client,
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: variant_name.to_string(),
        variant_config: variant_config.clone(),
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
    assert_eq!(
        evaluation_results.evaluation_infos.len(),
        2,
        "Expected 2 evaluation results"
    );

    // Verify evaluation results structure
    assert_evaluation_results_valid(&evaluation_results, 2);

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_evaluate_variant_per_datapoint_scores() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_per_datapoint_scores".to_string();

    // Clean up and create dataset with 3 samples
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
        create_test_chat_rendered_sample("input 3", "output 3"),
    ];
    let create_response =
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

    // Get first variant to evaluate
    let (variant_name, variant_config) =
        initial_variants.iter().next().expect("Should have variant");

    // Evaluate variant
    let evaluation_params = EvaluateVariantParams {
        gateway_client,
        clickhouse_connection_info: clickhouse.clone(),
        functions: config.functions.clone(),
        evaluation_config: Arc::clone(&function_context.evaluation_config),
        evaluation_name: gepa_config.evaluation_name.clone(),
        variant_name: variant_name.to_string(),
        variant_config: variant_config.clone(),
        dataset_name: dataset_name.clone(),
        concurrency: gepa_config.max_concurrency as usize,
    };

    let evaluation_result = evaluate_variant(evaluation_params)
        .await
        .expect("Failed to evaluate variant");

    // Test per_datapoint_scores extraction
    let per_datapoint_scores = evaluation_result.per_datapoint_scores();
    assert_eq!(
        per_datapoint_scores.len(),
        3,
        "Expected 3 datapoints in per_datapoint_scores"
    );

    // Verify each returned datapoint ID exists in the per_datapoint_scores
    for datapoint_id in &create_response.ids {
        assert!(
            per_datapoint_scores.contains_key(datapoint_id),
            "Datapoint ID {datapoint_id} missing from per_datapoint_scores"
        );
    }

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}
