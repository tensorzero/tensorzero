use std::collections::HashMap;
use std::sync::Arc;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::evaluations::EvaluationConfig;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_optimizers::gepa::evaluate::{
    EvaluateVariantParams, VariantName, create_evaluation_dataset, evaluate_variant,
};
use tensorzero_optimizers::gepa::pareto::{Candidate, ParetoFrontier};
use tensorzero_optimizers::gepa::validate::get_uninitialized_variant_configs;
use tokio::time::{Duration, sleep};

use super::{
    TEST_CLICKHOUSE_WAIT_MS, build_gateway_client, cleanup_dataset, create_gepa_config_chat,
    create_gepa_config_json, create_test_chat_rendered_sample, create_test_json_rendered_sample,
    get_e2e_config, get_function_context,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_pareto_frontier_new() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_pareto_new".to_string();

    // Create dataset to get datapoint IDs
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
    ];
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Get evaluator configs
    let function_context = get_function_context(&gepa_config, &config);
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    // Initialize ParetoFrontier
    let pareto_frontier = ParetoFrontier::new(
        response.ids,
        evaluator_configs,
        gepa_config.seed.map(|s| s as u64),
    );

    // Verify frontier is empty initially
    let variant_configs = pareto_frontier.variant_configs();
    assert!(
        variant_configs.is_empty(),
        "New frontier should be empty before seeding"
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pareto_frontier_update_with_initial_variants() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_pareto_update_initial".to_string();

    // Create dataset and evaluate variants
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
    ];
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client and get variants
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get evaluator configs
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    // Evaluate all initial variants
    let num_variants = initial_variants.len();
    let per_variant_concurrency = (gepa_config.max_concurrency as usize / num_variants).max(1);
    let mut all_evaluation_results = Vec::new();

    for (variant_name, variant_config) in &initial_variants {
        let evaluation_params = EvaluateVariantParams {
            gateway_client: gateway_client.clone(),
            clickhouse_connection_info: clickhouse.clone(),
            functions: config.functions.clone(),
            evaluation_config: Arc::clone(&function_context.evaluation_config),
            evaluation_name: gepa_config.evaluation_name.clone(),
            variant_name: variant_name.to_string(),
            variant_config: variant_config.clone(),
            dataset_name: dataset_name.clone(),
            concurrency: per_variant_concurrency,
        };

        let evaluation_result = evaluate_variant(evaluation_params)
            .await
            .expect("Failed to evaluate variant");
        all_evaluation_results.push((variant_name.clone(), evaluation_result));
    }

    // Initialize Pareto frontier
    let mut pareto_frontier = ParetoFrontier::new(
        response.ids,
        evaluator_configs,
        gepa_config.seed.map(|s| s as u64),
    );

    // Seed the frontier with initial variants
    let mut initial_candidates: HashMap<VariantName, Candidate> = HashMap::new();
    for (variant_name, eval_results) in &all_evaluation_results {
        let scores = eval_results.per_datapoint_scores();
        if !scores.is_empty()
            && let Some(variant_config) = initial_variants.get(variant_name)
        {
            initial_candidates.insert(
                variant_name.clone(),
                Candidate {
                    variant: variant_config.clone(),
                    scores,
                },
            );
        }
    }

    assert_eq!(
        initial_candidates.len(),
        initial_variants.len(),
        "All initial variants should have scores before frontier update"
    );

    pareto_frontier
        .update(initial_candidates)
        .expect("Failed to seed Pareto frontier with initial candidates");

    // Verify frontier contains at least one variant
    let frontier_variants = pareto_frontier.variant_configs();
    assert!(
        !frontier_variants.is_empty(),
        "Pareto frontier should contain at least one variant after seeding"
    );
    assert!(
        frontier_variants.len() <= initial_variants.len(),
        "Pareto frontier should not have more variants than initial set"
    );

    // Verify all frontier variants are from the initial set
    for variant_name in frontier_variants.keys() {
        assert!(
            initial_variants.contains_key(variant_name),
            "Frontier variant '{variant_name}' should be from the initial variants"
        );
    }

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pareto_frontier_sample_by_frequency() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_json();
    let dataset_name = "test_pareto_sample".to_string();

    // Create dataset and evaluate variants
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_json_rendered_sample("input 1", r#"{"answer": "output 1"}"#),
        create_test_json_rendered_sample("input 2", r#"{"answer": "output 2"}"#),
    ];
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client and get variants
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get evaluator configs
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    // Evaluate all initial variants
    let num_variants = initial_variants.len();
    let per_variant_concurrency = (gepa_config.max_concurrency as usize / num_variants).max(1);
    let mut all_evaluation_results = Vec::new();

    for (variant_name, variant_config) in &initial_variants {
        let evaluation_params = EvaluateVariantParams {
            gateway_client: gateway_client.clone(),
            clickhouse_connection_info: clickhouse.clone(),
            functions: config.functions.clone(),
            evaluation_config: Arc::clone(&function_context.evaluation_config),
            evaluation_name: gepa_config.evaluation_name.clone(),
            variant_name: variant_name.to_string(),
            variant_config: variant_config.clone(),
            dataset_name: dataset_name.clone(),
            concurrency: per_variant_concurrency,
        };

        let evaluation_result = evaluate_variant(evaluation_params)
            .await
            .expect("Failed to evaluate variant");
        all_evaluation_results.push((variant_name.clone(), evaluation_result));
    }

    // Initialize and seed Pareto frontier
    let mut pareto_frontier = ParetoFrontier::new(
        response.ids,
        evaluator_configs,
        gepa_config.seed.map(|s| s as u64),
    );

    let mut initial_candidates: HashMap<VariantName, Candidate> = HashMap::new();
    for (variant_name, eval_results) in &all_evaluation_results {
        let scores = eval_results.per_datapoint_scores();
        if !scores.is_empty()
            && let Some(variant_config) = initial_variants.get(variant_name)
        {
            initial_candidates.insert(
                variant_name.clone(),
                Candidate {
                    variant: variant_config.clone(),
                    scores,
                },
            );
        }
    }

    pareto_frontier
        .update(initial_candidates)
        .expect("Failed to seed Pareto frontier");

    // Sample parent from frontier
    let parent = pareto_frontier
        .sample_by_frequency()
        .expect("Failed to sample parent from Pareto frontier");

    // Verify parent is one of the initial variants
    assert!(
        initial_variants.contains_key(&parent.name),
        "Sampled parent '{}' should be one of the initial variants",
        parent.name
    );

    // Verify we can sample multiple times
    for _ in 0..5 {
        let sampled = pareto_frontier
            .sample_by_frequency()
            .expect("Should be able to sample from frontier multiple times");
        assert!(
            initial_variants.contains_key(&sampled.name),
            "Each sampled variant should be from the initial variants"
        );
    }

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pareto_frontier_maintains_valid_state() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let gepa_config = create_gepa_config_chat();
    let dataset_name = "test_pareto_valid_state".to_string();

    // Create dataset and evaluate variants
    cleanup_dataset(&clickhouse, &dataset_name).await;
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
    ];
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create dataset");
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Build gateway client and get variants
    let gateway_client = build_gateway_client(
        config.clone(),
        clickhouse.clone(),
        http_client.clone(),
        gepa_config.timeout,
    )
    .await;
    let function_context = get_function_context(&gepa_config, &config);
    let initial_variants = get_uninitialized_variant_configs(&gepa_config, &function_context)
        .expect("Failed to get initial variants");

    // Get evaluator configs
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => &cfg.evaluators,
    };

    // Evaluate all initial variants
    let num_variants = initial_variants.len();
    let per_variant_concurrency = (gepa_config.max_concurrency as usize / num_variants).max(1);
    let mut all_evaluation_results = Vec::new();

    for (variant_name, variant_config) in &initial_variants {
        let evaluation_params = EvaluateVariantParams {
            gateway_client: gateway_client.clone(),
            clickhouse_connection_info: clickhouse.clone(),
            functions: config.functions.clone(),
            evaluation_config: Arc::clone(&function_context.evaluation_config),
            evaluation_name: gepa_config.evaluation_name.clone(),
            variant_name: variant_name.to_string(),
            variant_config: variant_config.clone(),
            dataset_name: dataset_name.clone(),
            concurrency: per_variant_concurrency,
        };

        let evaluation_result = evaluate_variant(evaluation_params)
            .await
            .expect("Failed to evaluate variant");
        all_evaluation_results.push((variant_name.clone(), evaluation_result));
    }

    // Initialize and seed Pareto frontier
    let mut pareto_frontier = ParetoFrontier::new(
        response.ids,
        evaluator_configs,
        gepa_config.seed.map(|s| s as u64),
    );

    let mut initial_candidates: HashMap<VariantName, Candidate> = HashMap::new();
    for (variant_name, eval_results) in &all_evaluation_results {
        let scores = eval_results.per_datapoint_scores();
        if !scores.is_empty()
            && let Some(variant_config) = initial_variants.get(variant_name)
        {
            initial_candidates.insert(
                variant_name.clone(),
                Candidate {
                    variant: variant_config.clone(),
                    scores,
                },
            );
        }
    }

    let frontier_size_before = pareto_frontier.variant_configs().len();
    assert_eq!(
        frontier_size_before, 0,
        "Frontier should be empty before update"
    );

    pareto_frontier
        .update(initial_candidates)
        .expect("Failed to seed Pareto frontier");

    let frontier_size_after = pareto_frontier.variant_configs().len();
    assert!(
        frontier_size_after > 0,
        "Frontier should have variants after seeding"
    );

    // Verify we can still sample after update
    let _sampled = pareto_frontier
        .sample_by_frequency()
        .expect("Should be able to sample from frontier after update");

    // Verify frontier maintains valid state after multiple samples
    for _ in 0..3 {
        let _ = pareto_frontier
            .sample_by_frequency()
            .expect("Frontier should remain valid after multiple samples");
    }

    // Verify variant_configs returns consistent results
    let configs1 = pareto_frontier.variant_configs();
    let configs2 = pareto_frontier.variant_configs();
    assert_eq!(
        configs1.len(),
        configs2.len(),
        "variant_configs should return consistent results"
    );

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}
