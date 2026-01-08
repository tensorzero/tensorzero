#![allow(clippy::expect_used, clippy::unwrap_used)]
// ============================================================================
// Top-K Evaluation Tests
// ============================================================================
//
// Integration tests for the top-k variant selection algorithm using confidence
// sequences. These tests verify the durable task implementation that identifies
// the best-performing variants from a set of candidates.
//
// There are four possible stopping conditions for the algorithm. These tests
// check all four conditions.
// - test_topk_found_topk: Verifies correct identification of the winning variant
//   (`TopKFound` stopping condition) when k = 1.
// - test_topk_dataset_exhaustion: Verifies `DatasetExhausted` stopping condition,
//   when there is insufficient data to identify a top-k set of variants.
// - test_topk_evaluator_failure_threshold: Verifies `EvaluatorsFailed` stopping
//   condition, when the confidence sequence for any evaluator's failure rate lies
//   above a user-chosen threshold.
// - test_topk_variant_failure_threshold: Verifies `TooManyVariantsFailed` stopping
//   condition, when enough variants have failed that there are fewer than k variants
//   remaining. Variant failure occurs when the confidence sequence for a variant's
//   failure rate lies above a user-chosen threshold.
//
// ============================================================================

mod common;

use common::{get_config, get_tensorzero_client, init_tracing_for_tests};
use durable::WorkerOptions;
use evaluations::topk::{
    GlobalStoppingReason, ScoringFunctionType, TopKTaskOutput, TopKTaskParams, TopKTaskState,
    TopKUpdate, VariantStatus, create_client,
};
use evaluations::{
    ClientInferenceExecutor, Clients, EvaluationFunctionConfig, EvaluationFunctionConfigTable,
};
use serde_json::Map;
use sqlx::{AssertSqlSafe, PgPool, query_as};
use std::sync::Arc;
use std::time::Duration;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::clickhouse::TableName;
use tensorzero_core::db::clickhouse::test_helpers::{
    clickhouse_flush_async_insert, get_clickhouse,
};
use tensorzero_core::db::stored_datapoint::StoredChatInferenceDatapoint;
use tensorzero_core::evaluations::EvaluationConfig;
use tensorzero_core::inference::types::stored_input::{StoredInput, StoredInputMessage};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Role, StoredInputMessageContent, System, Text,
};
use tokio::time::sleep;
use uuid::Uuid;

/// Creates deterministic test datapoints for the `basic_test` function and writes them to ClickHouse.
/// This is used for top-k tests to avoid dependency on fixture files.
///
/// # Arguments
/// * `dataset_name` - The name of the dataset to write to
/// * `count` - The number of datapoints to create
async fn write_basic_test_datapoints(dataset_name: &str, count: usize) {
    let messages = [
        "Hello",
        "How are you?",
        "Tell me a joke",
        "What is the weather?",
        "Good morning",
        "Goodbye",
        "Help me",
        "Thanks",
        "What can you do?",
        "Test message",
    ];

    let datapoints: Vec<StoredChatInferenceDatapoint> = (0..count)
        .map(|i| {
            let message_text = messages[i % messages.len()];
            StoredChatInferenceDatapoint {
                dataset_name: dataset_name.to_string(),
                function_name: "basic_test".to_string(),
                id: Uuid::now_v7(),
                episode_id: None,
                input: StoredInput {
                    system: Some(System::Template(Arguments(Map::from_iter([(
                        "assistant_name".to_string(),
                        serde_json::json!("TestBot"),
                    )])))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: message_text.to_string(),
                        })],
                    }],
                },
                // Reference output equals input text - used with echo/empty models for exact_match testing
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: message_text.to_string(),
                })]),
                tool_params: None,
                tags: None,
                is_custom: false,
                source_inference_id: None,
                staled_at: None,
                name: None,
                snapshot_hash: None,
                is_deleted: false,
                auxiliary: String::new(),
                updated_at: String::new(),
            }
        })
        .collect();

    let clickhouse = get_clickhouse().await;
    clickhouse
        .write_batched(&datapoints, TableName::ChatInferenceDatapoint)
        .await
        .unwrap();
}

/// Helper to get a Postgres pool for tests
async fn get_postgres_pool() -> PgPool {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for top-k tests");
    PgPool::connect(&postgres_url)
        .await
        .expect("Failed to connect to Postgres")
}

/// Helper to create a durable queue if it doesn't exist.
///
/// Each test should use a unique queue name to prevent shared state.
async fn ensure_queue_exists(pool: &PgPool, queue_name: &str) {
    // The queue should be created by the durable migrations for production,
    // but for tests we create unique queues to prevent shared state.
    let client = durable::Durable::builder()
        .pool(pool.clone())
        .queue_name(queue_name)
        .build()
        .await
        .expect("Failed to create durable client");
    client
        .create_queue(None)
        .await
        .expect("Failed to create queue");
}

/// Test that top-k evaluation identifies the correct winner (TopKFound).
///
/// Setup:
/// - 3 variants: "echo" (uses echo model), "empty" and "empty2" (use empty model)
/// - Evaluators: "zero" (always 0), "one" (always 1), "exact_match" (1 if output matches reference)
/// - Datapoints have reference output equal to input text
/// - Scoring: AverageEvaluatorScore averages all evaluator scores
///
/// Expected scores:
/// - echo: (0 + 1 + 1) / 3 = 2/3 (exact_match=1 because echo returns input)
/// - empty/empty2: (0 + 1 + 0) / 3 = 1/3 (exact_match=0 because empty returns "")
///
/// The test verifies that "echo" is identified as the top-1 variant.
#[tokio::test(flavor = "multi_thread")]
async fn test_topk_found_topk() {
    init_tracing_for_tests();
    // Setup
    let config = get_config().await;
    let clickhouse = get_clickhouse().await;
    let tensorzero_client = get_tensorzero_client().await;
    let pg_pool = get_postgres_pool().await;

    // Use a unique queue name for this test to prevent shared state
    // Keep it short to avoid Postgres identifier length limits (63 chars)
    // Use simple format (no hyphens) to avoid SQL identifier issues
    let queue_name = format!("topk1_{}", Uuid::now_v7().simple());
    ensure_queue_exists(&pg_pool, &queue_name).await;

    // Create a unique dataset for this test with programmatically generated datapoints
    let dataset_name = format!("topk_test_topk_found_{}", Uuid::now_v7());

    // Write enough datapoints for the confidence sequences to converge
    // Based on simulation, exactly 25 datapoints needed for top-1 identification
    write_basic_test_datapoints(&dataset_name, 25).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    sleep(Duration::from_secs(1)).await;

    // Get the evaluation config for test_topk_evaluation
    // This evaluation uses zero, one, and exact_match evaluators (no error evaluator)
    let evaluation_config = config
        .evaluations
        .get("test_topk_evaluation")
        .expect("test_topk_evaluation not found in config")
        .clone();

    // Build the function configs table
    let EvaluationConfig::Inference(_inference_config) = &*evaluation_config;
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();

    // Use echo (score 2/3) and two empty variants (score 1/3 each)
    let variant_names = vec![
        "echo".to_string(),
        "empty".to_string(),
        "empty2".to_string(),
    ];

    // Create clients
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        clickhouse_client: clickhouse.clone(),
    });

    // Create the top-k task state (only clients)
    let state = TopKTaskState { clients };

    // Create the durable client with test-specific queue
    let durable_client = create_client(pg_pool.clone(), state, Some(&queue_name))
        .await
        .expect("Failed to create durable client");

    // Create task params with configs and scoring function
    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let params = TopKTaskParams {
        evaluation_name: "test_topk_evaluation".to_string(),
        dataset_name: dataset_name.clone(),
        variant_names: variant_names.clone(),
        k_min: 1,
        k_max: 1,
        epsilon: None, // No epsilon relaxation - require strict separation
        max_datapoints: Some(25),
        batch_size: Some(25),
        variant_failure_threshold: 1.0,   // disabled
        evaluator_failure_threshold: 1.0, // disabled
        concurrency: 10,
        inference_cache: CacheEnabledMode::Off, // No caching for clean test
        evaluation_config: EvaluationConfig::Inference(inference_config.clone()),
        function_configs,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    // Spawn the task
    let spawn_result = durable_client
        .spawn::<evaluations::topk::TopKTask>(params)
        .await
        .expect("Failed to spawn top-k task");

    // Start a worker to process the task
    let worker = durable_client
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(100),
            claim_timeout: Duration::from_secs(60),
            ..Default::default()
        })
        .await
        .expect("Failed to start worker");

    // Wait for the task to complete (with timeout)
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(180);

    loop {
        if start.elapsed() > timeout {
            worker.shutdown().await;
            panic!("Top-k task timed out after {timeout:?}");
        }

        // Check task state (use dynamic table name based on queue)
        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(spawn_result.task_id)
            .fetch_optional(&pg_pool)
            .await
            .expect("Failed to query task state");

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                worker.shutdown().await;
                panic!("Top-k task failed");
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    // Get the task result
    let query = format!("SELECT completed_payload FROM durable.t_{queue_name} WHERE task_id = $1");
    let result: Option<(Option<serde_json::Value>,)> = query_as(AssertSqlSafe(query))
        .bind(spawn_result.task_id)
        .fetch_optional(&pg_pool)
        .await
        .expect("Failed to query task result");

    let output: TopKTaskOutput = result
        .and_then(|(payload,)| payload)
        .map(|v| serde_json::from_value(v).expect("Failed to deserialize output"))
        .expect("No task output found");

    // Shutdown worker
    worker.shutdown().await;

    // === Assertions ===

    // 1. Check stopping reason is TopKFound with k=1 and echo as winner
    match &output.stopping_reason {
        GlobalStoppingReason::TopKFound { k, top_variants } => {
            assert_eq!(*k, 1, "Should have found top-1");
            assert_eq!(top_variants.len(), 1, "Should have exactly one top variant");
            assert_eq!(
                top_variants[0], "echo",
                "Echo should be the top variant (score 2/3 vs 1/3)"
            );
        }
        other => {
            panic!("Expected TopKFound, got: {other:?}");
        }
    }

    // 2. Check variant statuses
    assert_eq!(
        output.variant_status.get("echo"),
        Some(&VariantStatus::Include),
        "Echo should be Included (winner)"
    );
    assert_eq!(
        output.variant_status.get("empty"),
        Some(&VariantStatus::Exclude),
        "empty should be Excluded (loser)"
    );
    assert_eq!(
        output.variant_status.get("empty2"),
        Some(&VariantStatus::Exclude),
        "Empty2 should be Excluded (loser)"
    );

    // 3. Check variant performance confidence sequences
    let echo_cs = output
        .variant_performance
        .get("echo")
        .expect("Echo performance not found");
    let empty_cs = output
        .variant_performance
        .get("empty")
        .expect("empty performance not found");
    let empty2_cs = output
        .variant_performance
        .get("empty2")
        .expect("Empty2 performance not found");

    // 3.1. Echo variant confidence sequence
    assert_eq!(echo_cs.count, 25, "echo count");
    assert!(
        (echo_cs.mean_est - 0.666000000000000).abs() < 1e-10,
        "echo mean_est {} != 0.666",
        echo_cs.mean_est
    );
    assert!(
        (echo_cs.cs_lower - 0.505000000000000).abs() < 1e-10,
        "echo cs_lower {} != 0.505",
        echo_cs.cs_lower
    );
    assert!(
        (echo_cs.cs_upper - 0.748000000000000).abs() < 1e-10,
        "echo cs_upper {} != 0.748",
        echo_cs.cs_upper
    );
    assert!(
        (echo_cs.mean_regularized - 0.660256410256410).abs() < 1e-10,
        "echo mean_regularized {} != 0.660256410256410",
        echo_cs.mean_regularized
    );
    assert!(
        (echo_cs.variance_regularized - 0.010264105441808).abs() < 1e-10,
        "echo variance_regularized {} != 0.010264105441808",
        echo_cs.variance_regularized
    );

    // 3.2. empty variant confidence sequence
    assert_eq!(empty_cs.count, 25, "empty count");
    assert!(
        (empty_cs.mean_est - 0.333000000000000).abs() < 1e-10,
        "empty mean_est {} != 0.333",
        empty_cs.mean_est
    );
    assert!(
        (empty_cs.cs_lower - 0.252000000000000).abs() < 1e-10,
        "empty cs_lower {} != 0.252",
        empty_cs.cs_lower
    );
    assert!(
        (empty_cs.cs_upper - 0.495000000000000).abs() < 1e-10,
        "empty cs_upper {} != 0.495",
        empty_cs.cs_upper
    );
    assert!(
        (empty_cs.mean_regularized - 0.339743589743590).abs() < 1e-10,
        "empty mean_regularized {} != 0.339743589743590",
        empty_cs.mean_regularized
    );
    assert!(
        (empty_cs.variance_regularized - 0.010264105441808).abs() < 1e-10,
        "empty variance_regularized {} != 0.010264105441808",
        empty_cs.variance_regularized
    );

    // 3.3. Empty2 variant confidence sequence (should be identical to empty)
    assert_eq!(empty2_cs.count, 25, "empty2 count");
    assert!(
        (empty2_cs.mean_est - 0.333000000000000).abs() < 1e-10,
        "empty2 mean_est {} != 0.333",
        empty2_cs.mean_est
    );
    assert!(
        (empty2_cs.cs_lower - 0.252000000000000).abs() < 1e-10,
        "empty2 cs_lower {} != 0.252",
        empty2_cs.cs_lower
    );
    assert!(
        (empty2_cs.cs_upper - 0.495000000000000).abs() < 1e-10,
        "empty2 cs_upper {} != 0.495",
        empty2_cs.cs_upper
    );
    assert!(
        (empty2_cs.mean_regularized - 0.339743589743590).abs() < 1e-10,
        "empty2 mean_regularized {} != 0.339743589743590",
        empty2_cs.mean_regularized
    );
    assert!(
        (empty2_cs.variance_regularized - 0.010264105441808).abs() < 1e-10,
        "empty2 variance_regularized {} != 0.010264105441808",
        empty2_cs.variance_regularized
    );

    // 4. Check variant failures confidence sequences. There shouldn't be any inference failures,
    // so all variants should have identical failure statistics.
    let echo_failures = output
        .variant_failures
        .get("echo")
        .expect("Echo failures not found");
    let empty_failures = output
        .variant_failures
        .get("empty")
        .expect("empty failures not found");
    let empty2_failures = output
        .variant_failures
        .get("empty2")
        .expect("Empty2 failures not found");

    // 4.1. Echo variant failures
    assert_eq!(echo_failures.count, 25, "echo failures count");
    assert!(
        (echo_failures.mean_est - 0.0).abs() < 1e-10,
        "echo failures mean_est {} != 0.0",
        echo_failures.mean_est
    );
    assert!(
        (echo_failures.cs_lower - 0.0).abs() < 1e-10,
        "echo failures cs_lower {} != 0.0",
        echo_failures.cs_lower
    );
    assert!(
        (echo_failures.cs_upper - 0.242).abs() < 1e-10,
        "echo failures cs_upper {} != 0.242",
        echo_failures.cs_upper
    );
    assert!(
        (echo_failures.mean_regularized - 0.019230769230769).abs() < 1e-10,
        "echo failures mean_regularized {} != 0.019230769230769",
        echo_failures.mean_regularized
    );
    assert!(
        (echo_failures.variance_regularized - 0.015453872053191).abs() < 1e-10,
        "echo failures variance_regularized {} != 0.015453872053191",
        echo_failures.variance_regularized
    );

    // 4.2. empty variant failures (should be identical to echo)
    assert_eq!(empty_failures.count, 25, "empty failures count");
    assert!(
        (empty_failures.mean_est - 0.0).abs() < 1e-10,
        "empty failures mean_est {} != 0.0",
        empty_failures.mean_est
    );
    assert!(
        (empty_failures.cs_lower - 0.0).abs() < 1e-10,
        "empty failures cs_lower {} != 0.0",
        empty_failures.cs_lower
    );
    assert!(
        (empty_failures.cs_upper - 0.242).abs() < 1e-10,
        "empty failures cs_upper {} != 0.242",
        empty_failures.cs_upper
    );
    assert!(
        (empty_failures.mean_regularized - 0.019230769230769).abs() < 1e-10,
        "empty failures mean_regularized {} != 0.019230769230769",
        empty_failures.mean_regularized
    );
    assert!(
        (empty_failures.variance_regularized - 0.015453872053191).abs() < 1e-10,
        "empty failures variance_regularized {} != 0.015453872053191",
        empty_failures.variance_regularized
    );

    // 4.3. Empty2 variant failures (should be identical to echo)
    assert_eq!(empty2_failures.count, 25, "empty2 failures count");
    assert!(
        (empty2_failures.mean_est - 0.0).abs() < 1e-10,
        "empty2 failures mean_est {} != 0.0",
        empty2_failures.mean_est
    );
    assert!(
        (empty2_failures.cs_lower - 0.0).abs() < 1e-10,
        "empty2 failures cs_lower {} != 0.0",
        empty2_failures.cs_lower
    );
    assert!(
        (empty2_failures.cs_upper - 0.242).abs() < 1e-10,
        "empty2 failures cs_upper {} != 0.242",
        empty2_failures.cs_upper
    );
    assert!(
        (empty2_failures.mean_regularized - 0.019230769230769).abs() < 1e-10,
        "empty2 failures mean_regularized {} != 0.019230769230769",
        empty2_failures.mean_regularized
    );
    assert!(
        (empty2_failures.variance_regularized - 0.015453872053191).abs() < 1e-10,
        "empty2 failures variance_regularized {} != 0.015453872053191",
        empty2_failures.variance_regularized
    );

    // 5. Check evaluator failures confidence sequences.
    // test_topk_evaluation uses zero, one, and exact_match evaluators - none of which fail.
    // Each evaluator processes 25 datapoints * 3 variants = 75 observations, all with 0 failures.
    let zero_failures = output
        .evaluator_failures
        .get("zero")
        .expect("Zero failures not found");
    let one_failures = output
        .evaluator_failures
        .get("one")
        .expect("One failures not found");
    let exact_match_failures = output
        .evaluator_failures
        .get("exact_match")
        .expect("Exact_match failures not found");

    // 5.1. Zero evaluator failures
    assert_eq!(zero_failures.count, 75, "zero failures count");
    assert!(
        (zero_failures.mean_est - 0.0).abs() < 1e-10,
        "zero failures mean_est {} != 0.0",
        zero_failures.mean_est
    );
    assert!(
        (zero_failures.cs_lower - 0.0).abs() < 1e-10,
        "zero failures cs_lower {} != 0.0",
        zero_failures.cs_lower
    );
    assert!(
        (zero_failures.cs_upper - 0.092).abs() < 1e-10,
        "zero failures cs_upper {} != 0.092",
        zero_failures.cs_upper
    );
    assert!(
        (zero_failures.mean_regularized - 0.006578947368421).abs() < 1e-10,
        "zero failures mean_regularized {} != 0.006578947368421",
        zero_failures.mean_regularized
    );
    assert!(
        (zero_failures.variance_regularized - 0.005367968281414).abs() < 1e-10,
        "zero failures variance_regularized {} != 0.005367968281414",
        zero_failures.variance_regularized
    );

    // 5.2. One evaluator failures (should be identical to zero)
    assert_eq!(one_failures.count, 75, "one failures count");
    assert!(
        (one_failures.mean_est - 0.0).abs() < 1e-10,
        "one failures mean_est {} != 0.0",
        one_failures.mean_est
    );
    assert!(
        (one_failures.cs_lower - 0.0).abs() < 1e-10,
        "one failures cs_lower {} != 0.0",
        one_failures.cs_lower
    );
    assert!(
        (one_failures.cs_upper - 0.092).abs() < 1e-10,
        "one failures cs_upper {} != 0.092",
        one_failures.cs_upper
    );
    assert!(
        (one_failures.mean_regularized - 0.006578947368421).abs() < 1e-10,
        "one failures mean_regularized {} != 0.006578947368421",
        one_failures.mean_regularized
    );
    assert!(
        (one_failures.variance_regularized - 0.005367968281414).abs() < 1e-10,
        "one failures variance_regularized {} != 0.005367968281414",
        one_failures.variance_regularized
    );

    // 5.3. Exact_match evaluator failures (should be identical to zero and one)
    assert_eq!(exact_match_failures.count, 75, "exact_match failures count");
    assert!(
        (exact_match_failures.mean_est - 0.0).abs() < 1e-10,
        "exact_match failures mean_est {} != 0.0",
        exact_match_failures.mean_est
    );
    assert!(
        (exact_match_failures.cs_lower - 0.0).abs() < 1e-10,
        "exact_match failures cs_lower {} != 0.0",
        exact_match_failures.cs_lower
    );
    assert!(
        (exact_match_failures.cs_upper - 0.092).abs() < 1e-10,
        "exact_match failures cs_upper {} != 0.092",
        exact_match_failures.cs_upper
    );
    assert!(
        (exact_match_failures.mean_regularized - 0.006578947368421).abs() < 1e-10,
        "exact_match failures mean_regularized {} != 0.006578947368421",
        exact_match_failures.mean_regularized
    );
    assert!(
        (exact_match_failures.variance_regularized - 0.005367968281414).abs() < 1e-10,
        "exact_match failures variance_regularized {} != 0.005367968281414",
        exact_match_failures.variance_regularized
    );

    // 6. Check number of datapoints processed
    assert_eq!(
        output.num_datapoints_processed, 25,
        "Should process exactly 25 datapoints before finding top-1"
    );
}

/// Test that top-k evaluation stops with DatasetExhausted when variants can't be separated.
///
/// Setup:
/// - 4 variants: "test", "test2", "empty1", "empty2"
///   - test/test2 use dummy model returning a fixed string
///   - empty1/empty2 use empty model returning ""
/// - Evaluators: "zero" (always 0), "one" (always 1), "exact_match" (0 for all since output != input)
///
/// Since all variants have identical scores, they will never be distinguishable.
#[tokio::test(flavor = "multi_thread")]
async fn test_topk_dataset_exhaustion() {
    // Setup
    let config = get_config().await;
    let clickhouse = get_clickhouse().await;
    let tensorzero_client = get_tensorzero_client().await;
    let pg_pool = get_postgres_pool().await;

    // Use a unique queue name for this test to prevent shared state
    // Keep it short to avoid Postgres identifier length limits (63 chars)
    // Use simple format (no hyphens) to avoid SQL identifier issues
    let queue_name = format!("topk2_{}", Uuid::now_v7().simple());
    ensure_queue_exists(&pg_pool, &queue_name).await;

    // Create a unique dataset
    let dataset_name = format!("topk_test_exhaustion_{}", Uuid::now_v7());

    // Write 25 datapoints - same as test_topk_topk_found
    write_basic_test_datapoints(&dataset_name, 25).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    sleep(Duration::from_secs(1)).await;

    // Get the evaluation config for test_topk_evaluation
    // This evaluation uses zero, one, and exact_match evaluators (no error evaluator)
    let evaluation_config = config
        .evaluations
        .get("test_topk_evaluation")
        .expect("test_topk_evaluation not found in config")
        .clone();

    // Build the function configs table
    let EvaluationConfig::Inference(_inference_config) = &*evaluation_config;
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();

    // Use four variants that all have identical scores (1/3 each)
    // - test/test2: return fixed dummy string, exact_match = 0
    // - empty1/empty2: return "", exact_match = 0
    // Since they're identical, confidence intervals will never separate
    let variant_names = vec![
        "test".to_string(),
        "test2".to_string(),
        "empty1".to_string(),
        "empty2".to_string(),
    ];

    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        clickhouse_client: clickhouse.clone(),
    });

    let state = TopKTaskState { clients };

    // Create the durable client with test-specific queue
    let durable_client = create_client(pg_pool.clone(), state, Some(&queue_name))
        .await
        .expect("Failed to create durable client");

    // With 25 datapoints and 4 identical variants, the dataset will be exhausted
    // before we can identify top-2 or top-3 (confidence intervals will always overlap)
    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let params = TopKTaskParams {
        evaluation_name: "test_topk_evaluation".to_string(),
        dataset_name: dataset_name.clone(),
        variant_names,
        k_min: 2,
        k_max: 3,
        epsilon: None, // No epsilon relaxation - require strict separation
        max_datapoints: Some(25),
        batch_size: Some(5),
        variant_failure_threshold: 1.0,   // disabled
        evaluator_failure_threshold: 1.0, // disabled
        concurrency: 10,
        inference_cache: CacheEnabledMode::Off, // No caching for clean test
        evaluation_config: EvaluationConfig::Inference(inference_config.clone()),
        function_configs,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let spawn_result = durable_client
        .spawn::<evaluations::topk::TopKTask>(params)
        .await
        .expect("Failed to spawn top-k task");

    let worker = durable_client
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(100),
            claim_timeout: Duration::from_secs(60),
            ..Default::default()
        })
        .await
        .expect("Failed to start worker");

    // Wait for completion
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(120);

    loop {
        if start.elapsed() > timeout {
            worker.shutdown().await;
            panic!("Top-k task timed out after {timeout:?}");
        }

        // Check task state (use dynamic table name based on queue)
        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(spawn_result.task_id)
            .fetch_optional(&pg_pool)
            .await
            .expect("Failed to query task state");

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                worker.shutdown().await;
                panic!("Top-k task failed");
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    // Get the task result
    let query = format!("SELECT completed_payload FROM durable.t_{queue_name} WHERE task_id = $1");
    let result: Option<(Option<serde_json::Value>,)> = query_as(AssertSqlSafe(query))
        .bind(spawn_result.task_id)
        .fetch_optional(&pg_pool)
        .await
        .expect("Failed to query task result");

    let output: TopKTaskOutput = result
        .and_then(|(payload,)| payload)
        .map(|v| serde_json::from_value(v).expect("Failed to deserialize output"))
        .expect("No task output found");

    worker.shutdown().await;

    // Verify we got DatasetExhausted
    assert!(
        matches!(
            output.stopping_reason,
            GlobalStoppingReason::DatasetExhausted
        ),
        "Expected DatasetExhausted, got: {:?}",
        output.stopping_reason
    );

    // Should have processed exactly max_datapoints
    assert_eq!(
        output.num_datapoints_processed, 25,
        "Should have processed exactly 25 datapoints"
    );
}

/// Test that top-k evaluation stops with EvaluatorsFailed when evaluators exceed failure threshold.
///
/// Setup:
/// - 2 variants: "test", "test2" (both use dummy model)
/// - Evaluators: happy_bool, sad_bool, zero, one, error, exact_match
/// - The "error" evaluator always fails (failure rate = 1.0)
/// - Failure threshold: 0.05 (5%)
/// - Each datapoint produces 2 observations per evaluator (one per variant)
///
/// Expected behavior:
/// - After 1 datapoint (2 observations): cs_lower = 0.034 < 0.05 (continue)
/// - After 2 datapoints (4 observations): cs_lower = 0.248 > 0.05 (stop)
#[tokio::test(flavor = "multi_thread")]
async fn test_topk_evaluator_failure_threshold() {
    // Setup
    let config = get_config().await;
    let clickhouse = get_clickhouse().await;
    let tensorzero_client = get_tensorzero_client().await;
    let pg_pool = get_postgres_pool().await;

    // Use a unique queue name for this test to prevent shared state
    // Keep it short to avoid Postgres identifier length limits (63 chars)
    // Use simple format (no hyphens) to avoid SQL identifier issues
    let queue_name = format!("topk3_{}", Uuid::now_v7().simple());
    ensure_queue_exists(&pg_pool, &queue_name).await;

    // Create a unique dataset
    let dataset_name = format!("topk_test_eval_fail_{}", Uuid::now_v7());

    // Write deterministic test datapoints for test_evaluation (has dummy providers and error evaluator)
    // Write 25 datapoints to ensure enough are available even if ClickHouse is slow
    write_basic_test_datapoints(&dataset_name, 25).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    sleep(Duration::from_secs(1)).await;

    // Get the evaluation config for test_evaluation which has an "error" evaluator
    let evaluation_config = config
        .evaluations
        .get("test_evaluation")
        .expect("test_evaluation not found in config")
        .clone();

    let EvaluationConfig::Inference(_inference_config) = &*evaluation_config;
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();

    // Use two dummy variants for basic_test function
    // Each datapoint produces 2 observations per evaluator
    let variant_names = vec!["test".to_string(), "test2".to_string()];

    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        clickhouse_client: clickhouse.clone(),
    });

    let state = TopKTaskState { clients };

    // Create the durable client with test-specific queue
    let durable_client = create_client(pg_pool.clone(), state, Some(&queue_name))
        .await
        .expect("Failed to create durable client");

    // Set evaluator failure threshold to 0.05
    // The "error" evaluator always fails (100% failure rate)
    // After 2 datapoints (4 observations), cs_lower = 0.248 > 0.05, so we stop
    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let params = TopKTaskParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: dataset_name.clone(),
        variant_names,
        k_min: 1,
        k_max: 1,
        epsilon: None,
        max_datapoints: Some(25),
        batch_size: Some(1),               // Process one datapoint at a time
        variant_failure_threshold: 1.0,    // disabled
        evaluator_failure_threshold: 0.05, // 5% failure rate threshold
        concurrency: 10,
        inference_cache: CacheEnabledMode::Off,
        evaluation_config: EvaluationConfig::Inference(inference_config.clone()),
        function_configs,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let spawn_result = durable_client
        .spawn::<evaluations::topk::TopKTask>(params)
        .await
        .expect("Failed to spawn top-k task");

    let worker = durable_client
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(100),
            claim_timeout: Duration::from_secs(60),
            ..Default::default()
        })
        .await
        .expect("Failed to start worker");

    // Wait for completion
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(120);

    loop {
        if start.elapsed() > timeout {
            worker.shutdown().await;
            panic!("Top-k task timed out after {timeout:?}");
        }

        // Check task state (use dynamic table name based on queue)
        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(spawn_result.task_id)
            .fetch_optional(&pg_pool)
            .await
            .expect("Failed to query task state");

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                worker.shutdown().await;
                panic!("Top-k task failed");
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    // Get the task result
    let query = format!("SELECT completed_payload FROM durable.t_{queue_name} WHERE task_id = $1");
    let result: Option<(Option<serde_json::Value>,)> = query_as(AssertSqlSafe(query))
        .bind(spawn_result.task_id)
        .fetch_optional(&pg_pool)
        .await
        .expect("Failed to query task result");

    let output: TopKTaskOutput = result
        .and_then(|(payload,)| payload)
        .map(|v| serde_json::from_value(v).expect("Failed to deserialize output"))
        .expect("No task output found");

    worker.shutdown().await;

    // === Assertions ===

    // 1. Verify stopping reason is EvaluatorsFailed with "error" evaluator
    match &output.stopping_reason {
        GlobalStoppingReason::EvaluatorsFailed { evaluator_names } => {
            assert!(
                evaluator_names.contains(&"error".to_string()),
                "error evaluator should be in the failed list"
            );
        }
        other => {
            panic!("Unexpected stopping reason: {other:?}");
        }
    }

    // 2. Verify number of datapoints processed
    assert_eq!(
        output.num_datapoints_processed, 2,
        "Should process exactly 2 datapoints before evaluator failure threshold exceeded"
    );

    // 3. Verify error evaluator failures confidence sequence
    let error_failures = output
        .evaluator_failures
        .get("error")
        .expect("error failures not found");

    assert_eq!(error_failures.count, 4, "error failures count");
    assert!(
        (error_failures.mean_est - 1.0).abs() < 1e-10,
        "error failures mean_est {} != 1.0",
        error_failures.mean_est
    );
    assert!(
        (error_failures.cs_lower - 0.248).abs() < 1e-10,
        "error failures cs_lower {} != 0.248",
        error_failures.cs_lower
    );
    assert!(
        (error_failures.cs_upper - 1.0).abs() < 1e-10,
        "error failures cs_upper {} != 1.0",
        error_failures.cs_upper
    );
    assert!(
        (error_failures.mean_regularized - 0.9).abs() < 1e-10,
        "error failures mean_regularized {} != 0.9",
        error_failures.mean_regularized
    );
    assert!(
        (error_failures.variance_regularized - 0.073180555555556).abs() < 1e-10,
        "error failures variance_regularized {} != 0.073180555555556",
        error_failures.variance_regularized
    );
}

/// Test that top-k evaluation handles variant failures correctly.
/// When too many variants fail, the task should stop with TooManyVariantsFailed.
///
/// Setup:
/// - 3 variants: "test" (working), "error" (always fails), "error2" (always fails)
/// - k_min=2, so we need at least 2 active (non-failed) variants
/// - The error variants use models starting with "error" which the dummy provider fails
/// - Threshold: 0.05 (5%)
/// - Each datapoint produces 1 observation per variant
///
/// Expected:
/// - After 2 datapoints (2 obs): cs_lower = 0.034 < 0.05 (continue)
/// - After 3 datapoints (3 obs): cs_lower = 0.171 > 0.05 (error and error2 are Failed)
/// - num_failed=2 > num_variants - k_min = 3-2=1, so TooManyVariantsFailed triggers
#[tokio::test(flavor = "multi_thread")]
async fn test_topk_variant_failure_threshold() {
    // Setup
    let config = get_config().await;
    let clickhouse = get_clickhouse().await;
    let tensorzero_client = get_tensorzero_client().await;
    let pg_pool = get_postgres_pool().await;

    // Use a unique queue name for this test to prevent shared state
    // Keep it short to avoid Postgres identifier length limits (63 chars)
    // Use simple format (no hyphens) to avoid SQL identifier issues
    let queue_name = format!("topk4_{}", Uuid::now_v7().simple());
    ensure_queue_exists(&pg_pool, &queue_name).await;

    // Create a unique dataset
    let dataset_name = format!("topk_test_variant_fail_{}", Uuid::now_v7());

    // Write deterministic test datapoints
    // Write 25 datapoints to ensure enough are available even if ClickHouse is slow
    write_basic_test_datapoints(&dataset_name, 25).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    sleep(Duration::from_secs(1)).await;

    // Get the test_topk_evaluation config (uses basic_test function)
    let evaluation_config = config
        .evaluations
        .get("test_topk_evaluation")
        .expect("test_topk_evaluation not found in config")
        .clone();

    let EvaluationConfig::Inference(_inference_config) = &*evaluation_config;
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();

    // Use 3 variants: 1 working, 2 failing
    // - "test": uses dummy model, always succeeds
    // - "error", "error2": use error models, always fail (model_name starts with "error")
    let variant_names = vec![
        "test".to_string(),
        "error".to_string(),
        "error2".to_string(),
    ];

    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        clickhouse_client: clickhouse.clone(),
    });

    let state = TopKTaskState { clients };

    // Create the durable client with test-specific queue
    let durable_client = create_client(pg_pool.clone(), state, Some(&queue_name))
        .await
        .expect("Failed to create durable client");

    // Set variant failure threshold to 0.05
    // After 3 datapoints (3 observations per variant), error variants have cs_lower = 0.171 > 0.05
    // With k_min=2, we need 2 active variants. When 2 fail, we only have 1 active,
    // so num_failed=2 > num_variants - k_min = 3-2=1, triggering TooManyVariantsFailed
    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let params = TopKTaskParams {
        evaluation_name: "test_topk_evaluation".to_string(),
        dataset_name: dataset_name.clone(),
        variant_names,
        k_min: 2,
        k_max: 2,
        epsilon: None,
        max_datapoints: Some(25),
        batch_size: Some(1),              // Process one datapoint at a time
        variant_failure_threshold: 0.05,  // 5% failure rate threshold
        evaluator_failure_threshold: 1.0, // disabled
        concurrency: 10,
        inference_cache: CacheEnabledMode::Off,
        evaluation_config: EvaluationConfig::Inference(inference_config.clone()),
        function_configs,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let spawn_result = durable_client
        .spawn::<evaluations::topk::TopKTask>(params)
        .await
        .expect("Failed to spawn top-k task");

    let worker = durable_client
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(100),
            claim_timeout: Duration::from_secs(60),
            ..Default::default()
        })
        .await
        .expect("Failed to start worker");

    // Wait for completion
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(120);

    loop {
        if start.elapsed() > timeout {
            worker.shutdown().await;
            panic!("Top-k task timed out after {timeout:?}");
        }

        // Check task state (use dynamic table name based on queue)
        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(spawn_result.task_id)
            .fetch_optional(&pg_pool)
            .await
            .expect("Failed to query task state");

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                worker.shutdown().await;
                panic!("Top-k task failed");
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    // Get the task result
    let query = format!("SELECT completed_payload FROM durable.t_{queue_name} WHERE task_id = $1");
    let result: Option<(Option<serde_json::Value>,)> = query_as(AssertSqlSafe(query))
        .bind(spawn_result.task_id)
        .fetch_optional(&pg_pool)
        .await
        .expect("Failed to query task result");

    let output: TopKTaskOutput = result
        .and_then(|(payload,)| payload)
        .map(|v| serde_json::from_value(v).expect("Failed to deserialize output"))
        .expect("No task output found");

    worker.shutdown().await;

    // 1. Verify stopping reason is TooManyVariantsFailed
    match &output.stopping_reason {
        GlobalStoppingReason::TooManyVariantsFailed { num_failed } => {
            assert_eq!(*num_failed, 2, "Exactly 2 variants should have failed");
        }
        other => {
            panic!("Unexpected stopping reason: {other:?}");
        }
    }

    // 2. Verify number of datapoints processed
    assert_eq!(
        output.num_datapoints_processed, 3,
        "Should process exactly 3 datapoints before variant failure threshold exceeded"
    );

    // 3. Verify variant statuses
    assert_eq!(
        output.variant_status.get("test"),
        Some(&VariantStatus::Include),
        "test variant status should be Include"
    );
    assert_eq!(
        output.variant_status.get("error"),
        Some(&VariantStatus::Failed),
        "error variant status should be Failed"
    );
    assert_eq!(
        output.variant_status.get("error2"),
        Some(&VariantStatus::Failed),
        "error2 variant status should be Failed"
    );

    // 4. Verify error variant failures confidence sequence
    let error_failures = output
        .variant_failures
        .get("error")
        .expect("error failures not found");

    assert_eq!(error_failures.count, 3, "error failures count");
    assert!(
        (error_failures.mean_est - 1.0).abs() < 1e-10,
        "error failures mean_est {} != 1.0",
        error_failures.mean_est
    );
    assert!(
        (error_failures.cs_lower - 0.171).abs() < 1e-10,
        "error failures cs_lower {} != 0.171",
        error_failures.cs_lower
    );

    // 5. Verify error2 variant failures confidence sequence (should be identical to error)
    let error2_failures = output
        .variant_failures
        .get("error2")
        .expect("error2 failures not found");

    assert_eq!(error2_failures.count, 3, "error2 failures count");
    assert!(
        (error2_failures.mean_est - 1.0).abs() < 1e-10,
        "error2 failures mean_est {} != 1.0",
        error2_failures.mean_est
    );
    assert!(
        (error2_failures.cs_lower - 0.171).abs() < 1e-10,
        "error2 failures cs_lower {} != 0.171",
        error2_failures.cs_lower
    );

    // 6. Verify test variant has 0 failures
    let test_failures = output
        .variant_failures
        .get("test")
        .expect("test failures not found");

    assert_eq!(test_failures.count, 3, "test failures count");
    assert!(
        (test_failures.mean_est - 0.0).abs() < 1e-10,
        "test failures mean_est {} != 0.0",
        test_failures.mean_est
    );
}

/// Test that top-k evaluation emits progress events via durable's emit_event.
///
/// This test verifies that:
/// 1. Progress events are emitted after each batch via `topk_progress:{task_id}:{batch_idx}`
/// 2. Completion events are emitted via both:
///    - `topk_completed:{task_id}`
///    - `topk_progress:{task_id}:{N}` (where N is the number of batches)
/// 3. The event payloads contain correct data (variant summaries, statuses, etc.)
///
/// Setup is similar to test_topk_found_topk. We verify events by querying the durable
/// events table directly after task completion.
#[tokio::test(flavor = "multi_thread")]
async fn test_topk_emit_event_streaming() {
    init_tracing_for_tests();
    // Setup
    let config = get_config().await;
    let clickhouse = get_clickhouse().await;
    let tensorzero_client = get_tensorzero_client().await;
    let pg_pool = get_postgres_pool().await;

    // Use a unique queue name for this test
    let queue_name = format!("topk5_{}", Uuid::now_v7().simple());
    ensure_queue_exists(&pg_pool, &queue_name).await;

    // Create a unique dataset
    let dataset_name = format!("topk_test_emit_{}", Uuid::now_v7());

    // Write datapoints - use 25 to ensure we get progress updates
    write_basic_test_datapoints(&dataset_name, 25).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    sleep(Duration::from_secs(1)).await;

    // Get the evaluation config
    let evaluation_config = config
        .evaluations
        .get("test_topk_evaluation")
        .expect("test_topk_evaluation not found in config")
        .clone();

    let EvaluationConfig::Inference(_inference_config) = &*evaluation_config;
    let function_configs: EvaluationFunctionConfigTable = config
        .functions
        .iter()
        .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
        .collect();

    // Use echo and empty variants (same as test_topk_found_topk)
    let variant_names = vec![
        "echo".to_string(),
        "empty".to_string(),
        "empty2".to_string(),
    ];

    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        clickhouse_client: clickhouse.clone(),
    });

    let state = TopKTaskState { clients };

    let durable_client = create_client(pg_pool.clone(), state, Some(&queue_name))
        .await
        .expect("Failed to create durable client");

    // Use batch_size=5 so we get multiple batches (25/5 = 5 batches max)
    let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
    let params = TopKTaskParams {
        evaluation_name: "test_topk_evaluation".to_string(),
        dataset_name: dataset_name.clone(),
        variant_names: variant_names.clone(),
        k_min: 1,
        k_max: 1,
        epsilon: None,
        max_datapoints: Some(25),
        batch_size: Some(5), // 5 batches of 5 datapoints each
        variant_failure_threshold: 1.0,
        evaluator_failure_threshold: 1.0,
        concurrency: 10,
        inference_cache: CacheEnabledMode::Off,
        evaluation_config: EvaluationConfig::Inference(inference_config.clone()),
        function_configs,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    // Spawn the task
    let spawn_result = durable_client
        .spawn::<evaluations::topk::TopKTask>(params)
        .await
        .expect("Failed to spawn top-k task");

    let task_id = spawn_result.task_id;

    // Start the worker
    let worker = durable_client
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(100),
            claim_timeout: Duration::from_secs(60),
            ..Default::default()
        })
        .await
        .expect("Failed to start worker");

    // Wait for task completion (with timeout)
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(180);

    loop {
        if start.elapsed() > timeout {
            worker.shutdown().await;
            panic!("Top-k task timed out after {timeout:?}");
        }

        let query = format!("SELECT state FROM durable.t_{queue_name} WHERE task_id = $1");
        let state: Option<(String,)> = query_as(AssertSqlSafe(query))
            .bind(spawn_result.task_id)
            .fetch_optional(&pg_pool)
            .await
            .expect("Failed to query task state");

        if let Some((state,)) = state {
            if state == "completed" {
                break;
            } else if state == "failed" {
                worker.shutdown().await;
                panic!("Top-k task failed");
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    worker.shutdown().await;

    // === Verify emitted events by querying the events table ===

    // 1. Query all progress events (one per batch) and verify multiple batches were emitted
    let progress_event_prefix = format!("topk_progress:{task_id}:");
    let query = format!(
        "SELECT event_name, payload FROM durable.e_{queue_name} WHERE event_name LIKE $1 ORDER BY event_name"
    );
    let progress_results: Vec<(String, serde_json::Value)> = query_as(AssertSqlSafe(query))
        .bind(format!("{progress_event_prefix}%"))
        .fetch_all(&pg_pool)
        .await
        .expect("Failed to query progress events");

    // Verify we got at least 2 batches (this catches the first-writer-wins regression)
    assert!(
        progress_results.len() >= 2,
        "Expected at least 2 progress events (one per batch), got {}. \
         This may indicate the first-writer-wins regression where only the first batch is emitted.",
        progress_results.len()
    );

    // Verify each batch has valid data and num_datapoints_processed increases.
    // The last event in the sequence should be a Completed event (at batch index N).
    let mut prev_datapoints_processed = 0;
    let num_progress_events = progress_results.len();
    for (idx, (event_name, payload)) in progress_results.iter().enumerate() {
        // Verify event name has correct batch index
        let expected_event_name = format!("{progress_event_prefix}{idx}");
        assert_eq!(
            event_name, &expected_event_name,
            "Event name mismatch at index {idx}"
        );

        let progress_update: TopKUpdate =
            serde_json::from_value(payload.clone()).expect("Failed to deserialize progress event");

        let is_last = idx == num_progress_events - 1;

        match progress_update {
            TopKUpdate::BatchProgress(batch) => {
                assert!(
                    !is_last,
                    "Last event should be Completed, not BatchProgress"
                );
                assert_ne!(
                    batch.evaluation_run_id,
                    uuid::Uuid::nil(),
                    "Progress event should have a valid evaluation_run_id"
                );
                assert_eq!(
                    batch.total_datapoints, 25,
                    "Progress event should have correct total_datapoints"
                );
                assert!(
                    batch.num_datapoints_processed > prev_datapoints_processed,
                    "num_datapoints_processed should increase: batch {idx} has {} but previous was {}",
                    batch.num_datapoints_processed,
                    prev_datapoints_processed
                );
                prev_datapoints_processed = batch.num_datapoints_processed;
                assert_eq!(
                    batch.variant_summaries.len(),
                    3,
                    "Should have 3 variant summaries"
                );
                assert!(
                    batch.variant_summaries.contains_key("echo"),
                    "Should have echo variant summary"
                );
                assert_eq!(
                    batch.variant_statuses.len(),
                    3,
                    "Should have 3 variant statuses"
                );
            }
            TopKUpdate::Completed(completed) => {
                // The batch-style completion event should be the last one
                assert!(
                    is_last,
                    "Completed event should only appear at the last index, but found at {idx}"
                );
                assert_ne!(
                    completed.evaluation_run_id,
                    uuid::Uuid::nil(),
                    "Batch-style completion event should have a valid evaluation_run_id"
                );
            }
        }
    }

    // 2. Query the completion event
    let completion_event_name = format!("topk_completed:{task_id}");
    let query = format!("SELECT payload FROM durable.e_{queue_name} WHERE event_name = $1");
    let completion_result: Option<(serde_json::Value,)> = query_as(AssertSqlSafe(query))
        .bind(&completion_event_name)
        .fetch_optional(&pg_pool)
        .await
        .expect("Failed to query completion event");

    let completion_payload = completion_result.expect("Completion event should have been emitted");
    let completion_update: TopKUpdate = serde_json::from_value(completion_payload.0)
        .expect("Failed to deserialize completion event");

    match completion_update {
        TopKUpdate::Completed(completed) => {
            // The event payload contains evaluation_run_id (generated inside the task)
            assert_ne!(
                completed.evaluation_run_id,
                uuid::Uuid::nil(),
                "Completion event should have a valid evaluation_run_id"
            );

            // Verify stopping reason
            match &completed.stopping_reason {
                GlobalStoppingReason::TopKFound { k, top_variants } => {
                    assert_eq!(*k, 1, "Should have found top-1");
                    assert!(
                        top_variants.contains(&"echo".to_string()),
                        "Echo should be in top variants"
                    );
                }
                other => {
                    panic!("Expected TopKFound, got: {other:?}");
                }
            }

            // Verify final variant statuses
            assert_eq!(
                completed.final_variant_statuses.get("echo"),
                Some(&VariantStatus::Include),
                "Echo should be Included in completion event"
            );
        }
        TopKUpdate::BatchProgress(_) => panic!("Expected Completed event, got BatchProgress"),
    }
}
