use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use evaluations::{
    EvaluationCoreArgs, EvaluationFunctionConfig, EvaluationVariant, run_evaluation_core_streaming,
    stats::EvaluationUpdate,
};
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::datasets::v1::{list_datapoints, types::ListDatapointsRequest};
use tokio::time::sleep;
use uuid::Uuid;

use crate::common::{
    get_config, init_tracing_for_tests, query_boolean_feedback, write_chat_fixture_to_dataset,
};

/// Tests that a top-level evaluator (defined in `[evaluators.exact_match]`) works correctly
/// when used via `run_evaluation_core_streaming` with `evaluation_name: None`.
///
/// This exercises the "Named" evaluation path where evaluators are referenced by name
/// rather than defined inline in an `[evaluations.X]` block.
#[tokio::test(flavor = "multi_thread")]
async fn test_top_level_exact_match_evaluator() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haiku-data-top-level-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;

    // Load chat fixture data
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    // Get 2 datapoint IDs from the loaded dataset
    let request = ListDatapointsRequest {
        function_name: Some("write_haiku".to_string()),
        limit: Some(2),
        offset: Some(0),
        ..Default::default()
    };
    let dataset = list_datapoints(&db, dataset_name.clone(), request)
        .await
        .unwrap()
        .datapoints;
    assert_eq!(dataset.len(), 2, "Should have loaded at least 2 datapoints");
    let datapoint_ids: Vec<Uuid> = dataset.iter().map(|dp| dp.id()).collect();

    // Look up function config and its evaluators
    let write_haiku_func = config
        .functions
        .get("write_haiku")
        .expect("`write_haiku` function should exist");
    let function_config = EvaluationFunctionConfig::from(write_haiku_func.as_ref());

    // Build evaluators from the function's evaluator config
    let function_evaluator = write_haiku_func
        .evaluators()
        .get("exact_match")
        .expect("Function-level `exact_match` evaluator should exist on `write_haiku`");

    let mut evaluators = HashMap::new();
    evaluators.insert("exact_match".to_string(), function_evaluator.clone());

    // Build the inference executor using embedded gateway
    let tensorzero_client = make_embedded_gateway().await;
    let inference_executor = Arc::new(evaluations::ClientInferenceExecutor::new(tensorzero_client));

    let evaluation_run_id = Uuid::now_v7();

    let core_args = EvaluationCoreArgs {
        inference_executor,
        db: Arc::new(db.clone()),
        function_name: "write_haiku".to_string(),
        function_config,
        evaluators,
        evaluation_name: None, // Top-level evaluator: no evaluation name
        evaluation_run_id,
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids.clone()),
        variant: EvaluationVariant::Name("gpt_4o_mini".to_string()),
        concurrency: 2,
        inference_cache: CacheEnabledMode::Off,
        tags: HashMap::new(),
    };

    let result = run_evaluation_core_streaming(core_args, None, HashMap::new())
        .await
        .expect("Evaluation should succeed");

    assert_eq!(result.run_info.num_datapoints, 2);

    // Collect results
    let mut receiver = result.receiver;
    let mut successes = Vec::new();
    while let Some(update) = receiver.recv().await {
        match update {
            EvaluationUpdate::Success(info) => successes.push(info),
            EvaluationUpdate::Error(err) => panic!("Evaluation error: {}", err.message),
            EvaluationUpdate::RunInfo(_) | EvaluationUpdate::FatalError(_) => continue,
        }
    }

    assert_eq!(successes.len(), 2, "Should have 2 successful evaluations");

    // Wait for async writes to complete
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;

    // Verify feedback was written with function-level evaluator metric naming
    let expected_metric_name =
        "tensorzero::function_name::write_haiku::evaluator_name::exact_match";
    for info in &successes {
        let inference_id = info.response.inference_id();
        let feedback = query_boolean_feedback(&db, inference_id, Some(expected_metric_name))
            .await
            .expect("Should find boolean feedback for function-level evaluator");

        assert_eq!(
            feedback.metric_name, expected_metric_name,
            "Metric name should use function-level evaluator naming"
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(feedback.tags["tensorzero::evaluator_name"], "exact_match");
        // Function-level evaluators should not have an evaluation_name tag
        assert!(
            !feedback.tags.contains_key("tensorzero::evaluation_name"),
            "Function-level evaluator feedback should not have evaluation_name tag"
        );
    }
}
