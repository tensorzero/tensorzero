//! E2E tests for evaluation queries (ClickHouse and Postgres).

use std::collections::HashMap;

use rust_decimal::Decimal;
use tensorzero_core::config::MetricConfigOptimize;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::evaluation_queries::{
    ChatEvaluationResultRow, EvaluationQueries, EvaluationResultRow, InferenceEvaluationRunInsert,
    InferenceEvaluationRunMetricMetadata, InferenceEvaluationRunSource, JsonEvaluationResultRow,
};
use tensorzero_core::db::feedback::{BooleanMetricFeedbackInsert, FeedbackQueries};
use tensorzero_core::db::inferences::InferenceQueries;
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::stored_datapoint::{StoredChatInferenceDatapoint, StoredDatapoint};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::inference::InferenceParams;
use tensorzero_core::function::FunctionConfigType;
use tensorzero_core::inference::types::extra_body::UnfilteredInferenceExtraBody;
use tensorzero_core::inference::types::stored_input::StoredInput;
use tensorzero_core::inference::types::{
    ChatInferenceDatabaseInsert, FinishReason, StoredModelInference,
};
use tensorzero_core::tool::ToolCallConfigDatabaseInsert;
use uuid::Uuid;

fn make_test_inference_evaluation_run(run_id: Uuid) -> InferenceEvaluationRunInsert {
    InferenceEvaluationRunInsert {
        run_id,
        evaluation_name: format!("e2e_insert_eval_{run_id}"),
        function_name: "write_haiku".to_string(),
        function_type: FunctionConfigType::Chat,
        dataset_name: format!("e2e_insert_dataset_{run_id}"),
        variant_names: vec!["variant_a".to_string(), "variant_b".to_string()],
        metrics: vec![
            InferenceEvaluationRunMetricMetadata {
                name: format!("tensorzero::evaluation_name::e2e::{run_id}::bool"),
                evaluator_name: Some("exact_match".to_string()),
                value_type: "boolean".to_string(),
                optimize: Some(MetricConfigOptimize::Max),
            },
            InferenceEvaluationRunMetricMetadata {
                name: format!("tensorzero::evaluation_name::e2e::{run_id}::float"),
                evaluator_name: Some("llm_judge".to_string()),
                value_type: "float".to_string(),
                optimize: Some(MetricConfigOptimize::Min),
            },
        ],
        source: InferenceEvaluationRunSource::DatasetName,
        snapshot_hash: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
    }
}

/// Test that `insert_inference_evaluation_run` writes expected fields, verified via trait query methods.
async fn test_insert_inference_evaluation_run(conn: impl EvaluationQueries + TestDatabaseHelpers) {
    let run = make_test_inference_evaluation_run(Uuid::now_v7());

    conn.insert_inference_evaluation_run(&run)
        .await
        .expect("insert_inference_evaluation_run should succeed");
    conn.sleep_for_writes_to_be_visible().await;

    // Verify via search_evaluation_runs (queries the runs table by evaluation_name)
    let search_results = conn
        .search_evaluation_runs(
            Some(&run.evaluation_name),
            Some(&run.function_name),
            "",
            10,
            0,
        )
        .await
        .expect("search_evaluation_runs should succeed");
    assert_eq!(
        search_results.len(),
        1,
        "search should find exactly one run for the unique evaluation_name"
    );
    assert_eq!(
        search_results[0].evaluation_run_id, run.run_id,
        "search result run_id should match inserted value"
    );
    assert_eq!(
        search_results[0].evaluation_name, run.evaluation_name,
        "search result evaluation_name should match inserted value"
    );
    assert_eq!(
        search_results[0].dataset_name, run.dataset_name,
        "search result dataset_name should match inserted value"
    );
    assert_eq!(
        search_results[0].variant_name, "variant_a",
        "search result variant_name should be the first variant"
    );

    // Verify via list_evaluation_runs (returns more fields).
    // Fetch a page large enough to include our run even when other tests run concurrently.
    let listed = conn
        .list_evaluation_runs(100, 0)
        .await
        .expect("list_evaluation_runs should succeed");
    let row = listed
        .iter()
        .find(|r| r.evaluation_run_id == run.run_id)
        .expect("inserted run should appear in list_evaluation_runs");
    assert_eq!(
        row.evaluation_name, run.evaluation_name,
        "listed evaluation_name should match inserted value"
    );
    assert_eq!(
        row.function_name, run.function_name,
        "listed function_name should match inserted value"
    );
    assert_eq!(
        row.variant_name, "variant_a",
        "listed variant_name should be the first variant"
    );
    assert_eq!(
        row.dataset_name, run.dataset_name,
        "listed dataset_name should match inserted value"
    );
}
make_db_test!(test_insert_inference_evaluation_run);

// ============================================================================
// get_inference_evaluation_run_metadata tests
// ============================================================================

/// Test that `get_inference_evaluation_run_metadata` returns the correct metadata
/// after inserting a run.
async fn test_get_inference_evaluation_run_metadata(
    conn: impl EvaluationQueries + TestDatabaseHelpers,
) {
    let run = make_test_inference_evaluation_run(Uuid::now_v7());

    conn.insert_inference_evaluation_run(&run)
        .await
        .expect("insert should succeed");
    conn.sleep_for_writes_to_be_visible().await;

    let results = conn
        .get_inference_evaluation_run_metadata(&[run.run_id])
        .await
        .expect("get_inference_evaluation_run_metadata should succeed");
    assert_eq!(results.len(), 1, "should return one result");
    let (returned_run_id, metadata) = &results[0];
    assert_eq!(*returned_run_id, run.run_id, "run_id should match");

    assert_eq!(
        metadata.evaluation_name, run.evaluation_name,
        "evaluation_name should match"
    );
    assert_eq!(
        metadata.function_name, run.function_name,
        "function_name should match"
    );
    assert_eq!(
        metadata.function_type,
        FunctionConfigType::Chat,
        "function_type should match"
    );
    assert_eq!(metadata.metrics.len(), 2, "should have two metric entries");

    // Verify full metric details (order may differ, so sort by name)
    let mut actual_metrics = metadata.metrics.clone();
    actual_metrics.sort_by(|a, b| a.name.cmp(&b.name));
    let mut expected_metrics = run.metrics.clone();
    expected_metrics.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(actual_metrics, expected_metrics, "metrics should match");
}
make_db_test!(test_get_inference_evaluation_run_metadata);

/// Test that `get_inference_evaluation_run_metadata` returns empty for a nonexistent run.
async fn test_get_inference_evaluation_run_metadata_not_found(
    conn: impl EvaluationQueries + TestDatabaseHelpers,
) {
    let results = conn
        .get_inference_evaluation_run_metadata(&[Uuid::now_v7()])
        .await
        .expect("query should succeed even for nonexistent run");

    assert!(
        results.is_empty(),
        "should return empty for nonexistent run"
    );
}
make_db_test!(test_get_inference_evaluation_run_metadata_not_found);

// ============================================================================
// get_evaluation_run_infos tests
// ============================================================================

/// Test that get_evaluation_run_infos returns correct run infos for multiple evaluation runs.
async fn test_get_evaluation_run_infos_multiple_runs(conn: impl EvaluationQueries) {
    let evaluation_run_id1 =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos(
            &[evaluation_run_id1, evaluation_run_id2],
            "extract_entities",
        )
        .await
        .unwrap();

    assert_eq!(run_infos.len(), 2, "Expected 2 evaluation run infos");

    // Results are ordered by evaluation_run_id DESC
    let first = &run_infos[0];
    assert_eq!(first.evaluation_run_id, evaluation_run_id1);
    assert_eq!(first.variant_name, "gpt4o_mini_initial_prompt");

    let second = &run_infos[1];
    assert_eq!(second.evaluation_run_id, evaluation_run_id2);
    assert_eq!(second.variant_name, "gpt4o_initial_prompt");
}
make_db_test!(test_get_evaluation_run_infos_multiple_runs);

/// Test that get_evaluation_run_infos returns correct info for a single evaluation run.
async fn test_get_evaluation_run_infos_single_run(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos(&[evaluation_run_id], "extract_entities")
        .await
        .unwrap();

    assert_eq!(run_infos.len(), 1, "Expected 1 evaluation run info");
    assert_eq!(run_infos[0].evaluation_run_id, evaluation_run_id);
    assert_eq!(run_infos[0].variant_name, "gpt4o_mini_initial_prompt");
}
make_db_test!(test_get_evaluation_run_infos_single_run);

/// Test that get_evaluation_run_infos returns empty for nonexistent run IDs.
async fn test_get_evaluation_run_infos_nonexistent_run(conn: impl EvaluationQueries) {
    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos(&[nonexistent_id], "extract_entities")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for nonexistent run"
    );
}
make_db_test!(test_get_evaluation_run_infos_nonexistent_run);

/// Test that get_evaluation_run_infos returns empty when function name doesn't match.
async fn test_get_evaluation_run_infos_wrong_function(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos(&[evaluation_run_id], "nonexistent_function")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for wrong function"
    );
}
make_db_test!(test_get_evaluation_run_infos_wrong_function);

/// Test that get_evaluation_run_infos returns empty for empty input.
async fn test_get_evaluation_run_infos_empty_input(conn: impl EvaluationQueries) {
    let run_infos = conn
        .get_evaluation_run_infos(&[], "extract_entities")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for empty input"
    );
}
make_db_test!(test_get_evaluation_run_infos_empty_input);

// ============================================================================
// get_evaluation_run_infos_for_datapoint tests
// ============================================================================

/// Test that get_evaluation_run_infos_for_datapoint returns correct info for a JSON function datapoint.
async fn test_get_evaluation_run_infos_for_datapoint_json_function(conn: impl EvaluationQueries) {
    // Datapoint ID from the test fixture for extract_entities function
    let datapoint_id = Uuid::parse_str("0196368e-0b64-7321-ab5b-c32eefbf3e9f").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos_for_datapoint(
            &datapoint_id,
            "extract_entities",
            FunctionConfigType::Json,
        )
        .await
        .unwrap();

    assert_eq!(run_infos.len(), 1, "Expected 1 evaluation run info");
    assert_eq!(
        run_infos[0].evaluation_run_id,
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").expect("Valid UUID")
    );
    assert_eq!(run_infos[0].variant_name, "gpt4o_initial_prompt");
}
make_db_test!(test_get_evaluation_run_infos_for_datapoint_json_function);

/// Test that get_evaluation_run_infos_for_datapoint returns correct info for a chat function datapoint.
async fn test_get_evaluation_run_infos_for_datapoint_chat_function(conn: impl EvaluationQueries) {
    // Datapoint ID from the test fixture for write_haiku function
    let datapoint_id = Uuid::parse_str("0196374a-d03f-7420-9da5-1561cba71ddb").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos_for_datapoint(
            &datapoint_id,
            "write_haiku",
            FunctionConfigType::Chat,
        )
        .await
        .unwrap();

    assert_eq!(run_infos.len(), 1, "Expected 1 evaluation run info");
    assert_eq!(
        run_infos[0].evaluation_run_id,
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID")
    );
    assert_eq!(run_infos[0].variant_name, "better_prompt_haiku_4_5");
}
make_db_test!(test_get_evaluation_run_infos_for_datapoint_chat_function);

/// Test that get_evaluation_run_infos_for_datapoint returns empty for nonexistent datapoint.
async fn test_get_evaluation_run_infos_for_datapoint_nonexistent(conn: impl EvaluationQueries) {
    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos_for_datapoint(
            &nonexistent_id,
            "extract_entities",
            FunctionConfigType::Json,
        )
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for nonexistent datapoint"
    );
}
make_db_test!(test_get_evaluation_run_infos_for_datapoint_nonexistent);

/// Test that get_evaluation_run_infos_for_datapoint returns empty when function name doesn't match.
async fn test_get_evaluation_run_infos_for_datapoint_wrong_function(conn: impl EvaluationQueries) {
    // Use a valid datapoint ID but with wrong function name
    let datapoint_id = Uuid::parse_str("0196368e-0b64-7321-ab5b-c32eefbf3e9f").expect("Valid UUID");

    let run_infos = conn
        .get_evaluation_run_infos_for_datapoint(
            &datapoint_id,
            "nonexistent_function",
            FunctionConfigType::Json,
        )
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for wrong function"
    );
}
make_db_test!(test_get_evaluation_run_infos_for_datapoint_wrong_function);

// ============================================================================
// count_datapoints_for_evaluation tests
// ============================================================================

/// Test using the shared test database with pre-existing fixture data.
/// This test verifies that the correct number of datapoints is returned for a chat function evaluation.
/// The count should not include data that was created after the evaluation run.
async fn test_count_datapoints_for_haiku_evaluation(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("01963690-dff2-7cd3-b724-62fb705772a1").expect("Valid UUID");

    let count = conn
        .count_datapoints_for_evaluation("write_haiku", &[evaluation_run_id])
        .await
        .unwrap();

    // This should not include data that is after the evaluation run
    assert_eq!(
        count, 77,
        "Expected 77 datapoints for haiku evaluation, got {count}"
    );
}
make_db_test!(test_count_datapoints_for_haiku_evaluation);

/// Test using the shared test database with pre-existing fixture data.
/// This test verifies that the correct number of datapoints is returned for a JSON function evaluation.
async fn test_count_datapoints_for_entity_extraction_evaluation(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let count = conn
        .count_datapoints_for_evaluation("extract_entities", &[evaluation_run_id])
        .await
        .unwrap();

    assert_eq!(
        count, 41,
        "Expected 41 datapoints for entity_extraction evaluation, got {count}"
    );
}
make_db_test!(test_count_datapoints_for_entity_extraction_evaluation);

// ============================================================================
// get_evaluation_statistics tests
// ============================================================================

/// Test that get_evaluation_statistics returns correct statistics for a JSON function evaluation.
async fn test_get_evaluation_statistics_json_function(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports".to_string(),
    ];

    let statistics = conn
        .get_evaluation_statistics(
            "extract_entities",
            FunctionConfigType::Json,
            &metric_names,
            &[evaluation_run_id],
        )
        .await
        .unwrap();

    // Should have statistics for the metrics that have feedback
    assert!(
        !statistics.is_empty(),
        "Expected at least one statistics entry"
    );

    // Verify structure of returned statistics
    for stat in &statistics {
        assert_eq!(stat.evaluation_run_id, evaluation_run_id);
        assert!(metric_names.contains(&stat.metric_name));
        assert!(stat.datapoint_count > 0);
        // mean_metric should be between 0 and 1 for boolean metrics
        assert!((0.0..=1.0).contains(&stat.mean_metric) || stat.mean_metric > 1.0);
    }
}
make_db_test!(test_get_evaluation_statistics_json_function);

/// Test that get_evaluation_statistics returns correct statistics for multiple evaluation runs.
async fn test_get_evaluation_statistics_multiple_runs(conn: impl EvaluationQueries) {
    let evaluation_run_id1 =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = conn
        .get_evaluation_statistics(
            "extract_entities",
            FunctionConfigType::Json,
            &metric_names,
            &[evaluation_run_id1, evaluation_run_id2],
        )
        .await
        .unwrap();

    // At least one run should have statistics
    assert!(
        !statistics.is_empty(),
        "Expected statistics for at least one run"
    );
}
make_db_test!(test_get_evaluation_statistics_multiple_runs);

/// Test that get_evaluation_statistics returns empty for nonexistent evaluation runs.
async fn test_get_evaluation_statistics_nonexistent_run(conn: impl EvaluationQueries) {
    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = conn
        .get_evaluation_statistics(
            "extract_entities",
            FunctionConfigType::Json,
            &metric_names,
            &[nonexistent_id],
        )
        .await
        .unwrap();

    assert_eq!(
        statistics.len(),
        0,
        "Expected 0 statistics for nonexistent run"
    );
}
make_db_test!(test_get_evaluation_statistics_nonexistent_run);

/// Test that get_evaluation_statistics returns empty for nonexistent metrics.
async fn test_get_evaluation_statistics_nonexistent_metric(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec!["nonexistent_metric".to_string()];

    let statistics = conn
        .get_evaluation_statistics(
            "extract_entities",
            FunctionConfigType::Json,
            &metric_names,
            &[evaluation_run_id],
        )
        .await
        .unwrap();

    assert_eq!(
        statistics.len(),
        0,
        "Expected 0 statistics for nonexistent metric"
    );
}
make_db_test!(test_get_evaluation_statistics_nonexistent_metric);

/// Test that get_evaluation_statistics returns empty for wrong function name.
async fn test_get_evaluation_statistics_wrong_function(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = conn
        .get_evaluation_statistics(
            "nonexistent_function",
            FunctionConfigType::Json,
            &metric_names,
            &[evaluation_run_id],
        )
        .await
        .unwrap();

    assert_eq!(
        statistics.len(),
        0,
        "Expected 0 statistics for wrong function"
    );
}
make_db_test!(test_get_evaluation_statistics_wrong_function);

/// Test that get_evaluation_statistics handles chat function type correctly.
async fn test_get_evaluation_statistics_chat_function(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID");

    // This evaluation uses the write_haiku function which is a chat function
    let metric_names =
        vec!["tensorzero::evaluation_name::haiku_eval::evaluator_name::haiku_score".to_string()];

    let statistics = conn
        .get_evaluation_statistics(
            "write_haiku",
            FunctionConfigType::Chat,
            &metric_names,
            &[evaluation_run_id],
        )
        .await
        .unwrap();

    // Verify the query executed successfully (may or may not have data depending on fixtures)
    // The main goal is to ensure the chat function path works
    for stat in &statistics {
        assert_eq!(stat.evaluation_run_id, evaluation_run_id);
    }
}
make_db_test!(test_get_evaluation_statistics_chat_function);

// ============================================================================
// get_evaluation_results tests
// ============================================================================

/// Test that get_evaluation_results returns correct results for haiku evaluation.
async fn test_get_evaluation_results_haiku(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &[
                "tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                    .to_string(),
            ],
            None,
            5,
            0,
        )
        .await
        .unwrap();

    // Verify we get the expected number of results (5 datapoints * 2 metrics = 10)
    assert_eq!(
        results.len(),
        10,
        "Expected 10 results (5 datapoints * 2 metrics)"
    );

    // Verify all results belong to the correct evaluation run
    for result in &results {
        match result {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.evaluation_run_id, evaluation_run_id);
                assert_eq!(row.variant_name, "better_prompt_haiku_4_5");
            }
            EvaluationResultRow::Json(_) => panic!("Expected Chat result"),
        }
    }

    // Verify we have both metric types
    let metric_names: std::collections::HashSet<_> = results
        .iter()
        .filter_map(|r| match r {
            EvaluationResultRow::Chat(row) => row.metric_name.as_deref(),
            EvaluationResultRow::Json(row) => row.metric_name.as_deref(),
        })
        .collect();
    assert!(
        metric_names.contains("tensorzero::evaluation_name::haiku::evaluator_name::exact_match"),
        "Should have exact_match metric"
    );
    assert!(
        metric_names
            .contains("tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"),
        "Should have topic_starts_with_f metric"
    );

    // Verify datapoint count (should be 5 unique datapoints)
    let datapoint_ids: std::collections::HashSet<_> = results
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();
    assert_eq!(datapoint_ids.len(), 5, "Expected 5 unique datapoints");
}
make_db_test!(test_get_evaluation_results_haiku);

/// Test that get_evaluation_results handles entity_extraction (JSON function) correctly.
async fn test_get_evaluation_results_entity_extraction(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "extract_entities",
            &[evaluation_run_id],
            FunctionConfigType::Json,
            &[
                "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
                    .to_string(),
                "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
                    .to_string(),
            ],
            None,
            2,
            0,
        )
        .await
        .unwrap();

    // Verify we get 4 results (2 datapoints * 2 metrics)
    assert_eq!(
        results.len(),
        4,
        "Expected 4 results (2 datapoints * 2 metrics)"
    );

    // Verify we have both metrics
    let metric_names: std::collections::HashSet<_> = results
        .iter()
        .filter_map(|r| match r {
            EvaluationResultRow::Chat(row) => row.metric_name.as_deref(),
            EvaluationResultRow::Json(row) => row.metric_name.as_deref(),
        })
        .collect();
    assert!(
        metric_names.contains(
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
        ),
        "Should have exact_match metric"
    );
    assert!(
        metric_names.contains(
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
        ),
        "Should have count_sports metric"
    );

    // Verify datapoint count
    let datapoint_ids: std::collections::HashSet<_> = results
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();
    assert_eq!(datapoint_ids.len(), 2, "Expected 2 unique datapoints");

    // Verify results are Json type
    for result in &results {
        let EvaluationResultRow::Json(_) = result else {
            panic!("Expected Json result, got {result:?}");
        };
    }
}
make_db_test!(test_get_evaluation_results_entity_extraction);

/// Test that get_evaluation_results handles multiple evaluation runs (ragged case).
async fn test_get_evaluation_results_multiple_runs(conn: impl EvaluationQueries) {
    let evaluation_run_id1 =
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id1, evaluation_run_id2],
            FunctionConfigType::Chat,
            &[
                "tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                    .to_string(),
            ],
            None,
            5,
            0,
        )
        .await
        .unwrap();

    // With ragged data: 5 datapoints * 2 metrics * 2 runs - some missing = 18
    // (one datapoint is after all runs, another only in one run)
    assert_eq!(
        results.len(),
        18,
        "Expected 18 results for ragged evaluation"
    );

    // Verify both evaluation runs are present
    let eval_run_ids: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.evaluation_run_id,
            EvaluationResultRow::Json(row) => row.evaluation_run_id,
        })
        .collect();
    assert!(
        eval_run_ids.contains(&evaluation_run_id1),
        "Should have results from first evaluation run"
    );
    assert!(
        eval_run_ids.contains(&evaluation_run_id2),
        "Should have results from second evaluation run"
    );

    // Verify datapoint count
    let datapoint_ids: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();
    assert_eq!(datapoint_ids.len(), 5, "Expected 5 unique datapoints");
}
make_db_test!(test_get_evaluation_results_multiple_runs);

/// Test that get_evaluation_results returns empty for nonexistent function.
async fn test_get_evaluation_results_nonexistent_function(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "nonexistent_function",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["some_metric".to_string()],
            None,
            100,
            0,
        )
        .await
        .unwrap();

    assert!(
        results.is_empty(),
        "Expected no results for nonexistent function"
    );
}
make_db_test!(test_get_evaluation_results_nonexistent_function);

/// Test that get_evaluation_results respects pagination offset.
async fn test_get_evaluation_results_pagination(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    // Get first page (5 datapoints)
    let first_page = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            None,
            5,
            0,
        )
        .await
        .unwrap();

    // Get second page (5 datapoints starting from offset 5)
    let second_page = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            None,
            5,
            5,
        )
        .await
        .unwrap();

    // Verify we got results on both pages
    assert_eq!(first_page.len(), 5, "First page should have 5 results");
    assert_eq!(second_page.len(), 5, "Second page should have 5 results");

    // Verify no overlap between pages
    let first_datapoints: std::collections::HashSet<_> = first_page
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();
    let second_datapoints: std::collections::HashSet<_> = second_page
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();

    let overlap: Vec<_> = first_datapoints.intersection(&second_datapoints).collect();
    assert!(
        overlap.is_empty(),
        "Pages should not have overlapping datapoints"
    );
}
make_db_test!(test_get_evaluation_results_pagination);

/// Test that get_evaluation_results with datapoint_id filter returns results for only that datapoint.
async fn test_get_evaluation_results_with_datapoint_id_filter(conn: impl EvaluationQueries) {
    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    // First get all results without a filter to find a valid datapoint_id
    let all_results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            None,
            1,
            0,
        )
        .await
        .unwrap();

    assert!(!all_results.is_empty(), "Need at least one result to test");
    let target_datapoint_id = all_results[0].datapoint_id();

    // Now filter by that specific datapoint_id
    let filtered_results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            Some(&target_datapoint_id),
            u32::MAX,
            0,
        )
        .await
        .unwrap();

    // All results should be for the filtered datapoint
    assert!(
        !filtered_results.is_empty(),
        "Should have results for the datapoint"
    );
    for result in &filtered_results {
        assert_eq!(
            result.datapoint_id(),
            target_datapoint_id,
            "All results should be for the filtered datapoint"
        );
    }
}
make_db_test!(test_get_evaluation_results_with_datapoint_id_filter);

/// Test that get_evaluation_results with datapoint_id filter returns empty for nonexistent datapoint.
async fn test_get_evaluation_results_with_datapoint_id_filter_nonexistent(
    conn: impl EvaluationQueries,
) {
    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");
    let nonexistent_datapoint_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            Some(&nonexistent_datapoint_id),
            u32::MAX,
            0,
        )
        .await
        .unwrap();

    assert!(
        results.is_empty(),
        "Should have no results for nonexistent datapoint"
    );
}
make_db_test!(test_get_evaluation_results_with_datapoint_id_filter_nonexistent);

/// Test get_evaluation_results for a specific chat datapoint with detailed assertions.
/// This mirrors the TypeScript test "should return correct array for chat datapoint".
async fn test_get_evaluation_results_chat_datapoint_details(conn: impl EvaluationQueries) {
    let datapoint_id = Uuid::parse_str("0196374a-d03f-7420-9da5-1561cba71ddb").expect("Valid UUID");
    let evaluation_run_id =
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &[
                "tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                    .to_string(),
            ],
            Some(&datapoint_id),
            u32::MAX,
            0,
        )
        .await
        .unwrap();

    // Should have 2 results (1 datapoint * 2 metrics)
    assert_eq!(results.len(), 2, "Expected 2 results for chat datapoint");

    // Extract chat results
    let chat_results: Vec<&ChatEvaluationResultRow> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row,
            EvaluationResultRow::Json(_) => panic!("Expected Chat result"),
        })
        .collect();

    // Verify all results are for the correct datapoint and evaluation run
    for result in &chat_results {
        assert_eq!(result.datapoint_id, datapoint_id);
        assert_eq!(result.evaluation_run_id, evaluation_run_id);
        assert_eq!(result.variant_name, "better_prompt_haiku_4_5");
    }

    // Verify we have both metrics
    let metric_names: std::collections::HashSet<_> = chat_results
        .iter()
        .filter_map(|r| r.metric_name.as_deref())
        .collect();
    assert!(
        metric_names.contains("tensorzero::evaluation_name::haiku::evaluator_name::exact_match"),
        "Should have exact_match metric"
    );
    assert!(
        metric_names
            .contains("tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"),
        "Should have topic_starts_with_f metric"
    );

    // Verify the exact_match metric value is "true"
    let exact_match_result = chat_results
        .iter()
        .find(|r| {
            r.metric_name.as_ref()
                == Some(
                    &"tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                )
        })
        .expect("Should have exact_match result");
    assert_eq!(
        exact_match_result.metric_value.as_deref(),
        Some("true"),
        "exact_match metric value should be 'true'"
    );

    // Verify the topic_starts_with_f metric value is "false"
    let topic_result = chat_results
        .iter()
        .find(|r| {
            r.metric_name.as_ref()
                == Some(
                    &"tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                        .to_string(),
                )
        })
        .expect("Should have topic_starts_with_f result");
    assert_eq!(
        topic_result.metric_value.as_deref(),
        Some("false"),
        "topic_starts_with_f metric value should be 'false'"
    );

    // Verify input contains the expected topic
    let input_json = serde_json::to_string(&exact_match_result.input).unwrap();
    assert!(
        input_json.contains("sheet"),
        "Input should contain the topic 'sheet'"
    );

    // Verify generated output contains expected text
    let output_json = serde_json::to_string(&exact_match_result.generated_output).unwrap();
    assert!(
        output_json.contains("Swallowing moonlight"),
        "Generated output should contain 'Swallowing moonlight'"
    );

    // Verify usage fields are populated (aggregated from model_inferences via GROUP BY)
    // The same inference backs both metric rows, so usage should be identical across metrics
    for result in &chat_results {
        assert!(
            result.input_tokens.is_some(),
            "input_tokens should be populated from model_inferences"
        );
        assert!(
            result.output_tokens.is_some(),
            "output_tokens should be populated from model_inferences"
        );
        assert!(
            result.processing_time_ms.is_some(),
            "processing_time_ms should be populated from the inference table"
        );
        // input_tokens and output_tokens should be positive (SUM of model_inferences)
        assert!(
            result.input_tokens.unwrap() > 0,
            "input_tokens should be positive, got {}",
            result.input_tokens.unwrap()
        );
        assert!(
            result.output_tokens.unwrap() > 0,
            "output_tokens should be positive, got {}",
            result.output_tokens.unwrap()
        );
    }
    // Both metric rows come from the same inference, so usage values must match
    assert_eq!(
        chat_results[0].input_tokens, chat_results[1].input_tokens,
        "input_tokens should be the same across metrics for the same inference"
    );
    assert_eq!(
        chat_results[0].output_tokens, chat_results[1].output_tokens,
        "output_tokens should be the same across metrics for the same inference"
    );
    assert_eq!(
        chat_results[0].processing_time_ms, chat_results[1].processing_time_ms,
        "processing_time_ms should be the same across metrics for the same inference"
    );
}
make_db_test!(test_get_evaluation_results_chat_datapoint_details);

/// Test get_evaluation_results for a specific JSON datapoint with detailed assertions.
/// This mirrors the TypeScript test "should return correct array for json datapoint".
async fn test_get_evaluation_results_json_datapoint_details(conn: impl EvaluationQueries) {
    let datapoint_id = Uuid::parse_str("0193994e-5560-7610-a3a0-45fdd59338aa").expect("Valid UUID");
    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let results = conn
        .get_evaluation_results(
            "extract_entities",
            &[evaluation_run_id],
            FunctionConfigType::Json,
            &[
                "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
                    .to_string(),
                "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
                    .to_string(),
            ],
            Some(&datapoint_id),
            u32::MAX,
            0,
        )
        .await
        .unwrap();

    // Should have 2 results (1 datapoint * 2 metrics)
    assert_eq!(results.len(), 2, "Expected 2 results for JSON datapoint");

    // Extract JSON results
    let json_results: Vec<&JsonEvaluationResultRow> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Json(row) => row,
            EvaluationResultRow::Chat(_) => panic!("Expected Json result"),
        })
        .collect();

    // Verify all results are for the correct datapoint and evaluation run
    for result in &json_results {
        assert_eq!(result.datapoint_id, datapoint_id);
        assert_eq!(result.evaluation_run_id, evaluation_run_id);
        assert_eq!(result.variant_name, "gpt4o_mini_initial_prompt");
    }

    // Verify we have both metrics
    let metric_names: std::collections::HashSet<_> = json_results
        .iter()
        .filter_map(|r| r.metric_name.as_deref())
        .collect();
    assert!(
        metric_names.contains(
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
        ),
        "Should have exact_match metric"
    );
    assert!(
        metric_names.contains(
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
        ),
        "Should have count_sports metric"
    );

    // Verify metric values are defined
    for result in &json_results {
        assert!(
            result.metric_value.is_some(),
            "Metric value should be defined"
        );
    }

    // Verify JSON structure of input and output
    for result in &json_results {
        let input_json = serde_json::to_string(&result.input).unwrap();
        assert!(input_json.starts_with('{'), "Input should be JSON object");
        let output_json = serde_json::to_string(&result.generated_output).unwrap();
        assert!(
            output_json.contains("\"raw\""),
            "Generated output should have 'raw' field"
        );
    }

    // Verify usage fields are populated (aggregated from model_inferences via GROUP BY)
    for result in &json_results {
        assert!(
            result.input_tokens.is_some(),
            "input_tokens should be populated from model_inferences"
        );
        assert!(
            result.output_tokens.is_some(),
            "output_tokens should be populated from model_inferences"
        );
        assert!(
            result.processing_time_ms.is_some(),
            "processing_time_ms should be populated from the inference table"
        );
        assert!(
            result.input_tokens.unwrap() > 0,
            "input_tokens should be positive, got {}",
            result.input_tokens.unwrap()
        );
        assert!(
            result.output_tokens.unwrap() > 0,
            "output_tokens should be positive, got {}",
            result.output_tokens.unwrap()
        );
    }
    // Both metric rows come from the same inference, so usage values must match
    assert_eq!(
        json_results[0].input_tokens, json_results[1].input_tokens,
        "input_tokens should be the same across metrics for the same inference"
    );
    assert_eq!(
        json_results[0].output_tokens, json_results[1].output_tokens,
        "output_tokens should be the same across metrics for the same inference"
    );
    assert_eq!(
        json_results[0].processing_time_ms, json_results[1].processing_time_ms,
        "processing_time_ms should be the same across metrics for the same inference"
    );
}
make_db_test!(test_get_evaluation_results_json_datapoint_details);

// ============================================================================
// search_evaluation_runs tests
// ============================================================================

/// Test that search_evaluation_runs returns all runs for an evaluation with an empty query.
async fn test_search_evaluation_runs_empty_query(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "",
            100,
            0,
        )
        .await
        .unwrap();

    assert!(results.len() > 1, "There should be at lease 2 results");
    assert!(
        results[0].evaluation_run_id > results[1].evaluation_run_id,
        "Result IDs should be ordered DESC"
    );
}
make_db_test!(test_search_evaluation_runs_empty_query);

/// Test that search_evaluation_runs filters by variant name substring.
async fn test_search_evaluation_runs_by_variant_name(conn: impl EvaluationQueries) {
    // "mini" should only match the gpt4o_mini_initial_prompt variant
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "mini",
            100,
            0,
        )
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "Expected some evaluation runs matching 'mini'"
    );
    assert_eq!(results[0].variant_name, "gpt4o_mini_initial_prompt");
}
make_db_test!(test_search_evaluation_runs_by_variant_name);

/// Test that search_evaluation_runs matches both runs when query matches a common variant substring.
async fn test_search_evaluation_runs_common_variant_substring(conn: impl EvaluationQueries) {
    // "gpt4o" should match both variants
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "gpt4o",
            100,
            0,
        )
        .await
        .unwrap();

    assert!(
        results.len() > 1,
        "Expected some evaluation runs matching 'gpt4o'"
    );
}
make_db_test!(test_search_evaluation_runs_common_variant_substring);

/// Test that search_evaluation_runs is case-insensitive.
async fn test_search_evaluation_runs_case_insensitive(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "GPT4O_MINI",
            100,
            0,
        )
        .await
        .unwrap();

    assert!(
        results.len() > 1,
        "Case-insensitive search should match 'GPT4O_MINI' to gpt4o_mini_initial_prompt"
    );
    assert_eq!(results[0].variant_name, "gpt4o_mini_initial_prompt");
}
make_db_test!(test_search_evaluation_runs_case_insensitive);

/// Test that search_evaluation_runs can match by evaluation_run_id substring.
async fn test_search_evaluation_runs_by_run_id(conn: impl EvaluationQueries) {
    // "19bd" is a substring of "0196368f-19bd-7082-a677-1c0bf346ff24" but not of the other run ID
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "19bd",
            100,
            0,
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        1,
        "Expected 1 evaluation run matching '19bd'"
    );
    assert_eq!(
        results[0].evaluation_run_id,
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID"),
    );
}
make_db_test!(test_search_evaluation_runs_by_run_id);

/// Test that search_evaluation_runs returns empty when query matches nothing.
async fn test_search_evaluation_runs_no_match(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "zzz_nonexistent_zzz",
            100,
            0,
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for a query that matches nothing"
    );
}
make_db_test!(test_search_evaluation_runs_no_match);

/// Test that search_evaluation_runs returns empty for a nonexistent evaluation name.
async fn test_search_evaluation_runs_wrong_evaluation_name(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("nonexistent_evaluation"),
            Some("extract_entities"),
            "",
            100,
            0,
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for nonexistent evaluation name"
    );
}
make_db_test!(test_search_evaluation_runs_wrong_evaluation_name);

/// Test that search_evaluation_runs returns empty for a nonexistent function name.
async fn test_search_evaluation_runs_wrong_function_name(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("nonexistent_function"),
            "",
            100,
            0,
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for nonexistent function name"
    );
}
make_db_test!(test_search_evaluation_runs_wrong_function_name);

/// Test that search_evaluation_runs works without a function_name filter.
async fn test_search_evaluation_runs_no_function_name(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(Some("entity_extraction"), None, "", 100, 0)
        .await
        .unwrap();

    assert!(
        results.len() > 1,
        "Should return results even without function_name filter"
    );
    assert!(
        results[0].evaluation_run_id > results[1].evaluation_run_id,
        "Result IDs should be ordered DESC"
    );
}
make_db_test!(test_search_evaluation_runs_no_function_name);

/// Test that search_evaluation_runs works with only a function_name filter.
async fn test_search_evaluation_runs_function_only(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(None, Some("extract_entities"), "", 100, 0)
        .await
        .unwrap();

    assert!(
        results.len() > 1,
        "Should return results when filtering only by function_name"
    );
    assert!(
        results
            .iter()
            .all(|result| !result.evaluation_name.is_empty()),
        "Stored runs should always have a human-readable evaluation_name"
    );
}
make_db_test!(test_search_evaluation_runs_function_only);

/// Test that search_evaluation_runs can match evaluation_name and dataset_name substrings.
async fn test_search_evaluation_runs_by_evaluation_or_dataset_name(conn: impl EvaluationQueries) {
    let evaluation_name_results = conn
        .search_evaluation_runs(None, Some("extract_entities"), "entity_extraction", 100, 0)
        .await
        .unwrap();
    let dataset_name_results = conn
        .search_evaluation_runs(None, Some("extract_entities"), "entity_extraction", 100, 0)
        .await
        .unwrap();

    assert!(
        !evaluation_name_results.is_empty(),
        "Expected evaluation_name substring search to return results"
    );
    assert!(
        !dataset_name_results.is_empty(),
        "Expected dataset_name substring search to return results"
    );
}
make_db_test!(test_search_evaluation_runs_by_evaluation_or_dataset_name);

/// Test that search_evaluation_runs respects the limit parameter.
async fn test_search_evaluation_runs_with_limit(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "",
            1,
            0,
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1, "Expected 1 result with limit=1");
}
make_db_test!(test_search_evaluation_runs_with_limit);

/// Test that search_evaluation_runs respects the offset parameter.
async fn test_search_evaluation_runs_with_offset(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "",
            100,
            1,
        )
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "Expected >1 results with offset=1 (skipping first)"
    );
}
make_db_test!(test_search_evaluation_runs_with_offset);

/// Test that search_evaluation_runs returns empty when offset is beyond all results.
async fn test_search_evaluation_runs_offset_beyond_results(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs(
            Some("entity_extraction"),
            Some("extract_entities"),
            "",
            100,
            100,
        )
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results when offset is beyond all available runs"
    );
}
make_db_test!(test_search_evaluation_runs_offset_beyond_results);

// ============================================================================
// get_evaluation_results usage aggregation tests (GROUP BY)
// ============================================================================

/// Test that get_evaluation_results correctly aggregates usage fields (input_tokens,
/// output_tokens, cost) from multiple model inferences per inference via GROUP BY.
///
/// This test dynamically inserts:
/// 1. Two datapoints
/// 2. An evaluation run
/// 3. Two chat inferences (one per datapoint), each with tags linking to the eval run + datapoint
/// 4. Multiple model inferences per inference (to exercise the GROUP BY SUM aggregation)
/// 5. Boolean feedback for each inference
///
/// Then it calls `get_evaluation_results` and verifies:
/// - Usage fields are correctly summed across model inferences
/// - Cost is NULL when any model inference for that inference lacks cost
/// - Different inferences have different aggregated values
async fn test_get_evaluation_results_usage_aggregation(
    conn: impl EvaluationQueries
    + InferenceQueries
    + ModelInferenceQueries
    + DatasetQueries
    + FeedbackQueries
    + TestDatabaseHelpers,
) {
    let run_id = Uuid::now_v7();
    let evaluation_name = format!("e2e_usage_agg_{run_id}");
    let dataset_name = format!("e2e_usage_agg_dataset_{run_id}");
    let function_name = format!("e2e_usage_agg_func_{run_id}");
    let metric_name =
        format!("tensorzero::evaluation_name::{evaluation_name}::evaluator_name::exact_match");

    // --- 1. Insert datapoints ---
    let datapoint1_id = Uuid::now_v7();
    let datapoint2_id = Uuid::now_v7();

    let datapoints = vec![
        StoredDatapoint::Chat(StoredChatInferenceDatapoint {
            dataset_name: dataset_name.clone(),
            function_name: function_name.clone(),
            id: datapoint1_id,
            episode_id: None,
            input: StoredInput::default(),
            output: None,
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            tags: None,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
            name: None,
            snapshot_hash: None,
            is_deleted: false,
            auxiliary: String::new(),
            updated_at: String::new(),
        }),
        StoredDatapoint::Chat(StoredChatInferenceDatapoint {
            dataset_name: dataset_name.clone(),
            function_name: function_name.clone(),
            id: datapoint2_id,
            episode_id: None,
            input: StoredInput::default(),
            output: None,
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            tags: None,
            is_custom: false,
            source_inference_id: None,
            staled_at: None,
            name: None,
            snapshot_hash: None,
            is_deleted: false,
            auxiliary: String::new(),
            updated_at: String::new(),
        }),
    ];
    conn.insert_datapoints(&datapoints)
        .await
        .expect("insert datapoints should succeed");

    // --- 2. Insert evaluation run ---
    let eval_run = InferenceEvaluationRunInsert {
        run_id,
        evaluation_name: evaluation_name.clone(),
        function_name: function_name.clone(),
        function_type: FunctionConfigType::Chat,
        dataset_name: dataset_name.clone(),
        variant_names: vec!["test_variant".to_string()],
        metrics: vec![InferenceEvaluationRunMetricMetadata {
            name: metric_name.clone(),
            evaluator_name: Some("exact_match".to_string()),
            value_type: "boolean".to_string(),
            optimize: Some(MetricConfigOptimize::Max),
        }],
        source: InferenceEvaluationRunSource::DatasetName,
        snapshot_hash: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
    };
    conn.insert_inference_evaluation_run(&eval_run)
        .await
        .expect("insert evaluation run should succeed");

    // --- 3. Insert chat inferences with tags ---
    let inference1_id = Uuid::now_v7();
    let inference2_id = Uuid::now_v7();

    let make_tags = |datapoint_id: &Uuid| -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert(
            "tensorzero::evaluation_run_id".to_string(),
            run_id.to_string(),
        );
        tags.insert(
            "tensorzero::datapoint_id".to_string(),
            datapoint_id.to_string(),
        );
        tags.insert("tensorzero::dataset_name".to_string(), dataset_name.clone());
        tags.insert(
            "tensorzero::evaluation_name".to_string(),
            evaluation_name.clone(),
        );
        tags
    };

    let inferences = vec![
        ChatInferenceDatabaseInsert {
            id: inference1_id,
            function_name: function_name.clone(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            input: Some(StoredInput::default()),
            output: Some(vec![]),
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            inference_params: Some(InferenceParams::default()),
            processing_time_ms: Some(150),
            ttft_ms: None,
            tags: make_tags(&datapoint1_id),
            extra_body: Some(UnfilteredInferenceExtraBody::default()),
            snapshot_hash: None,
        },
        ChatInferenceDatabaseInsert {
            id: inference2_id,
            function_name: function_name.clone(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            input: Some(StoredInput::default()),
            output: Some(vec![]),
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            inference_params: Some(InferenceParams::default()),
            processing_time_ms: Some(300),
            ttft_ms: None,
            tags: make_tags(&datapoint2_id),
            extra_body: Some(UnfilteredInferenceExtraBody::default()),
            snapshot_hash: None,
        },
    ];
    conn.insert_chat_inferences(&inferences)
        .await
        .expect("insert chat inferences should succeed");

    // --- 4. Insert multiple model inferences per inference ---
    // Inference 1: two model inferences, both with cost => cost should be summed
    // Inference 2: two model inferences, one without cost => cost should be NULL
    let model_inferences = vec![
        // Inference 1 - model inference A
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: inference1_id,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(100),
            output_tokens: Some(40),
            response_time_ms: Some(50),
            model_name: "model-a".to_string(),
            model_provider_name: "provider-a".to_string(),
            ttft_ms: None,
            cached: false,
            cost: Some(Decimal::new(15, 4)), // 0.0015
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            timestamp: None,
        },
        // Inference 1 - model inference B (fallback)
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: inference1_id,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(200),
            output_tokens: Some(60),
            response_time_ms: Some(100),
            model_name: "model-b".to_string(),
            model_provider_name: "provider-b".to_string(),
            ttft_ms: None,
            cached: false,
            cost: Some(Decimal::new(25, 4)), // 0.0025
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            timestamp: None,
        },
        // Inference 2 - model inference A (has cost)
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: inference2_id,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(500),
            output_tokens: Some(100),
            response_time_ms: Some(200),
            model_name: "model-a".to_string(),
            model_provider_name: "provider-a".to_string(),
            ttft_ms: None,
            cached: false,
            cost: Some(Decimal::new(50, 4)), // 0.0050
            finish_reason: Some(FinishReason::Stop),
            snapshot_hash: None,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            timestamp: None,
        },
        // Inference 2 - model inference B (NO cost => total cost should be NULL)
        StoredModelInference {
            id: Uuid::now_v7(),
            inference_id: inference2_id,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            raw_request: Some("{}".to_string()),
            raw_response: Some("{}".to_string()),
            system: None,
            input_messages: Some(vec![]),
            output: Some(vec![]),
            input_tokens: Some(50),
            output_tokens: Some(10),
            response_time_ms: Some(50),
            model_name: "model-c".to_string(),
            model_provider_name: "provider-c".to_string(),
            ttft_ms: None,
            cached: false,
            cost: None, // Missing cost
            finish_reason: None,
            snapshot_hash: None,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
            timestamp: None,
        },
    ];
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("insert model inferences should succeed");

    // --- 5. Insert boolean feedback for each inference ---
    let snapshot_hash = SnapshotHash::new_test();
    conn.insert_boolean_feedback(&BooleanMetricFeedbackInsert {
        id: Uuid::now_v7(),
        target_id: inference1_id,
        metric_name: metric_name.clone(),
        value: true,
        tags: HashMap::new(),
        snapshot_hash: snapshot_hash.clone(),
    })
    .await
    .expect("insert feedback 1 should succeed");

    conn.insert_boolean_feedback(&BooleanMetricFeedbackInsert {
        id: Uuid::now_v7(),
        target_id: inference2_id,
        metric_name: metric_name.clone(),
        value: false,
        tags: HashMap::new(),
        snapshot_hash,
    })
    .await
    .expect("insert feedback 2 should succeed");

    // Wait for writes to be visible (ClickHouse eventual consistency)
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // --- 6. Query and verify ---
    let results = conn
        .get_evaluation_results(
            &function_name,
            &[run_id],
            FunctionConfigType::Chat,
            std::slice::from_ref(&metric_name),
            None,
            10,
            0,
        )
        .await
        .expect("get_evaluation_results should succeed");

    assert_eq!(
        results.len(),
        2,
        "Expected 2 results (2 datapoints * 1 metric)"
    );

    // Extract chat results and sort by datapoint_id for deterministic assertions
    let mut chat_results: Vec<&ChatEvaluationResultRow> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row,
            EvaluationResultRow::Json(_) => panic!("Expected Chat result"),
        })
        .collect();
    chat_results.sort_by_key(|r| r.inference_id);

    // Find the result for inference1 and inference2
    let result1 = chat_results
        .iter()
        .find(|r| r.inference_id == inference1_id)
        .expect("should have result for inference1");
    let result2 = chat_results
        .iter()
        .find(|r| r.inference_id == inference2_id)
        .expect("should have result for inference2");

    // --- Verify inference 1: tokens should be summed (100+200=300 input, 40+60=100 output) ---
    assert_eq!(
        result1.input_tokens,
        Some(300),
        "inference1 input_tokens should be 100+200=300"
    );
    assert_eq!(
        result1.output_tokens,
        Some(100),
        "inference1 output_tokens should be 40+60=100"
    );
    // Both model inferences have cost, so cost should be summed: 0.0015 + 0.0025 = 0.004
    assert!(
        result1.cost.is_some(),
        "inference1 cost should be Some (both model inferences have cost)"
    );
    let cost1 = result1.cost.unwrap();
    assert!(
        (cost1 - 0.004).abs() < 1e-9,
        "inference1 cost should be ~0.004, got {cost1}"
    );
    assert_eq!(
        result1.processing_time_ms,
        Some(150),
        "inference1 processing_time_ms should come from the inference table"
    );

    // --- Verify inference 2: tokens summed (500+50=550 input, 100+10=110 output) ---
    assert_eq!(
        result2.input_tokens,
        Some(550),
        "inference2 input_tokens should be 500+50=550"
    );
    assert_eq!(
        result2.output_tokens,
        Some(110),
        "inference2 output_tokens should be 100+10=110"
    );
    // One model inference has no cost, so cost should be NULL
    assert!(
        result2.cost.is_none(),
        "inference2 cost should be None (one model inference lacks cost)"
    );
    assert_eq!(
        result2.processing_time_ms,
        Some(300),
        "inference2 processing_time_ms should come from the inference table"
    );

    // --- Verify metric values ---
    assert_eq!(
        result1.metric_value.as_deref(),
        Some("true"),
        "inference1 metric_value should be true"
    );
    assert_eq!(
        result2.metric_value.as_deref(),
        Some("false"),
        "inference2 metric_value should be false"
    );
}
make_db_test!(test_get_evaluation_results_usage_aggregation);

// ============================================================================
// get_evaluation_usage_statistics tests
// ============================================================================

/// Helper to create a StoredModelInference with the most common defaults filled in.
fn make_model_inference_for_eval(
    inference_id: Uuid,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cost: Option<Decimal>,
    model_name: &str,
) -> StoredModelInference {
    StoredModelInference {
        id: Uuid::now_v7(),
        inference_id,
        function_name: "test_function".to_string(),
        variant_name: "test_variant".to_string(),
        raw_request: Some("{}".to_string()),
        raw_response: Some("{}".to_string()),
        system: None,
        input_messages: Some(vec![]),
        output: Some(vec![]),
        input_tokens,
        output_tokens,
        response_time_ms: Some(50),
        model_name: model_name.to_string(),
        model_provider_name: format!("{model_name}-provider"),
        ttft_ms: None,
        cached: false,
        cost,
        finish_reason: Some(FinishReason::Stop),
        snapshot_hash: None,
        provider_cache_read_input_tokens: None,
        provider_cache_write_input_tokens: None,
        timestamp: None,
    }
}

/// Parameters for creating a tagged evaluation chat inference.
struct EvalChatInferenceParams<'a> {
    id: Uuid,
    function_name: &'a str,
    run_id: Uuid,
    datapoint_id: Uuid,
    dataset_name: &'a str,
    evaluation_name: &'a str,
    processing_time_ms: Option<u32>,
}

/// Helper to create a ChatInferenceDatabaseInsert tagged for an evaluation run.
fn make_eval_chat_inference(p: EvalChatInferenceParams<'_>) -> ChatInferenceDatabaseInsert {
    let mut tags = HashMap::new();
    tags.insert(
        "tensorzero::evaluation_run_id".to_string(),
        p.run_id.to_string(),
    );
    tags.insert(
        "tensorzero::datapoint_id".to_string(),
        p.datapoint_id.to_string(),
    );
    tags.insert(
        "tensorzero::dataset_name".to_string(),
        p.dataset_name.to_string(),
    );
    tags.insert(
        "tensorzero::evaluation_name".to_string(),
        p.evaluation_name.to_string(),
    );

    ChatInferenceDatabaseInsert {
        id: p.id,
        function_name: p.function_name.to_string(),
        variant_name: "v1".to_string(),
        episode_id: Uuid::now_v7(),
        input: Some(StoredInput::default()),
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        inference_params: Some(InferenceParams::default()),
        processing_time_ms: p.processing_time_ms,
        ttft_ms: None,
        tags,
        extra_body: Some(UnfilteredInferenceExtraBody::default()),
        snapshot_hash: None,
    }
}

/// Helper to create and insert an evaluation run.
fn make_eval_run(
    run_id: Uuid,
    function_name: &str,
    evaluation_name: &str,
    dataset_name: &str,
) -> InferenceEvaluationRunInsert {
    InferenceEvaluationRunInsert {
        run_id,
        evaluation_name: evaluation_name.to_string(),
        function_name: function_name.to_string(),
        function_type: FunctionConfigType::Chat,
        dataset_name: dataset_name.to_string(),
        variant_names: vec!["test_variant".to_string()],
        metrics: vec![InferenceEvaluationRunMetricMetadata {
            name: format!("tensorzero::evaluation_name::{evaluation_name}::evaluator_name::em"),
            evaluator_name: Some("em".to_string()),
            value_type: "boolean".to_string(),
            optimize: Some(MetricConfigOptimize::Max),
        }],
        source: InferenceEvaluationRunSource::DatasetName,
        snapshot_hash: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
    }
}

/// Comprehensive test: two evaluation runs with different cost/token profiles.
///
/// Run 1 has 3 inferences:
///   - inference 1: 2 model inferences, both with cost (tokens and cost should sum)
///   - inference 2: 2 model inferences, one missing cost (per-inference cost = NULL)
///   - inference 3: 1 model inference with all fields set
///
/// Run 2 has 2 inferences:
///   - inference 4: 1 model inference, all fields set
///   - inference 5: 1 model inference, all fields set
///
/// This exercises:
///   - Multi-model-inference aggregation (SUM tokens per inference)
///   - Cost NULL propagation at the per-inference level (COUNT(*) = COUNT(cost))
///   - Cost NULL propagation at the per-run level (run 1 has one inference with NULL cost)
///   - Correct GROUP BY evaluation_run_id (two runs aggregated independently)
///   - avg(processing_time_ms) across inferences in a run
async fn test_get_evaluation_usage_statistics_multi_run(
    conn: impl EvaluationQueries + InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers,
) {
    let run_id1 = Uuid::now_v7();
    let run_id2 = Uuid::now_v7();
    let function_name = format!("e2e_usage_stats_{run_id1}");
    let eval_name = format!("e2e_usage_stats_eval_{run_id1}");
    let dataset_name = format!("e2e_usage_stats_ds_{run_id1}");

    // --- Insert two eval runs ---
    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id1,
        &function_name,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run 1");

    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id2,
        &function_name,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run 2");

    // --- Insert inferences ---
    let dp1 = Uuid::now_v7();
    let dp2 = Uuid::now_v7();
    let dp3 = Uuid::now_v7();
    let dp4 = Uuid::now_v7();
    let dp5 = Uuid::now_v7();
    let inf1 = Uuid::now_v7();
    let inf2 = Uuid::now_v7();
    let inf3 = Uuid::now_v7();
    let inf4 = Uuid::now_v7();
    let inf5 = Uuid::now_v7();

    let inferences = vec![
        // Run 1
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf1,
            function_name: &function_name,
            run_id: run_id1,
            datapoint_id: dp1,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(100),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf2,
            function_name: &function_name,
            run_id: run_id1,
            datapoint_id: dp2,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(200),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf3,
            function_name: &function_name,
            run_id: run_id1,
            datapoint_id: dp3,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(300),
        }),
        // Run 2
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf4,
            function_name: &function_name,
            run_id: run_id2,
            datapoint_id: dp4,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(500),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf5,
            function_name: &function_name,
            run_id: run_id2,
            datapoint_id: dp5,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(700),
        }),
    ];
    conn.insert_chat_inferences(&inferences)
        .await
        .expect("insert chat inferences");

    // --- Insert model inferences ---
    let model_inferences = vec![
        // inference 1 (run 1): 2 model calls, both with cost
        make_model_inference_for_eval(
            inf1,
            Some(100),
            Some(40),
            Some(Decimal::new(15, 4)),
            "model-a",
        ), // 0.0015
        make_model_inference_for_eval(
            inf1,
            Some(200),
            Some(60),
            Some(Decimal::new(25, 4)),
            "model-b",
        ), // 0.0025
        // inference 2 (run 1): 2 model calls, one WITHOUT cost → per-inference cost = NULL
        make_model_inference_for_eval(
            inf2,
            Some(300),
            Some(80),
            Some(Decimal::new(30, 4)),
            "model-a",
        ), // 0.003
        make_model_inference_for_eval(inf2, Some(50), Some(10), None, "model-c"), // no cost
        // inference 3 (run 1): 1 model call, with cost
        make_model_inference_for_eval(
            inf3,
            Some(150),
            Some(55),
            Some(Decimal::new(20, 4)),
            "model-a",
        ), // 0.002
        // inference 4 (run 2): 1 model call, with cost
        make_model_inference_for_eval(
            inf4,
            Some(400),
            Some(90),
            Some(Decimal::new(50, 4)),
            "model-a",
        ), // 0.005
        // inference 5 (run 2): 1 model call, with cost
        make_model_inference_for_eval(
            inf5,
            Some(600),
            Some(120),
            Some(Decimal::new(80, 4)),
            "model-a",
        ), // 0.008
    ];
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("insert model inferences");

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // --- Query ---
    let results = conn
        .get_evaluation_usage_statistics(
            &function_name,
            FunctionConfigType::Chat,
            &[run_id1, run_id2],
        )
        .await
        .expect("get_evaluation_usage_statistics should succeed");

    assert_eq!(results.len(), 2, "Expected exactly 2 run rows");

    let r1 = results
        .iter()
        .find(|r| r.evaluation_run_id == run_id1)
        .expect("should have result for run 1");
    let r2 = results
        .iter()
        .find(|r| r.evaluation_run_id == run_id2)
        .expect("should have result for run 2");

    // --- Run 1 assertions ---
    assert_eq!(r1.inference_count, 3, "run 1 should have 3 inferences");

    // Tokens: inf1 (100+200)=300, inf2 (300+50)=350, inf3 150 → total 800
    assert_eq!(
        r1.total_input_tokens,
        Some(800),
        "run 1 total_input_tokens: 300 + 350 + 150 = 800"
    );
    // Output tokens: inf1 (40+60)=100, inf2 (80+10)=90, inf3 55 → total 245
    assert_eq!(
        r1.total_output_tokens,
        Some(245),
        "run 1 total_output_tokens: 100 + 90 + 55 = 245"
    );

    // Cost: inf2 has one model inference with NULL cost → inf2 per-inference cost = NULL
    // The outer aggregation (per-run) sees 3 inferences but only 2 have non-NULL cost
    // → COUNT(*) != COUNT(cost) → total_cost = NULL
    assert!(
        r1.total_cost.is_none(),
        "run 1 total_cost should be None because inference 2 has a model inference with missing cost"
    );

    // avg processing time: (100 + 200 + 300) / 3 = 200
    let avg1 = r1
        .avg_processing_time_ms
        .expect("run 1 avg_processing_time_ms should be Some");
    assert!(
        (avg1 - 200.0).abs() < 1e-3,
        "run 1 avg_processing_time_ms should be ~200.0, got {avg1}"
    );

    // --- Run 2 assertions ---
    assert_eq!(r2.inference_count, 2, "run 2 should have 2 inferences");

    // Tokens: inf4 400 + inf5 600 = 1000
    assert_eq!(
        r2.total_input_tokens,
        Some(1000),
        "run 2 total_input_tokens: 400 + 600 = 1000"
    );
    // Output: inf4 90 + inf5 120 = 210
    assert_eq!(
        r2.total_output_tokens,
        Some(210),
        "run 2 total_output_tokens: 90 + 120 = 210"
    );

    // Cost: both inferences have cost → 0.005 + 0.008 = 0.013
    let cost2 = r2
        .total_cost
        .expect("run 2 total_cost should be Some (all model inferences have cost)");
    assert!(
        (cost2 - 0.013).abs() < 1e-6,
        "run 2 total_cost should be ~0.013, got {cost2}"
    );

    // avg processing time: (500 + 700) / 2 = 600
    let avg2 = r2
        .avg_processing_time_ms
        .expect("run 2 avg_processing_time_ms should be Some");
    assert!(
        (avg2 - 600.0).abs() < 1e-3,
        "run 2 avg_processing_time_ms should be ~600.0, got {avg2}"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_multi_run);

/// Test LEFT JOIN behavior: inferences exist but have NO model inferences.
///
/// When there are no model_inferences for an inference, the LEFT JOIN returns
/// NULL for all usage fields (input_tokens, output_tokens, cost).
/// But processing_time_ms comes from the inference table, so it should still be present.
async fn test_get_evaluation_usage_statistics_no_model_inferences(
    conn: impl EvaluationQueries + InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers,
) {
    let run_id = Uuid::now_v7();
    let function_name = format!("e2e_usage_no_mi_{run_id}");
    let eval_name = format!("e2e_usage_no_mi_eval_{run_id}");
    let dataset_name = format!("e2e_usage_no_mi_ds_{run_id}");

    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id,
        &function_name,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run");

    let inf1 = Uuid::now_v7();
    let inf2 = Uuid::now_v7();
    let dp1 = Uuid::now_v7();
    let dp2 = Uuid::now_v7();

    let inferences = vec![
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf1,
            function_name: &function_name,
            run_id,
            datapoint_id: dp1,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(250),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf2,
            function_name: &function_name,
            run_id,
            datapoint_id: dp2,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(350),
        }),
    ];
    conn.insert_chat_inferences(&inferences)
        .await
        .expect("insert chat inferences");

    // Deliberately insert ZERO model inferences
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let results = conn
        .get_evaluation_usage_statistics(&function_name, FunctionConfigType::Chat, &[run_id])
        .await
        .expect("get_evaluation_usage_statistics should succeed");

    assert_eq!(results.len(), 1, "Expected 1 run row");
    let r = &results[0];

    assert_eq!(r.evaluation_run_id, run_id);
    assert_eq!(r.inference_count, 2, "should count 2 inferences");

    // All model-inference-derived fields should be NULL (SUM of NULLs = NULL)
    assert!(
        r.total_input_tokens.is_none(),
        "total_input_tokens should be None when no model inferences exist"
    );
    assert!(
        r.total_output_tokens.is_none(),
        "total_output_tokens should be None when no model inferences exist"
    );
    assert!(
        r.total_cost.is_none(),
        "total_cost should be None when no model inferences exist"
    );

    // processing_time_ms comes from the inference table, so avg should still be present
    let avg_pt = r
        .avg_processing_time_ms
        .expect("avg_processing_time_ms should be Some even without model inferences");
    assert!(
        (avg_pt - 300.0).abs() < 1e-3,
        "avg_processing_time_ms should be (250 + 350) / 2 = 300.0, got {avg_pt}"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_no_model_inferences);

/// Test that querying usage stats for a nonexistent run returns empty results.
async fn test_get_evaluation_usage_statistics_nonexistent_run(
    conn: impl EvaluationQueries + TestDatabaseHelpers,
) {
    let results = conn
        .get_evaluation_usage_statistics(
            "nonexistent_function_xyz",
            FunctionConfigType::Chat,
            &[Uuid::now_v7()],
        )
        .await
        .expect("get_evaluation_usage_statistics should succeed for nonexistent run");

    assert!(
        results.is_empty(),
        "Expected empty results for nonexistent run"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_nonexistent_run);

/// Test that querying usage stats with empty run IDs returns empty results immediately.
async fn test_get_evaluation_usage_statistics_empty_run_ids(
    conn: impl EvaluationQueries + TestDatabaseHelpers,
) {
    let results = conn
        .get_evaluation_usage_statistics("any_function", FunctionConfigType::Chat, &[])
        .await
        .expect("get_evaluation_usage_statistics should succeed for empty run IDs");

    assert!(
        results.is_empty(),
        "Expected empty results for empty run IDs"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_empty_run_ids);

/// Test usage statistics with all costs present (no NULLs).
/// Verifies the happy path where everything aggregates cleanly.
///
/// Single run, 2 inferences:
///   - inference 1: 1 model inference (100 in, 40 out, cost 0.001)
///   - inference 2: 3 model inferences (all with cost)
///
/// This exercises multi-model-inference summing within a single inference
/// and correct aggregation across inferences when all costs are present.
async fn test_get_evaluation_usage_statistics_all_costs_present(
    conn: impl EvaluationQueries + InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers,
) {
    let run_id = Uuid::now_v7();
    let function_name = format!("e2e_usage_all_costs_{run_id}");
    let eval_name = format!("e2e_usage_all_costs_eval_{run_id}");
    let dataset_name = format!("e2e_usage_all_costs_ds_{run_id}");

    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id,
        &function_name,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run");

    let inf1 = Uuid::now_v7();
    let inf2 = Uuid::now_v7();
    let dp1 = Uuid::now_v7();
    let dp2 = Uuid::now_v7();

    let inferences = vec![
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf1,
            function_name: &function_name,
            run_id,
            datapoint_id: dp1,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(120),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf2,
            function_name: &function_name,
            run_id,
            datapoint_id: dp2,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(380),
        }),
    ];
    conn.insert_chat_inferences(&inferences)
        .await
        .expect("insert chat inferences");

    let model_inferences = vec![
        // inf1: single model inference
        make_model_inference_for_eval(
            inf1,
            Some(100),
            Some(40),
            Some(Decimal::new(10, 4)),
            "model-a",
        ), // 0.001
        // inf2: three model inferences (e.g. retries/fallbacks)
        make_model_inference_for_eval(
            inf2,
            Some(200),
            Some(60),
            Some(Decimal::new(20, 4)),
            "model-a",
        ), // 0.002
        make_model_inference_for_eval(
            inf2,
            Some(200),
            Some(60),
            Some(Decimal::new(20, 4)),
            "model-b",
        ), // 0.002
        make_model_inference_for_eval(
            inf2,
            Some(300),
            Some(80),
            Some(Decimal::new(35, 4)),
            "model-c",
        ), // 0.0035
    ];
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("insert model inferences");

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let results = conn
        .get_evaluation_usage_statistics(&function_name, FunctionConfigType::Chat, &[run_id])
        .await
        .expect("get_evaluation_usage_statistics should succeed");

    assert_eq!(results.len(), 1, "Expected 1 run row");
    let r = &results[0];

    assert_eq!(r.inference_count, 2);

    // inf1: 100 in, 40 out
    // inf2: 200+200+300=700 in, 60+60+80=200 out
    // total: 800 in, 240 out
    assert_eq!(
        r.total_input_tokens,
        Some(800),
        "total_input_tokens: 100 + 700 = 800"
    );
    assert_eq!(
        r.total_output_tokens,
        Some(240),
        "total_output_tokens: 40 + 200 = 240"
    );

    // inf1 cost: 0.001
    // inf2 cost: 0.002 + 0.002 + 0.0035 = 0.0075
    // total: 0.0085
    let cost = r
        .total_cost
        .expect("total_cost should be Some when all model inferences have cost");
    assert!(
        (cost - 0.0085).abs() < 1e-6,
        "total_cost should be ~0.0085, got {cost}"
    );

    // avg processing time: (120 + 380) / 2 = 250
    let avg_pt = r
        .avg_processing_time_ms
        .expect("avg_processing_time_ms should be Some");
    assert!(
        (avg_pt - 250.0).abs() < 1e-3,
        "avg_processing_time_ms should be ~250.0, got {avg_pt}"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_all_costs_present);

/// Test that usage statistics only includes inferences for the correct function,
/// even when other functions' inferences share the same evaluation_run_id tag.
async fn test_get_evaluation_usage_statistics_filters_by_function(
    conn: impl EvaluationQueries + InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers,
) {
    let run_id = Uuid::now_v7();
    let function_a = format!("e2e_usage_func_a_{run_id}");
    let function_b = format!("e2e_usage_func_b_{run_id}");
    let eval_name = format!("e2e_usage_filter_eval_{run_id}");
    let dataset_name = format!("e2e_usage_filter_ds_{run_id}");

    // Insert eval runs for both functions (same run_id but different function_name in the runs)
    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id,
        &function_a,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run");

    let dp1 = Uuid::now_v7();
    let dp2 = Uuid::now_v7();
    let inf_a = Uuid::now_v7();
    let inf_b = Uuid::now_v7();

    // Insert inference for function_a (should be counted)
    let inf_for_a = make_eval_chat_inference(EvalChatInferenceParams {
        id: inf_a,
        function_name: &function_a,
        run_id,
        datapoint_id: dp1,
        dataset_name: &dataset_name,
        evaluation_name: &eval_name,
        processing_time_ms: Some(100),
    });
    // Insert inference for function_b (should NOT be counted when querying function_a)
    let inf_for_b = make_eval_chat_inference(EvalChatInferenceParams {
        id: inf_b,
        function_name: &function_b,
        run_id,
        datapoint_id: dp2,
        dataset_name: &dataset_name,
        evaluation_name: &eval_name,
        processing_time_ms: Some(999),
    });
    conn.insert_chat_inferences(&[inf_for_a, inf_for_b])
        .await
        .expect("insert chat inferences");

    let model_inferences = vec![
        make_model_inference_for_eval(
            inf_a,
            Some(100),
            Some(40),
            Some(Decimal::new(10, 4)),
            "model-a",
        ),
        make_model_inference_for_eval(
            inf_b,
            Some(9999),
            Some(9999),
            Some(Decimal::new(9999, 4)),
            "model-b",
        ),
    ];
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("insert model inferences");

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Query for function_a only
    let results = conn
        .get_evaluation_usage_statistics(&function_a, FunctionConfigType::Chat, &[run_id])
        .await
        .expect("get_evaluation_usage_statistics should succeed");

    assert_eq!(results.len(), 1, "Expected 1 run row");
    let r = &results[0];

    assert_eq!(
        r.inference_count, 1,
        "should only count function_a's inference"
    );
    assert_eq!(
        r.total_input_tokens,
        Some(100),
        "should only include function_a tokens"
    );
    assert_eq!(
        r.total_output_tokens,
        Some(40),
        "should only include function_a tokens"
    );

    let avg_pt = r.avg_processing_time_ms.expect("avg should be present");
    assert!(
        (avg_pt - 100.0).abs() < 1e-3,
        "avg_processing_time_ms should be 100.0, not 999.0"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_filters_by_function);

/// Test usage statistics when some inferences have NULL token counts.
///
/// inference 1: model inference with input_tokens=None, output_tokens=None, cost=0.001
/// inference 2: model inference with input_tokens=200, output_tokens=80, cost=0.002
///
/// SUM(NULL, 200) = NULL for input (since NULL + 200 propagates NULL through SUM at the outer level)
/// Actually, the inner SUM per-inference turns NULL tokens into NULL,
/// then the outer SUM(NULL, 200) = 200 (SUM ignores NULLs).
/// Wait — this depends on the implementation:
///   - Inner: SUM(mi.input_tokens) GROUP BY inference_id → NULL if all tokens NULL for that inference
///   - Outer: SUM(iu.input_tokens) → SUM ignores NULLs → 200
///
/// This test verifies that NULL token values in individual model inferences
/// are handled correctly through the two-level aggregation.
async fn test_get_evaluation_usage_statistics_null_tokens(
    conn: impl EvaluationQueries + InferenceQueries + ModelInferenceQueries + TestDatabaseHelpers,
) {
    let run_id = Uuid::now_v7();
    let function_name = format!("e2e_usage_null_tok_{run_id}");
    let eval_name = format!("e2e_usage_null_tok_eval_{run_id}");
    let dataset_name = format!("e2e_usage_null_tok_ds_{run_id}");

    conn.insert_inference_evaluation_run(&make_eval_run(
        run_id,
        &function_name,
        &eval_name,
        &dataset_name,
    ))
    .await
    .expect("insert eval run");

    let inf1 = Uuid::now_v7();
    let inf2 = Uuid::now_v7();
    let dp1 = Uuid::now_v7();
    let dp2 = Uuid::now_v7();

    let inferences = vec![
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf1,
            function_name: &function_name,
            run_id,
            datapoint_id: dp1,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(100),
        }),
        make_eval_chat_inference(EvalChatInferenceParams {
            id: inf2,
            function_name: &function_name,
            run_id,
            datapoint_id: dp2,
            dataset_name: &dataset_name,
            evaluation_name: &eval_name,
            processing_time_ms: Some(200),
        }),
    ];
    conn.insert_chat_inferences(&inferences)
        .await
        .expect("insert chat inferences");

    let model_inferences = vec![
        // inf1: NULL tokens, but has cost
        make_model_inference_for_eval(inf1, None, None, Some(Decimal::new(10, 4)), "model-a"),
        // inf2: has tokens and cost
        make_model_inference_for_eval(
            inf2,
            Some(200),
            Some(80),
            Some(Decimal::new(20, 4)),
            "model-a",
        ),
    ];
    conn.insert_model_inferences(&model_inferences)
        .await
        .expect("insert model inferences");

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let results = conn
        .get_evaluation_usage_statistics(&function_name, FunctionConfigType::Chat, &[run_id])
        .await
        .expect("get_evaluation_usage_statistics should succeed");

    assert_eq!(results.len(), 1);
    let r = &results[0];

    assert_eq!(r.inference_count, 2);

    // Inner SUM for inf1: SUM(NULL) = NULL
    // Inner SUM for inf2: SUM(200) = 200
    // Outer SUM: SUM(NULL, 200) = 200 (SUM ignores NULLs in both Postgres and ClickHouse)
    assert_eq!(
        r.total_input_tokens,
        Some(200),
        "SUM should skip NULL token values: only inf2's 200 contributes"
    );
    assert_eq!(
        r.total_output_tokens,
        Some(80),
        "SUM should skip NULL token values: only inf2's 80 contributes"
    );

    // Both model inferences have cost, so both per-inference costs are non-NULL
    // → per-run cost = SUM(0.001, 0.002) = 0.003
    let cost = r
        .total_cost
        .expect("total_cost should be Some (all model inferences have cost)");
    assert!(
        (cost - 0.003).abs() < 1e-6,
        "total_cost should be ~0.003, got {cost}"
    );
}
make_db_test!(test_get_evaluation_usage_statistics_null_tokens);
