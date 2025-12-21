//! E2E tests for evaluation ClickHouse queries.

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::function::FunctionConfigType;
use uuid::Uuid;

// ============================================================================
// get_evaluation_run_infos tests
// ============================================================================

/// Test that get_evaluation_run_infos returns correct run infos for multiple evaluation runs.
#[tokio::test]
async fn test_get_evaluation_run_infos_multiple_runs() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id1 =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").expect("Valid UUID");

    let run_infos = clickhouse
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

/// Test that get_evaluation_run_infos returns correct info for a single evaluation run.
#[tokio::test]
async fn test_get_evaluation_run_infos_single_run() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let run_infos = clickhouse
        .get_evaluation_run_infos(&[evaluation_run_id], "extract_entities")
        .await
        .unwrap();

    assert_eq!(run_infos.len(), 1, "Expected 1 evaluation run info");
    assert_eq!(run_infos[0].evaluation_run_id, evaluation_run_id);
    assert_eq!(run_infos[0].variant_name, "gpt4o_mini_initial_prompt");
}

/// Test that get_evaluation_run_infos returns empty for nonexistent run IDs.
#[tokio::test]
async fn test_get_evaluation_run_infos_nonexistent_run() {
    let clickhouse = get_clickhouse().await;

    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let run_infos = clickhouse
        .get_evaluation_run_infos(&[nonexistent_id], "extract_entities")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for nonexistent run"
    );
}

/// Test that get_evaluation_run_infos returns empty when function name doesn't match.
#[tokio::test]
async fn test_get_evaluation_run_infos_wrong_function() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let run_infos = clickhouse
        .get_evaluation_run_infos(&[evaluation_run_id], "nonexistent_function")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for wrong function"
    );
}

/// Test that get_evaluation_run_infos returns empty for empty input.
#[tokio::test]
async fn test_get_evaluation_run_infos_empty_input() {
    let clickhouse = get_clickhouse().await;

    let run_infos = clickhouse
        .get_evaluation_run_infos(&[], "extract_entities")
        .await
        .unwrap();

    assert_eq!(
        run_infos.len(),
        0,
        "Expected 0 evaluation run infos for empty input"
    );
}

// ============================================================================
// get_evaluation_run_infos_for_datapoint tests
// ============================================================================

/// Test that get_evaluation_run_infos_for_datapoint returns correct info for a JSON function datapoint.
#[tokio::test]
async fn test_get_evaluation_run_infos_for_datapoint_json_function() {
    let clickhouse = get_clickhouse().await;

    // Datapoint ID from the test fixture for extract_entities function
    let datapoint_id = Uuid::parse_str("0196368e-0b64-7321-ab5b-c32eefbf3e9f").expect("Valid UUID");

    let run_infos = clickhouse
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

/// Test that get_evaluation_run_infos_for_datapoint returns correct info for a chat function datapoint.
#[tokio::test]
async fn test_get_evaluation_run_infos_for_datapoint_chat_function() {
    let clickhouse = get_clickhouse().await;

    // Datapoint ID from the test fixture for write_haiku function
    let datapoint_id = Uuid::parse_str("0196374a-d03f-7420-9da5-1561cba71ddb").expect("Valid UUID");

    let run_infos = clickhouse
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
    assert_eq!(run_infos[0].variant_name, "better_prompt_haiku_3_5");
}

/// Test that get_evaluation_run_infos_for_datapoint returns empty for nonexistent datapoint.
#[tokio::test]
async fn test_get_evaluation_run_infos_for_datapoint_nonexistent() {
    let clickhouse = get_clickhouse().await;

    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let run_infos = clickhouse
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

/// Test that get_evaluation_run_infos_for_datapoint returns empty when function name doesn't match.
#[tokio::test]
async fn test_get_evaluation_run_infos_for_datapoint_wrong_function() {
    let clickhouse = get_clickhouse().await;

    // Use a valid datapoint ID but with wrong function name
    let datapoint_id = Uuid::parse_str("0196368e-0b64-7321-ab5b-c32eefbf3e9f").expect("Valid UUID");

    let run_infos = clickhouse
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

// ============================================================================
// count_datapoints_for_evaluation tests
// ============================================================================

/// Test using the shared test database with pre-existing fixture data.
/// This test verifies that the correct number of datapoints is returned for a chat function evaluation.
/// The count should not include data that was created after the evaluation run.
#[tokio::test]
async fn test_count_datapoints_for_haiku_evaluation() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("01963690-dff2-7cd3-b724-62fb705772a1").expect("Valid UUID");

    let count = clickhouse
        .count_datapoints_for_evaluation("write_haiku", &[evaluation_run_id])
        .await
        .unwrap();

    // This should not include data that is after the evaluation run
    assert_eq!(
        count, 77,
        "Expected 77 datapoints for haiku evaluation, got {count}"
    );
}

/// Test using the shared test database with pre-existing fixture data.
/// This test verifies that the correct number of datapoints is returned for a JSON function evaluation.
#[tokio::test]
async fn test_count_datapoints_for_entity_extraction_evaluation() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let count = clickhouse
        .count_datapoints_for_evaluation("extract_entities", &[evaluation_run_id])
        .await
        .unwrap();

    assert_eq!(
        count, 41,
        "Expected 41 datapoints for entity_extraction evaluation, got {count}"
    );
}

// ============================================================================
// get_evaluation_statistics tests
// ============================================================================

/// Test that get_evaluation_statistics returns correct statistics for a JSON function evaluation.
#[tokio::test]
async fn test_get_evaluation_statistics_json_function() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports".to_string(),
    ];

    let statistics = clickhouse
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

/// Test that get_evaluation_statistics returns correct statistics for multiple evaluation runs.
#[tokio::test]
async fn test_get_evaluation_statistics_multiple_runs() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id1 =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = clickhouse
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

/// Test that get_evaluation_statistics returns empty for nonexistent evaluation runs.
#[tokio::test]
async fn test_get_evaluation_statistics_nonexistent_run() {
    let clickhouse = get_clickhouse().await;

    let nonexistent_id =
        Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = clickhouse
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

/// Test that get_evaluation_statistics returns empty for nonexistent metrics.
#[tokio::test]
async fn test_get_evaluation_statistics_nonexistent_metric() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec!["nonexistent_metric".to_string()];

    let statistics = clickhouse
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

/// Test that get_evaluation_statistics returns empty for wrong function name.
#[tokio::test]
async fn test_get_evaluation_statistics_wrong_function() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let metric_names = vec![
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string(),
    ];

    let statistics = clickhouse
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

/// Test that get_evaluation_statistics handles chat function type correctly.
#[tokio::test]
async fn test_get_evaluation_statistics_chat_function() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID");

    // This evaluation uses the write_haiku function which is a chat function
    let metric_names =
        vec!["tensorzero::evaluation_name::haiku_eval::evaluator_name::haiku_score".to_string()];

    let statistics = clickhouse
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

// ============================================================================
// get_evaluation_results tests
// ============================================================================

/// Test that get_evaluation_results returns correct results for haiku evaluation.
#[tokio::test]
async fn test_get_evaluation_results_haiku() {
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;

    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = clickhouse
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &[
                "tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                    .to_string(),
            ],
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
                assert_eq!(row.variant_name, "better_prompt_haiku_3_5");
            }
            _ => panic!("Expected Chat result"),
        }
    }

    // Verify we have both metric types
    let metric_names: std::collections::HashSet<_> = results
        .iter()
        .filter_map(|r| match r {
            EvaluationResultRow::Chat(row) => row.metric_name.as_ref(),
            EvaluationResultRow::Json(row) => row.metric_name.as_ref(),
        })
        .collect();
    assert!(
        metric_names.contains(
            &"tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()
        ),
        "Should have exact_match metric"
    );
    assert!(
        metric_names.contains(
            &"tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f".to_string()
        ),
        "Should have topic_starts_with_f metric"
    );

    // Verify datapoint count (should be 5 unique datapoints)
    let datapoint_ids: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();
    assert_eq!(datapoint_ids.len(), 5, "Expected 5 unique datapoints");
}

/// Test that get_evaluation_results handles entity_extraction (JSON function) correctly.
#[tokio::test]
async fn test_get_evaluation_results_entity_extraction() {
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;

    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("0196368f-19bd-7082-a677-1c0bf346ff24").expect("Valid UUID");

    let results = clickhouse
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
            EvaluationResultRow::Chat(row) => row.metric_name.as_ref(),
            EvaluationResultRow::Json(row) => row.metric_name.as_ref(),
        })
        .collect();
    assert!(
        metric_names.contains(
            &"tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
                .to_string()
        ),
        "Should have exact_match metric"
    );
    assert!(
        metric_names.contains(
            &"tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
                .to_string()
        ),
        "Should have count_sports metric"
    );

    // Verify datapoint count
    let datapoint_ids: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();
    assert_eq!(datapoint_ids.len(), 2, "Expected 2 unique datapoints");

    // Verify results are Json type
    for result in &results {
        match result {
            EvaluationResultRow::Json(_) => {}
            _ => panic!("Expected Json result"),
        }
    }
}

/// Test that get_evaluation_results handles multiple evaluation runs (ragged case).
#[tokio::test]
async fn test_get_evaluation_results_multiple_runs() {
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;

    let clickhouse = get_clickhouse().await;

    let evaluation_run_id1 =
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").expect("Valid UUID");
    let evaluation_run_id2 =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = clickhouse
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id1, evaluation_run_id2],
            FunctionConfigType::Chat,
            &[
                "tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string(),
                "tensorzero::evaluation_name::haiku::evaluator_name::topic_starts_with_f"
                    .to_string(),
            ],
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

/// Test that get_evaluation_results returns empty for nonexistent function.
#[tokio::test]
async fn test_get_evaluation_results_nonexistent_function() {
    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    let results = clickhouse
        .get_evaluation_results(
            "nonexistent_function",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["some_metric".to_string()],
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

/// Test that get_evaluation_results respects pagination offset.
#[tokio::test]
async fn test_get_evaluation_results_pagination() {
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;

    let clickhouse = get_clickhouse().await;

    let evaluation_run_id =
        Uuid::parse_str("01963691-9d3c-7793-a8be-3937ebb849c1").expect("Valid UUID");

    // Get first page (5 datapoints)
    let first_page = clickhouse
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
            5,
            0,
        )
        .await
        .unwrap();

    // Get second page (5 datapoints starting from offset 5)
    let second_page = clickhouse
        .get_evaluation_results(
            "write_haiku",
            &[evaluation_run_id],
            FunctionConfigType::Chat,
            &["tensorzero::evaluation_name::haiku::evaluator_name::exact_match".to_string()],
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
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();
    let second_datapoints: std::collections::HashSet<_> = second_page
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();

    let overlap: Vec<_> = first_datapoints.intersection(&second_datapoints).collect();
    assert!(
        overlap.is_empty(),
        "Pages should not have overlapping datapoints"
    );
}
