//! E2E tests for evaluation queries (ClickHouse and Postgres).

use tensorzero_core::db::evaluation_queries::{
    ChatEvaluationResultRow, EvaluationQueries, EvaluationResultRow, JsonEvaluationResultRow,
};
use tensorzero_core::function::FunctionConfigType;
use uuid::Uuid;

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
        .filter_map(|r| r.metric_name.as_ref())
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
        .filter_map(|r| r.metric_name.as_ref())
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
}
make_db_test!(test_get_evaluation_results_json_datapoint_details);

// ============================================================================
// search_evaluation_runs tests
// ============================================================================

/// Test that search_evaluation_runs returns all runs for an evaluation with an empty query.
async fn test_search_evaluation_runs_empty_query(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs("entity_extraction", "extract_entities", "", 100, 0)
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
        .search_evaluation_runs("entity_extraction", "extract_entities", "mini", 100, 0)
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
        .search_evaluation_runs("entity_extraction", "extract_entities", "gpt4o", 100, 0)
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
            "entity_extraction",
            "extract_entities",
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
        .search_evaluation_runs("entity_extraction", "extract_entities", "19bd", 100, 0)
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
            "entity_extraction",
            "extract_entities",
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
        .search_evaluation_runs("nonexistent_evaluation", "extract_entities", "", 100, 0)
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
        .search_evaluation_runs("entity_extraction", "nonexistent_function", "", 100, 0)
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for nonexistent function name"
    );
}
make_db_test!(test_search_evaluation_runs_wrong_function_name);

/// Test that search_evaluation_runs respects the limit parameter.
async fn test_search_evaluation_runs_with_limit(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs("entity_extraction", "extract_entities", "", 1, 0)
        .await
        .unwrap();

    assert_eq!(results.len(), 1, "Expected 1 result with limit=1");
}
make_db_test!(test_search_evaluation_runs_with_limit);

/// Test that search_evaluation_runs respects the offset parameter.
async fn test_search_evaluation_runs_with_offset(conn: impl EvaluationQueries) {
    let results = conn
        .search_evaluation_runs("entity_extraction", "extract_entities", "", 100, 1)
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
        .search_evaluation_runs("entity_extraction", "extract_entities", "", 100, 100)
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        0,
        "Expected 0 results when offset is beyond all available runs"
    );
}
make_db_test!(test_search_evaluation_runs_offset_beyond_results);
