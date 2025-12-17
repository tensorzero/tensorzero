//! E2E tests for evaluation ClickHouse queries.

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use uuid::Uuid;

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
