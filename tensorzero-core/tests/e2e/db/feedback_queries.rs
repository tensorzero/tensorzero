#![expect(clippy::print_stdout)]
use tensorzero_core::db::{
    clickhouse::test_helpers::get_clickhouse, feedback::FeedbackQueries, feedback::FeedbackRow,
};
use uuid::Uuid;

#[tokio::test]
async fn test_query_feedback_by_target_id_basic() {
    let clickhouse = get_clickhouse().await;

    // Use a known inference ID from the test database
    let target_id = Uuid::parse_str("0aaedb76-b442-7a94-830d-5c8202975950").unwrap();

    let feedback = clickhouse
        .query_feedback_by_target_id(target_id, None, None, Some(100))
        .await
        .unwrap();

    println!("Feedback count: {}", feedback.len());
    for (i, f) in feedback.iter().enumerate() {
        println!("Feedback {i}: {f:?}");
    }

    // Basic assertions
    assert!(!feedback.is_empty(), "Should have feedback for target");

    // Verify feedback is sorted by ID in descending order
    for window in feedback.windows(2) {
        let id_a = match &window[0] {
            FeedbackRow::Boolean(f) => f.id,
            FeedbackRow::Float(f) => f.id,
            FeedbackRow::Comment(f) => f.id,
            FeedbackRow::Demonstration(f) => f.id,
        };
        let id_b = match &window[1] {
            FeedbackRow::Boolean(f) => f.id,
            FeedbackRow::Float(f) => f.id,
            FeedbackRow::Comment(f) => f.id,
            FeedbackRow::Demonstration(f) => f.id,
        };
        assert!(id_a >= id_b, "Feedback should be sorted by ID descending");
    }
}

#[tokio::test]
async fn test_query_feedback_by_target_id_pagination() {
    let clickhouse = get_clickhouse().await;

    // Use a known inference ID with multiple feedback items
    let target_id = Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d721").unwrap();

    // Get first page
    let first_page = clickhouse
        .query_feedback_by_target_id(target_id, None, None, Some(5))
        .await
        .unwrap();

    assert!(!first_page.is_empty(), "Should have feedback items");
    assert!(first_page.len() <= 5, "Should respect page size limit");

    // If we have multiple items, test pagination with 'before'
    if first_page.len() > 1 {
        let second_id = match &first_page[1] {
            FeedbackRow::Boolean(f) => f.id,
            FeedbackRow::Float(f) => f.id,
            FeedbackRow::Comment(f) => f.id,
            FeedbackRow::Demonstration(f) => f.id,
        };

        let second_page = clickhouse
            .query_feedback_by_target_id(target_id, Some(second_id), None, Some(5))
            .await
            .unwrap();

        // Second page should not contain items from first page
        for item in &second_page {
            let item_id = match item {
                FeedbackRow::Boolean(f) => f.id,
                FeedbackRow::Float(f) => f.id,
                FeedbackRow::Comment(f) => f.id,
                FeedbackRow::Demonstration(f) => f.id,
            };
            assert!(item_id < second_id, "Second page should have older items");
        }
    }
}

#[tokio::test]
async fn test_query_feedback_bounds_by_target_id() {
    let clickhouse = get_clickhouse().await;

    // Use a known inference ID
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();

    let bounds = clickhouse
        .query_feedback_bounds_by_target_id(target_id)
        .await
        .unwrap();

    println!("Feedback bounds: {bounds:#?}");

    // If there's feedback, bounds should exist
    let feedback_count = clickhouse
        .count_feedback_by_target_id(target_id)
        .await
        .unwrap();

    if feedback_count > 0 {
        assert!(
            bounds.first_id.is_some(),
            "Should have first_id when feedback exists"
        );
        assert!(
            bounds.last_id.is_some(),
            "Should have last_id when feedback exists"
        );
        assert!(
            bounds.first_id.unwrap() <= bounds.last_id.unwrap(),
            "first_id should be <= last_id"
        );
    }
}

#[tokio::test]
async fn test_count_feedback_by_target_id() {
    let clickhouse = get_clickhouse().await;

    // Use a known inference ID
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();

    let count = clickhouse
        .count_feedback_by_target_id(target_id)
        .await
        .unwrap();

    println!("Total feedback count: {count}");

    // Get actual feedback to verify count
    let feedback = clickhouse
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .unwrap();

    assert_eq!(
        count as usize,
        feedback.len(),
        "Count should match actual feedback items"
    );
}

#[tokio::test]
async fn test_query_feedback_by_target_id_empty() {
    let clickhouse = get_clickhouse().await;

    // Use a UUID that likely doesn't exist
    let nonexistent_id = Uuid::parse_str("00000000-0000-0000-0000-000000000000").unwrap();

    let feedback = clickhouse
        .query_feedback_by_target_id(nonexistent_id, None, None, Some(100))
        .await
        .unwrap();

    assert!(
        feedback.is_empty(),
        "Should have no feedback for nonexistent target"
    );

    let count = clickhouse
        .count_feedback_by_target_id(nonexistent_id)
        .await
        .unwrap();

    assert_eq!(count, 0, "Count should be 0 for nonexistent target");
}
