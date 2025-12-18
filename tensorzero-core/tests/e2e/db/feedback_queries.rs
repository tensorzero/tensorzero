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
    assert!(first_page.len() > 1, "Should have multiple feedback items");
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

#[tokio::test]
async fn test_query_feedback_bounds_by_target_id() {
    let clickhouse = get_clickhouse().await;

    // Use a known inference ID
    let target_id = Uuid::parse_str("019634c7-768e-7630-8d51-b9a93e79911e").unwrap();

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

    assert!(feedback_count > 0, "Should have feedback for target");

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

    // Aggregate bounds should match min/max of the per-type bounds that exist
    let mut first_ids: Vec<Uuid> = Vec::new();
    let mut last_ids: Vec<Uuid> = Vec::new();

    for table_bounds in [
        bounds.by_type.boolean,
        bounds.by_type.float,
        bounds.by_type.comment,
        bounds.by_type.demonstration,
    ] {
        if let Some(first) = table_bounds.first_id {
            first_ids.push(first);
        }
        if let Some(last) = table_bounds.last_id {
            last_ids.push(last);
        }
    }

    assert!(!first_ids.is_empty(), "Should have first_ids");
    assert!(!last_ids.is_empty(), "Should have last_ids");

    let min_first = first_ids.iter().min().copied().unwrap();
    assert_eq!(
        bounds.first_id,
        Some(min_first),
        "Aggregate first_id should be min of per-type bounds"
    );
    let max_last = last_ids.iter().max().copied().unwrap();
    assert_eq!(
        bounds.last_id,
        Some(max_last),
        "Aggregate last_id should be max of per-type bounds"
    );
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

    let nonexistent_id = Uuid::now_v7();

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

#[tokio::test]
async fn test_query_feedback_bounds_by_target_id_empty() {
    let clickhouse = get_clickhouse().await;

    let nonexistent_id = Uuid::now_v7();

    let bounds = clickhouse
        .query_feedback_bounds_by_target_id(nonexistent_id)
        .await
        .unwrap();

    assert!(
        bounds.first_id.is_none() && bounds.last_id.is_none(),
        "Expected no aggregate bounds for nonexistent target"
    );
    assert!(
        bounds.by_type.boolean.first_id.is_none()
            && bounds.by_type.float.first_id.is_none()
            && bounds.by_type.comment.first_id.is_none()
            && bounds.by_type.demonstration.first_id.is_none(),
        "Expected all per-type bounds to be empty for nonexistent target"
    );
}
