#![expect(clippy::print_stdout)]
use tensorzero_core::db::{
    TimeWindow, clickhouse::test_helpers::get_clickhouse, feedback::FeedbackQueries,
    feedback::FeedbackRow,
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

// ==================== Get Cumulative Feedback Timeseries Tests ====================

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_basic() {
    let clickhouse = get_clickhouse().await;

    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    let timeseries = clickhouse
        .get_cumulative_feedback_timeseries(function_name, metric_name, None, TimeWindow::Day, 30)
        .await
        .unwrap();

    println!("Timeseries points: {}", timeseries.len());
    for (i, point) in timeseries.iter().take(5).enumerate() {
        println!(
            "Point {i}: period_end={}, variant={}, mean={}, count={}, cs_lower={:?}, cs_upper={:?}",
            point.period_end,
            point.variant_name,
            point.mean,
            point.count,
            point.cs_lower,
            point.cs_upper
        );
    }

    // Verify each point has valid data
    for point in &timeseries {
        assert!(
            !point.variant_name.is_empty(),
            "Variant name should not be empty"
        );
        assert!(point.count > 0, "Count should be positive");
        // Mean should be between 0 and 1 for boolean metrics
        assert!(
            (0.0..=1.0).contains(&point.mean),
            "Mean should be between 0 and 1 for boolean metric"
        );
        // Alpha should be the default confidence level
        assert!(
            point.alpha > 0.0 && point.alpha < 1.0,
            "Alpha should be between 0 and 1"
        );
    }

    // Verify cumulative property: counts should be non-decreasing for each variant
    let mut counts_by_variant: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for point in &timeseries {
        let prev_count = counts_by_variant
            .entry(point.variant_name.clone())
            .or_insert(0);
        assert!(
            point.count >= *prev_count,
            "Cumulative count should be non-decreasing for variant {}",
            point.variant_name
        );
        *prev_count = point.count;
    }
}

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_with_variant_filter() {
    let clickhouse = get_clickhouse().await;

    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // First get all data to find variant names
    let all_data = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name.clone(),
            metric_name.clone(),
            None,
            TimeWindow::Day,
            30,
        )
        .await
        .unwrap();

    if !all_data.is_empty() {
        let variant_name = all_data[0].variant_name.clone();
        let variant_filter = vec![variant_name.clone()];

        let filtered = clickhouse
            .get_cumulative_feedback_timeseries(
                function_name,
                metric_name,
                Some(variant_filter),
                TimeWindow::Day,
                30,
            )
            .await
            .unwrap();

        println!("Filtered timeseries points: {}", filtered.len());

        // All results should be for the filtered variant
        for point in &filtered {
            assert_eq!(
                point.variant_name, variant_name,
                "All points should be for the filtered variant"
            );
        }
    }
}

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_different_time_windows() {
    let clickhouse = get_clickhouse().await;

    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // Test different time windows (excluding Cumulative which is not supported)
    let time_windows = [
        ("minute", TimeWindow::Minute),
        ("hour", TimeWindow::Hour),
        ("day", TimeWindow::Day),
        ("week", TimeWindow::Week),
        ("month", TimeWindow::Month),
    ];

    for (name, time_window) in time_windows {
        let timeseries = clickhouse
            .get_cumulative_feedback_timeseries(
                function_name.clone(),
                metric_name.clone(),
                None,
                time_window,
                10,
            )
            .await
            .unwrap();

        println!("Time window {}: {} points", name, timeseries.len());

        // Should not error and should return valid data
        for point in &timeseries {
            assert!(!point.variant_name.is_empty());
        }
    }
}

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_cumulative_window_not_supported() {
    let clickhouse = get_clickhouse().await;

    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // Cumulative time window is not supported for feedback timeseries
    let result = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Cumulative,
            10,
        )
        .await;

    assert!(
        result.is_err(),
        "Cumulative time window should return an error"
    );
}

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_empty_result() {
    let clickhouse = get_clickhouse().await;

    // Use a nonexistent function/metric combination
    let function_name = "nonexistent_function_xyz".to_string();
    let metric_name = "nonexistent_metric_xyz".to_string();

    let timeseries = clickhouse
        .get_cumulative_feedback_timeseries(function_name, metric_name, None, TimeWindow::Day, 30)
        .await
        .unwrap();

    assert!(
        timeseries.is_empty(),
        "Should have no timeseries data for nonexistent function/metric"
    );
}

#[tokio::test]
async fn test_get_cumulative_feedback_timeseries_max_periods() {
    let clickhouse = get_clickhouse().await;

    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // Request only 1 period
    let timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name.clone(),
            metric_name.clone(),
            None,
            TimeWindow::Day,
            1,
        )
        .await
        .unwrap();

    // With max_periods=1, we should get at most 2 time periods (1 complete + current)
    // multiplied by the number of variants
    let unique_periods: std::collections::HashSet<_> =
        timeseries.iter().map(|p| p.period_end).collect();
    println!(
        "With max_periods=1: {} total points, {} unique periods",
        timeseries.len(),
        unique_periods.len()
    );

    // Request more periods
    let timeseries_more = clickhouse
        .get_cumulative_feedback_timeseries(function_name, metric_name, None, TimeWindow::Day, 30)
        .await
        .unwrap();

    let unique_periods_more: std::collections::HashSet<_> =
        timeseries_more.iter().map(|p| p.period_end).collect();
    println!(
        "With max_periods=30: {} total points, {} unique periods",
        timeseries_more.len(),
        unique_periods_more.len()
    );

    // More periods requested should give equal or more unique time periods
    assert!(
        unique_periods_more.len() >= unique_periods.len(),
        "More max_periods should give equal or more unique time periods"
    );
}
