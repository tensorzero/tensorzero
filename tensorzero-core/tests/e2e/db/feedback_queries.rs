#![expect(clippy::print_stdout)]
//! Shared test logic for FeedbackQueries implementations (ClickHouse and Postgres).
//!
//! Each test function accepts a connection implementing `FeedbackQueries`.
//! Tests use `>=` assertions since ClickHouse may accumulate data from other tests.

use std::collections::HashMap;
use std::time::Duration;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::config::{
    MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
};
use tensorzero_core::db::TimeWindow;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CommentTargetType,
    DemonstrationFeedbackInsert, FeedbackQueries, FeedbackRow, FloatMetricFeedbackInsert,
    GetVariantPerformanceParams, StaticEvaluationHumanFeedbackInsert,
};
use tensorzero_core::function::FunctionConfigType;
use uuid::Uuid;

// ===== SHARED READ TEST IMPLEMENTATIONS =====

async fn test_query_feedback_by_target_id_basic(conn: impl FeedbackQueries) {
    // Use a known inference ID from the test database
    let target_id = Uuid::parse_str("0aaedb76-b442-7a94-830d-5c8202975950").unwrap();

    let feedback = conn
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
make_db_test!(test_query_feedback_by_target_id_basic);

async fn test_query_feedback_by_target_id_pagination(conn: impl FeedbackQueries) {
    // Use a known inference ID with multiple feedback items
    let target_id = Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d721").unwrap();

    // Get first page
    let first_page = conn
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

    let second_page = conn
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
make_db_test!(test_query_feedback_by_target_id_pagination);

async fn test_query_feedback_bounds_by_target_id(conn: impl FeedbackQueries) {
    // Use a known inference ID
    let target_id = Uuid::parse_str("019634c7-768e-7630-8d51-b9a93e79911e").unwrap();

    let bounds = conn
        .query_feedback_bounds_by_target_id(target_id)
        .await
        .unwrap();

    println!("Feedback bounds: {bounds:#?}");

    // If there's feedback, bounds should exist
    let feedback_count = conn.count_feedback_by_target_id(target_id).await.unwrap();

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
make_db_test!(test_query_feedback_bounds_by_target_id);

async fn test_count_feedback_by_target_id(conn: impl FeedbackQueries) {
    // Use a known inference ID
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();

    let count = conn.count_feedback_by_target_id(target_id).await.unwrap();

    println!("Total feedback count: {count}");

    // Get actual feedback to verify count
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .unwrap();

    assert_eq!(
        count as usize,
        feedback.len(),
        "Count should match actual feedback items"
    );
}
make_db_test!(test_count_feedback_by_target_id);

async fn test_query_feedback_by_target_id_empty(conn: impl FeedbackQueries) {
    let nonexistent_id = Uuid::now_v7();

    let feedback = conn
        .query_feedback_by_target_id(nonexistent_id, None, None, Some(100))
        .await
        .unwrap();

    assert!(
        feedback.is_empty(),
        "Should have no feedback for nonexistent target"
    );

    let count = conn
        .count_feedback_by_target_id(nonexistent_id)
        .await
        .unwrap();

    assert_eq!(count, 0, "Count should be 0 for nonexistent target");
}
make_db_test!(test_query_feedback_by_target_id_empty);

async fn test_query_feedback_bounds_by_target_id_empty(conn: impl FeedbackQueries) {
    let nonexistent_id = Uuid::now_v7();

    let bounds = conn
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
make_db_test!(test_query_feedback_bounds_by_target_id_empty);

// ==================== Get Cumulative Feedback Timeseries Tests ====================

async fn test_get_cumulative_feedback_timeseries_basic(conn: impl FeedbackQueries) {
    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    let timeseries = conn
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
make_db_test!(test_get_cumulative_feedback_timeseries_basic);

async fn test_get_cumulative_feedback_timeseries_with_variant_filter(conn: impl FeedbackQueries) {
    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // First get all data to find variant names
    let all_data = conn
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

        let filtered = conn
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
make_db_test!(test_get_cumulative_feedback_timeseries_with_variant_filter);

async fn test_get_cumulative_feedback_timeseries_different_time_windows(
    conn: impl FeedbackQueries,
) {
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
        let timeseries = conn
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
make_db_test!(test_get_cumulative_feedback_timeseries_different_time_windows);

async fn test_get_cumulative_feedback_timeseries_cumulative_window_not_supported(
    conn: impl FeedbackQueries,
) {
    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // Cumulative time window is not supported for feedback timeseries
    let result = conn
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
make_db_test!(test_get_cumulative_feedback_timeseries_cumulative_window_not_supported);

async fn test_get_cumulative_feedback_timeseries_empty_result(conn: impl FeedbackQueries) {
    // Use a nonexistent function/metric combination
    let function_name = "nonexistent_function_xyz".to_string();
    let metric_name = "nonexistent_metric_xyz".to_string();

    let timeseries = conn
        .get_cumulative_feedback_timeseries(function_name, metric_name, None, TimeWindow::Day, 30)
        .await
        .unwrap();

    assert!(
        timeseries.is_empty(),
        "Should have no timeseries data for nonexistent function/metric"
    );
}
make_db_test!(test_get_cumulative_feedback_timeseries_empty_result);

async fn test_get_cumulative_feedback_timeseries_max_periods(conn: impl FeedbackQueries) {
    let function_name = "basic_test".to_string();
    let metric_name = "task_success".to_string();

    // Request only 1 period
    let timeseries = conn
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
    let timeseries_more = conn
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
make_db_test!(test_get_cumulative_feedback_timeseries_max_periods);

// =====================================================================
// Tests for get_variant_performances
// =====================================================================

async fn test_get_variant_performances_inference_level_cumulative(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "weather_helper",
        function_type: FunctionConfigType::Chat,
        metric_name: "user_rating",
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("Variant performance results (cumulative):");
    for result in &results {
        println!("  {result:?}");
    }

    // All results should have the epoch timestamp for cumulative
    for result in &results {
        assert_eq!(
            result.period_start.to_rfc3339(),
            "1970-01-01T00:00:00+00:00",
            "Cumulative results should have epoch timestamp"
        );
        assert!(result.count > 0, "Count should be positive");
    }
}
make_db_test!(test_get_variant_performances_inference_level_cumulative);

async fn test_get_variant_performances_inference_level_week(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "weather_helper",
        function_type: FunctionConfigType::Chat,
        metric_name: "user_rating",
        metric_config: &metric_config,
        time_window: TimeWindow::Week,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("Variant performance results (weekly):");
    for result in &results {
        println!("  {result:?}");
    }

    // Results should be ordered by period_start ASC, then variant_name ASC
    for window in results.windows(2) {
        assert!(
            window[0].period_start <= window[1].period_start
                || (window[0].period_start == window[1].period_start
                    && window[0].variant_name <= window[1].variant_name),
            "Results should be ordered by period_start ASC, variant_name ASC"
        );
    }
}
make_db_test!(test_get_variant_performances_inference_level_week);

async fn test_get_variant_performances_episode_level_cumulative(conn: impl FeedbackQueries) {
    // Use episode-level metric
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Episode,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "weather_helper",
        function_type: FunctionConfigType::Chat,
        metric_name: "task_success", // Assume this is an episode-level metric
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("Episode-level variant performance results (cumulative):");
    for result in &results {
        println!("  {result:?}");
    }

    // All results should have the epoch timestamp for cumulative
    for result in &results {
        assert_eq!(
            result.period_start.to_rfc3339(),
            "1970-01-01T00:00:00+00:00",
            "Cumulative results should have epoch timestamp"
        );
    }
}
make_db_test!(test_get_variant_performances_episode_level_cumulative);

async fn test_get_variant_performances_episode_level_week(conn: impl FeedbackQueries) {
    // Use episode-level metric with time window to test ordering
    // Uses haiku_rating_episode on write_haiku which has data across multiple weeks
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Episode,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating_episode",
        metric_config: &metric_config,
        time_window: TimeWindow::Week,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("Episode-level variant performance results (weekly):");
    for result in &results {
        println!("  {result:?}");
    }

    assert!(results.len() > 1, "Should have at least two results");

    // Results should be ordered by period_start ASC, then variant_name ASC
    for window in results.windows(2) {
        assert!(
            window[0].period_start <= window[1].period_start
                || (window[0].period_start == window[1].period_start
                    && window[0].variant_name <= window[1].variant_name),
            "Results should be ordered by period_start ASC, variant_name ASC"
        );
    }
}
make_db_test!(test_get_variant_performances_episode_level_week);

async fn test_get_variant_performances_with_variant_filter(conn: impl FeedbackQueries) {
    // Uses haiku_rating on write_haiku which has fixture data in both ClickHouse and Postgres
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    // First get all variants
    let params_all = GetVariantPerformanceParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating",
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: None,
    };

    let all_results = conn.get_variant_performances(params_all).await.unwrap();

    assert!(!all_results.is_empty(), "Should have at least one result");

    // Filter by specific variant
    let target_variant = &all_results[0].variant_name;
    let params_filtered = GetVariantPerformanceParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating",
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: Some(target_variant.as_str()),
    };

    let filtered_results = conn
        .get_variant_performances(params_filtered)
        .await
        .unwrap();

    println!("Filtered results for variant '{target_variant}': {filtered_results:?}");

    // All filtered results should be for the target variant
    for result in &filtered_results {
        assert_eq!(
            &result.variant_name, target_variant,
            "Filtered results should only contain target variant"
        );
    }
}
make_db_test!(test_get_variant_performances_with_variant_filter);

async fn test_get_variant_performances_empty_for_nonexistent_function(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "nonexistent_function_12345",
        function_type: FunctionConfigType::Chat,
        metric_name: "user_rating",
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    assert!(
        results.is_empty(),
        "Should return empty results for nonexistent function"
    );
}
make_db_test!(test_get_variant_performances_empty_for_nonexistent_function);

async fn test_get_variant_performances_different_time_windows(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    // Test each time window type (excluding cumulative which is tested separately)
    let time_windows = [
        TimeWindow::Minute,
        TimeWindow::Hour,
        TimeWindow::Day,
        TimeWindow::Week,
        TimeWindow::Month,
    ];

    for time_window in time_windows {
        let params = GetVariantPerformanceParams {
            function_name: "weather_helper",
            function_type: FunctionConfigType::Chat,
            metric_name: "user_rating",
            metric_config: &metric_config,
            time_window: time_window.clone(),
            variant_name: None,
        };

        let results = conn.get_variant_performances(params).await;
        assert!(
            results.is_ok(),
            "Query should succeed for time window {time_window:?}"
        );

        println!(
            "Time window {:?}: {} results",
            time_window,
            results.unwrap().len()
        );
    }
}
make_db_test!(test_get_variant_performances_different_time_windows);

async fn test_get_variant_performances_boolean_metric(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "weather_helper",
        function_type: FunctionConfigType::Chat,
        metric_name: "thumbs_up",
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: None,
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("Boolean metric variant performance results:");
    for result in &results {
        println!("  {result:?}");
        // For boolean metrics, avg_metric should be between 0 and 1
        assert!(
            result.avg_metric >= 0.0 && result.avg_metric <= 1.0,
            "Boolean metric avg should be between 0 and 1, got {}",
            result.avg_metric
        );
    }
}
make_db_test!(test_get_variant_performances_boolean_metric);

async fn test_get_variant_performances_empty_for_nonexistent_metric(conn: impl FeedbackQueries) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "non_existent_metric",
        metric_config: &metric_config,
        time_window: TimeWindow::Week,
        variant_name: Some("gpt4o_initial_prompt"),
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    assert!(
        results.is_empty(),
        "Should return empty results for non-existent metric"
    );
}
make_db_test!(test_get_variant_performances_empty_for_nonexistent_metric);

async fn test_get_variant_performances_ask_question_solved_with_variant(
    conn: impl FeedbackQueries,
) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Episode,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "ask_question",
        function_type: FunctionConfigType::Json,
        metric_name: "solved",
        metric_config: &metric_config,
        time_window: TimeWindow::Week,
        variant_name: Some("baseline"),
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    // Find the expected data points
    let dec30_result = results
        .iter()
        .find(|r| {
            r.period_start.to_rfc3339().starts_with("2024-12-30") && r.variant_name == "baseline"
        })
        .expect("There should be a result for 2024-12-30 and baseline variant");
    let apr28_result = results
        .iter()
        .find(|r| {
            r.period_start.to_rfc3339().starts_with("2025-04-28") && r.variant_name == "baseline"
        })
        .expect("There should be a result for 2025-04-28 and baseline variant");

    // Verify the 2024-12-30 data point
    assert_eq!(dec30_result.variant_name, "baseline");
    assert_eq!(dec30_result.count, 23);
    assert!(
        (dec30_result.avg_metric - 0.043478260869565216).abs() < 1e-6,
        "avg_metric mismatch: {}",
        dec30_result.avg_metric
    );
    assert!(
        (dec30_result.stdev.expect("Result should contain stdev") - 0.20851441405707477).abs()
            < 1e-6,
        "stdev mismatch",
    );
    assert!(
        (dec30_result
            .ci_error
            .expect("Result should contain ci_error")
            - 0.08521739130434784)
            .abs()
            < 1e-6,
        "ci_error mismatch",
    );

    // Verify the 2025-04-28 data point
    assert_eq!(apr28_result.variant_name, "baseline");
    assert_eq!(apr28_result.count, 48);
    assert!(
        (apr28_result.avg_metric - 0.4791666666666667).abs() < 1e-6,
        "avg_metric mismatch: {}",
        apr28_result.avg_metric
    );
    assert!(
        (apr28_result.stdev.expect("Result should contain stdev") - 0.5048523413086471).abs()
            < 1e-6,
        "stdev mismatch",
    );
    assert!(
        (apr28_result
            .ci_error
            .expect("Result should contain ci_error")
            - 0.1428235512262245)
            .abs()
            < 1e-6,
        "ci_error mismatch",
    );

    // All results should be for the baseline variant
    for result in &results {
        assert_eq!(
            result.variant_name, "baseline",
            "All results should be for baseline variant"
        );
    }
}
make_db_test!(test_get_variant_performances_ask_question_solved_with_variant);

/// Matches TypeScript test: "getVariantPerformances for ask_question and num_questions with specific variant"
async fn test_get_variant_performances_ask_question_num_questions_with_variant(
    conn: impl FeedbackQueries,
) {
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Min,
        level: MetricConfigLevel::Episode,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "ask_question",
        function_type: FunctionConfigType::Json,
        metric_name: "num_questions",
        metric_config: &metric_config,
        time_window: TimeWindow::Week,
        variant_name: Some("baseline"),
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    // Find the expected data point
    let dec30_result = results
        .iter()
        .find(|r| {
            r.period_start.to_rfc3339().starts_with("2024-12-30") && r.variant_name == "baseline"
        })
        .expect("There should be a result for 2024-12-30 and baseline variant");

    // Verify the 2024-12-30 data point
    assert_eq!(dec30_result.variant_name, "baseline");
    assert_eq!(dec30_result.count, 49);
    assert!(
        (dec30_result.avg_metric - 15.653061224489797).abs() < 1e-6,
        "avg_metric mismatch: {}",
        dec30_result.avg_metric
    );
    assert!(
        (dec30_result.stdev.expect("Result should contain stdev") - 5.9496174).abs() < 1e-5,
        "stdev mismatch",
    );
    assert!(
        (dec30_result
            .ci_error
            .expect("Result should contain ci_error")
            - 1.665892868041992)
            .abs()
            < 1e-6,
        "ci_error mismatch",
    );

    // All results should be for the baseline variant
    for result in &results {
        assert_eq!(
            result.variant_name, "baseline",
            "All results should be for baseline variant"
        );
    }
}
make_db_test!(test_get_variant_performances_ask_question_num_questions_with_variant);

async fn test_insert_boolean_feedback(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    // Use a known inference ID from the test database
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();
    let metric_name = format!("e2e_test_boolean_metric_{feedback_id}");

    let insert = BooleanMetricFeedbackInsert {
        id: feedback_id,
        target_id,
        metric_name: metric_name.clone(),
        value: true,
        tags: {
            let mut tags = HashMap::new();
            tags.insert("test_tag".to_string(), "test_value".to_string());
            tags
        },
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_boolean_feedback(&insert)
        .await
        .expect("Boolean feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback
    let found = feedback.iter().any(|f| match f {
        FeedbackRow::Boolean(b) => b.id == feedback_id && b.metric_name == metric_name,
        _ => false,
    });
    assert!(found, "Should find the inserted boolean feedback");
}
make_db_test!(test_insert_boolean_feedback);

async fn test_insert_float_feedback(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    // Use a known inference ID from the test database
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();
    let metric_name = format!("e2e_test_float_metric_{feedback_id}");

    let insert = FloatMetricFeedbackInsert {
        id: feedback_id,
        target_id,
        metric_name: metric_name.clone(),
        value: 0.87,
        tags: HashMap::new(),
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_float_feedback(&insert)
        .await
        .expect("Float feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback
    let found = feedback.iter().any(|f| match f {
        FeedbackRow::Float(fl) => fl.id == feedback_id && fl.metric_name == metric_name,
        _ => false,
    });
    assert!(found, "Should find the inserted float feedback");
}
make_db_test!(test_insert_float_feedback);

async fn test_insert_comment_feedback_inference_level(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    // Use a known inference ID from the test database
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();
    let comment_value = format!("E2E test comment for inference {feedback_id}");

    let insert = CommentFeedbackInsert {
        id: feedback_id,
        target_id,
        target_type: CommentTargetType::Inference,
        value: comment_value.clone(),
        tags: HashMap::new(),
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_comment_feedback(&insert)
        .await
        .expect("Comment feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback
    let found = feedback.iter().any(|f| match f {
        FeedbackRow::Comment(c) => c.id == feedback_id && c.value == comment_value,
        _ => false,
    });
    assert!(found, "Should find the inserted comment feedback");
}
make_db_test!(test_insert_comment_feedback_inference_level);

async fn test_insert_comment_feedback_episode_level(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    // Use a known episode ID from the test database
    let target_id = Uuid::parse_str("0192e14c-09b8-7d3e-8618-46aed8c213dc").unwrap();
    let comment_value = format!("E2E test comment for episode {feedback_id}");

    let insert = CommentFeedbackInsert {
        id: feedback_id,
        target_id,
        target_type: CommentTargetType::Episode,
        value: comment_value.clone(),
        tags: {
            let mut tags = HashMap::new();
            tags.insert("priority".to_string(), "high".to_string());
            tags
        },
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_comment_feedback(&insert)
        .await
        .expect("Comment feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback
    let found = feedback.iter().any(|f| match f {
        FeedbackRow::Comment(c) => c.id == feedback_id && c.value == comment_value,
        _ => false,
    });
    assert!(found, "Should find the inserted comment feedback");
}
make_db_test!(test_insert_comment_feedback_episode_level);

async fn test_insert_demonstration_feedback(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    // Use a known inference ID from the test database
    let inference_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();
    let demo_value = format!(r#"{{"content":"E2E test demonstration {feedback_id}"}}"#);

    let insert = DemonstrationFeedbackInsert {
        id: feedback_id,
        inference_id,
        value: demo_value.clone(),
        tags: HashMap::new(),
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_demonstration_feedback(&insert)
        .await
        .expect("Demonstration feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback using demonstration-specific query
    let feedback = conn
        .query_demonstration_feedback_by_inference_id(inference_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback
    let found = feedback
        .iter()
        .any(|d| d.id == feedback_id && d.value == demo_value);
    assert!(found, "Should find the inserted demonstration feedback");
}
make_db_test!(test_insert_demonstration_feedback);

async fn test_insert_static_eval_feedback(conn: impl FeedbackQueries + EvaluationQueries) {
    let feedback_id = Uuid::now_v7();
    let datapoint_id = Uuid::now_v7();
    let evaluator_inference_id = Uuid::now_v7();
    let metric_name = format!("e2e_test_quality_{feedback_id}");
    let output = format!("Test output for static evaluation {feedback_id}");

    let insert = StaticEvaluationHumanFeedbackInsert {
        feedback_id,
        metric_name: metric_name.clone(),
        datapoint_id,
        output: output.clone(),
        value: "0.95".to_string(),
        evaluator_inference_id: Some(evaluator_inference_id),
    };

    // Insert should succeed
    conn.insert_static_eval_feedback(&insert)
        .await
        .expect("Static eval feedback insert should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback using evaluation queries
    let feedback = conn
        .get_inference_evaluation_human_feedback(&metric_name, &datapoint_id, &output)
        .await
        .expect("Query should succeed");

    assert!(
        feedback.is_some(),
        "Should find the inserted static eval feedback"
    );
    let feedback = feedback.unwrap();
    assert_eq!(
        feedback.value,
        serde_json::json!(0.95),
        "Value should match"
    );
    assert_eq!(
        feedback.evaluator_inference_id, evaluator_inference_id,
        "Evaluator inference ID should match"
    );
}
// TODO(#5691): Implement after we support EvaluationQueries in Postgres.
make_clickhouse_only_test!(test_insert_static_eval_feedback);

async fn test_insert_static_eval_feedback_without_evaluator_inference_id(
    conn: impl FeedbackQueries + EvaluationQueries,
) {
    let feedback_id = Uuid::now_v7();
    let datapoint_id = Uuid::now_v7();
    let metric_name = format!("e2e_test_quality_no_evaluator_{feedback_id}");
    let output = format!("Test output without evaluator {feedback_id}");

    let insert = StaticEvaluationHumanFeedbackInsert {
        feedback_id,
        metric_name: metric_name.clone(),
        datapoint_id,
        output: output.clone(),
        value: "true".to_string(),
        evaluator_inference_id: None,
    };

    // Insert should succeed
    conn.insert_static_eval_feedback(&insert)
        .await
        .expect("Static eval feedback insert without evaluator_inference_id should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback using evaluation queries
    let feedback = conn
        .get_inference_evaluation_human_feedback(&metric_name, &datapoint_id, &output)
        .await
        .expect("Query should succeed");

    assert!(
        feedback.is_some(),
        "Should find the inserted static eval feedback"
    );
    let feedback = feedback.unwrap();
    assert_eq!(
        feedback.value,
        serde_json::json!(true),
        "Value should match"
    );
    // When evaluator_inference_id is not provided, ClickHouse uses default zero UUID
    assert_eq!(
        feedback.evaluator_inference_id,
        Uuid::nil(),
        "Evaluator inference ID should be nil UUID when not provided"
    );
}
// TODO(#5691): Implement after we support EvaluationQueries in Postgres.
make_clickhouse_only_test!(test_insert_static_eval_feedback_without_evaluator_inference_id);

/// Tests that `get_variant_performances` returns only the latest feedback per inference
/// when there are multiple feedbacks for the same inference (deduplication via DISTINCT ON).
///
/// This verifies the SQL semantics of:
/// `SELECT DISTINCT ON (target_id) ... ORDER BY target_id, created_at DESC`
/// which should keep only the latest feedback per target_id.
async fn test_get_variant_performances_distinct_on_semantics(conn: impl FeedbackQueries) {
    // Use the inference ID from fixture data for write_haiku function
    let target_id = Uuid::parse_str("0196c682-72e0-7c83-a92b-9d1a3c7630f2").unwrap();
    let unique_metric_name = format!("e2e_distinct_on_test_{}", Uuid::now_v7());

    // Insert first feedback with value 1.0
    let first_feedback_id = Uuid::now_v7();
    let first_insert = FloatMetricFeedbackInsert {
        id: first_feedback_id,
        target_id,
        metric_name: unique_metric_name.clone(),
        value: 1.0,
        tags: HashMap::new(),
        snapshot_hash: SnapshotHash::new_test(),
    };
    conn.insert_float_feedback(&first_insert)
        .await
        .expect("First feedback insert should succeed");

    // Sleep to ensure different `created_at` timestamps
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Insert second feedback with value 5.0 (this should be the one kept)
    let second_feedback_id = Uuid::now_v7();
    let second_insert = FloatMetricFeedbackInsert {
        id: second_feedback_id,
        target_id,
        metric_name: unique_metric_name.clone(),
        value: 5.0,
        tags: HashMap::new(),
        snapshot_hash: SnapshotHash::new_test(),
    };
    conn.insert_float_feedback(&second_insert)
        .await
        .expect("Second feedback insert should succeed");

    // Wait for database to process (especially for ClickHouse)
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query variant performances
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };

    let params = GetVariantPerformanceParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: &unique_metric_name,
        metric_config: &metric_config,
        time_window: TimeWindow::Cumulative,
        variant_name: Some("initial_prompt_gpt4o_mini"),
    };

    let results = conn.get_variant_performances(params).await.unwrap();

    println!("DISTINCT ON test results: {results:?}");

    // Should have exactly one result (one inference with deduplicated feedback)
    assert_eq!(
        results.len(),
        1,
        "Should have exactly one result for the variant"
    );

    let result = &results[0];
    assert_eq!(
        result.variant_name, "initial_prompt_gpt4o_mini",
        "Result should be for the expected variant"
    );
    assert_eq!(
        result.count, 1,
        "Count should be 1 since DISTINCT ON keeps only one feedback per target_id"
    );
    assert!(
        (result.avg_metric - 5.0).abs() < 1e-6,
        "avg_metric should be 5.0 (the later feedback value), got {}",
        result.avg_metric
    );
}
make_db_test!(test_get_variant_performances_distinct_on_semantics);

async fn test_insert_feedback_with_multiple_tags(conn: impl FeedbackQueries) {
    let feedback_id = Uuid::now_v7();
    let target_id = Uuid::parse_str("0192e14c-09b8-738c-970e-c0bb29429e3e").unwrap();
    let metric_name = format!("e2e_test_with_tags_{feedback_id}");

    let mut tags = HashMap::new();
    tags.insert("user_id".to_string(), "user_12345".to_string());
    tags.insert("session_id".to_string(), "session_abc".to_string());
    tags.insert("source".to_string(), "e2e_test".to_string());
    tags.insert("environment".to_string(), "test".to_string());

    let insert = BooleanMetricFeedbackInsert {
        id: feedback_id,
        target_id,
        metric_name: metric_name.clone(),
        value: false,
        tags: tags.clone(),
        snapshot_hash: SnapshotHash::new_test(),
    };

    // Insert should succeed
    conn.insert_boolean_feedback(&insert)
        .await
        .expect("Feedback insert with multiple tags should succeed");

    // Wait for ClickHouse to process the insert
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read back the feedback
    let feedback = conn
        .query_feedback_by_target_id(target_id, None, None, Some(1000))
        .await
        .expect("Query should succeed");

    // Find our inserted feedback and verify tags
    let found = feedback.iter().find(|f| match f {
        FeedbackRow::Boolean(b) => b.id == feedback_id && b.metric_name == metric_name,
        _ => false,
    });
    assert!(
        found.is_some(),
        "Should find the inserted feedback with tags"
    );

    if let Some(FeedbackRow::Boolean(b)) = found {
        assert_eq!(
            b.tags.get("user_id"),
            Some(&"user_12345".to_string()),
            "user_id tag should match"
        );
        assert_eq!(
            b.tags.get("session_id"),
            Some(&"session_abc".to_string()),
            "session_id tag should match"
        );
        assert_eq!(
            b.tags.get("source"),
            Some(&"e2e_test".to_string()),
            "source tag should match"
        );
        assert_eq!(
            b.tags.get("environment"),
            Some(&"test".to_string()),
            "environment tag should match"
        );
    }
}
make_db_test!(test_insert_feedback_with_multiple_tags);
