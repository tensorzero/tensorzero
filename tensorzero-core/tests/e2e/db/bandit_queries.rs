#![expect(clippy::print_stdout)]
use tensorzero_core::db::{
    clickhouse::test_helpers::get_clickhouse, feedback::FeedbackQueries, TimeWindow,
};

fn assert_float_eq(actual: f32, expected: f32, epsilon: Option<f32>) {
    let epsilon = epsilon.unwrap_or(1e-4);
    assert!(
        (actual - expected).abs() < epsilon,
        "actual: {actual}, expected: {expected}",
    );
}

#[tokio::test]
async fn test_clickhouse_metrics_by_variant_singleton() {
    let clickhouse = get_clickhouse().await;
    let metrics_by_variant = clickhouse
        .get_feedback_by_variant("haiku_score_episode", "write_haiku", None)
        .await
        .unwrap();
    assert_eq!(metrics_by_variant.len(), 1);
    let metric = metrics_by_variant.first().unwrap();
    assert_eq!(metric.variant_name, "initial_prompt_gpt4o_mini");
    assert_float_eq(metric.mean, 0.12, None);
    assert_float_eq(metric.variance.unwrap(), 0.10703, None);
    assert_eq!(metric.count, 75);
}

#[tokio::test]
async fn test_clickhouse_metrics_by_variant_filter_all() {
    let clickhouse = get_clickhouse().await;
    let metrics_by_variant = clickhouse
        .get_feedback_by_variant(
            "haiku_score_episode",
            "write_haiku",
            // nonexistent so there should be no results
            Some(&vec!["foo".to_string()]),
        )
        .await
        .unwrap();
    assert!(metrics_by_variant.is_empty());
}

#[tokio::test]
async fn test_clickhouse_metrics_by_variant_many_results() {
    let clickhouse = get_clickhouse().await;
    let metrics_by_variant = clickhouse
        .get_feedback_by_variant("exact_match", "extract_entities", None)
        .await
        .unwrap();
    println!("metrics_by_variant: {metrics_by_variant:?}");
    assert_eq!(metrics_by_variant.len(), 3);
    // Sort by count in descending order for deterministic results
    let mut metrics_by_variant = metrics_by_variant;
    metrics_by_variant.sort_by(|a, b| b.count.cmp(&a.count));
    let metric = metrics_by_variant.first().unwrap();
    assert_eq!(metric.variant_name, "dicl");
    assert_eq!(metric.count, 39);
    assert_float_eq(metric.mean, 0.33333333, None);
    assert_float_eq(metric.variance.unwrap(), 0.22807, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "turbo");
    assert_eq!(metric.count, 35);
    assert_float_eq(metric.mean, 0.65714, None);
    assert_float_eq(metric.variance.unwrap(), 0.23193, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "baseline");
    assert_eq!(metric.count, 25);
    assert_float_eq(metric.mean, 0.2, None);
    assert_float_eq(metric.variance.unwrap(), 0.16666667, None);
}

#[tokio::test]
async fn test_clickhouse_metrics_by_variant_episode_boolean() {
    let clickhouse = get_clickhouse().await;
    let metrics_by_variant = clickhouse
        .get_feedback_by_variant("solved", "ask_question", None)
        .await
        .unwrap();
    // Sort by count in descending order for deterministic results
    let mut metrics_by_variant = metrics_by_variant;
    metrics_by_variant.sort_by(|a, b| b.count.cmp(&a.count));
    println!("metrics_by_variant: {metrics_by_variant:?}");
    assert_eq!(metrics_by_variant.len(), 3);
    let metric = metrics_by_variant.first().unwrap();
    assert_eq!(metric.variant_name, "baseline");
    assert_eq!(metric.count, 72);
    assert_float_eq(metric.mean, 0.33333334, None);
    assert_float_eq(metric.variance.unwrap(), 0.22535211, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-nano");
    assert_eq!(metric.count, 49);
    assert_float_eq(metric.mean, 0.4489796, None);
    assert_float_eq(metric.variance.unwrap(), 0.25255102, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-mini");
    assert_eq!(metric.count, 3);
    assert_float_eq(metric.mean, 1.0, None);
    assert_float_eq(metric.variance.unwrap(), 0.0, None);
}

#[tokio::test]
async fn test_clickhouse_metrics_by_variant_episode_float() {
    let clickhouse = get_clickhouse().await;
    let metrics_by_variant = clickhouse
        .get_feedback_by_variant("elapsed_ms", "ask_question", None)
        .await
        .unwrap();
    // Sort by count in descending order for deterministic results
    let mut metrics_by_variant = metrics_by_variant;
    metrics_by_variant.sort_by(|a, b| b.count.cmp(&a.count));
    println!("metrics_by_variant: {metrics_by_variant:?}");
    assert_eq!(metrics_by_variant.len(), 3);
    let metric = metrics_by_variant.first().unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-nano");
    assert_eq!(metric.count, 49);
    assert_float_eq(metric.mean, 91678.72, None);
    assert_float_eq(metric.variance.unwrap(), 443305500.0, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "baseline");
    assert_eq!(metric.count, 48);
    assert_float_eq(metric.mean, 118620.79, None);
    assert_float_eq(metric.variance.unwrap(), 885428200.0, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-mini");
    assert_eq!(metric.count, 3);
    assert_float_eq(metric.mean, 65755.3, None);
    assert_float_eq(metric.variance.unwrap(), 22337140.0, None);
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_minute_level() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test minute-level aggregation
    // Fixture data is from April 2025, so we need to look back far enough
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Minute,
            525600, // Look back a year for data (365 days * 24 hours * 60 minutes)
        )
        .await
        .unwrap();

    println!("Minute-level data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Minute: {point:?}");
    }

    // Data that's expected:
    // - Period 1 (22:46): 1 variant (llama_8b)
    // - Period 2 (22:47): 2 variants (gpt4o_mini, llama_8b)
    // - Period 3 (23:08): 2 variants (gpt4o_initial_prompt, gpt4o_mini)
    // - Period 4 (02:35): 1 variant (gpt4o_mini)
    // Total: 1 + 2 + 2 + 1 = 6 points
    assert_eq!(
        feedback_timeseries.len(),
        6,
        "Should have 6 data points across all variants and periods"
    );

    // Count unique time periods
    let periods: std::collections::HashSet<_> =
        feedback_timeseries.iter().map(|p| p.period_end).collect();
    assert_eq!(periods.len(), 4, "Should have 4 unique time periods");

    // Verify all data points have valid values
    for point in &feedback_timeseries {
        assert!(!point.variant_name.is_empty());
        assert!(point.count > 0);
        assert!(!point.mean.is_nan());
        assert!(!point.variance.unwrap().is_nan());
        assert!(point.variance.unwrap() >= 0.0);
    }

    // Verify final cumulative values for each variant (last period for each variant)
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(
        gpt4o_initial_prompt.variance.unwrap(),
        0.255_516_84,
        Some(1e-6),
    );

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(
        gpt4o_mini_initial_prompt.variance.unwrap(),
        0.094_610_54,
        Some(1e-6),
    );

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(
        llama_8b_initial_prompt.variance.unwrap(),
        0.231_152_2,
        Some(1e-6),
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_hourly() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test hourly aggregation
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Hour,
            8760, // Look back a year (365 days * 24 hours)
        )
        .await
        .unwrap();

    println!("Hourly data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Hourly: {point:?}");
    }

    // Data that's expected:
    // - Period 1 (23:00): 2 variants (gpt4o_mini, llama_8b)
    // - Period 2 (00:00): 2 variants (gpt4o_initial_prompt, gpt4o_mini)
    // - Period 3 (03:00): 1 variant (gpt4o_mini)
    // Total: 2 + 2 + 1 = 5 points
    assert_eq!(
        feedback_timeseries.len(),
        5,
        "Should have 5 data points across all variants and periods"
    );

    // Count unique time periods
    let periods: std::collections::HashSet<_> =
        feedback_timeseries.iter().map(|p| p.period_end).collect();
    assert_eq!(periods.len(), 3, "Should have 3 unique hourly periods");

    // Verify all data points have valid values
    for point in &feedback_timeseries {
        assert!(!point.variant_name.is_empty());
        assert!(point.count > 0);
        assert!(!point.mean.is_nan());
        assert!(!point.variance.unwrap().is_nan());
        assert!(point.variance.unwrap() >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(
        gpt4o_initial_prompt.variance.unwrap(),
        0.255_516_84,
        Some(1e-6),
    );

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(
        gpt4o_mini_initial_prompt.variance.unwrap(),
        0.094_610_54,
        Some(1e-6),
    );

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(
        llama_8b_initial_prompt.variance.unwrap(),
        0.231_152_2,
        Some(1e-6),
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_daily() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test daily aggregation
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Day,
            365, // Look back a year
        )
        .await
        .unwrap();

    println!("Daily data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Daily: {point:?}");
    }

    // Data that's expected:
    // - Period 1 (2025-04-15): 3 variants (gpt4o_initial_prompt, gpt4o_mini, , llama_8b)
    // - Period 2 (2025-04-16): 1 variant (only gpt4o_mini has new data)
    // Total: 3 + 1 = 4 points
    assert_eq!(
        feedback_timeseries.len(),
        4,
        "Should have 4 data points across variants and periods"
    );

    // Count unique time periods
    let periods: std::collections::HashSet<_> =
        feedback_timeseries.iter().map(|p| p.period_end).collect();
    assert_eq!(periods.len(), 2, "Should have 2 unique daily periods");

    // Verify all data points have valid values
    for point in &feedback_timeseries {
        assert!(!point.variant_name.is_empty());
        assert!(point.count > 0);
        assert!(!point.mean.is_nan());
        assert!(!point.variance.unwrap().is_nan());
        assert!(point.variance.unwrap() >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(
        gpt4o_initial_prompt.variance.unwrap(),
        0.255_516_84,
        Some(1e-6),
    );

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(
        gpt4o_mini_initial_prompt.variance.unwrap(),
        0.094_610_54,
        Some(1e-6),
    );

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(
        llama_8b_initial_prompt.variance.unwrap(),
        0.231_152_2,
        Some(1e-6),
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_weekly() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test weekly aggregation
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Week,
            52, // Look back a year (52 weeks)
        )
        .await
        .unwrap();

    println!("Weekly data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Weekly: {point:?}");
    }

    // All data is within a single week, so we expect one weekly period with 3 variants
    assert_eq!(
        feedback_timeseries.len(),
        3,
        "Should have 3 data points (one per variant in a single week)"
    );

    // Count unique time periods
    let periods: std::collections::HashSet<_> =
        feedback_timeseries.iter().map(|p| p.period_end).collect();
    assert_eq!(periods.len(), 1, "Should have 1 unique weekly period");

    // Verify all data points have valid values
    for point in &feedback_timeseries {
        assert!(!point.variant_name.is_empty());
        assert!(point.count > 0);
        assert!(!point.mean.is_nan());
        assert!(!point.variance.unwrap().is_nan());
        assert!(point.variance.unwrap() >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(
        gpt4o_initial_prompt.variance.unwrap(),
        0.255_516_84,
        Some(1e-6),
    );

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(
        gpt4o_mini_initial_prompt.variance.unwrap(),
        0.094_610_54,
        Some(1e-6),
    );

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(
        llama_8b_initial_prompt.variance.unwrap(),
        0.231_152_2,
        Some(1e-6),
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_monthly() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test monthly aggregation
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Month,
            12, // Look back a year (12 months)
        )
        .await
        .unwrap();

    println!("Monthly data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Monthly: {point:?}");
    }

    // All data is within a single month, so we expect one monthly period with 3 variants
    assert_eq!(
        feedback_timeseries.len(),
        3,
        "Should have 3 data points (one per variant in a single month)"
    );

    // Count unique time periods
    let periods: std::collections::HashSet<_> =
        feedback_timeseries.iter().map(|p| p.period_end).collect();
    assert_eq!(periods.len(), 1, "Should have 1 unique monthly period");

    // Verify all data points have valid values
    for point in &feedback_timeseries {
        assert!(!point.variant_name.is_empty());
        assert!(point.count > 0);
        assert!(!point.mean.is_nan());
        assert!(!point.variance.unwrap().is_nan());
        assert!(point.variance.unwrap() >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(
        gpt4o_initial_prompt.variance.unwrap(),
        0.255_516_84,
        Some(1e-6),
    );

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(
        gpt4o_mini_initial_prompt.variance.unwrap(),
        0.094_610_54,
        Some(1e-6),
    );

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(
        llama_8b_initial_prompt.variance.unwrap(),
        0.231_152_2,
        Some(1e-6),
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_cumulative_returns_error() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test that Cumulative time window returns an error
    let result = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Cumulative,
            1, // max_periods doesn't matter for this test
        )
        .await;

    assert!(
        result.is_err(),
        "TimeWindow::Cumulative should return an error"
    );

    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("Cumulative time window is not supported"),
        "Error message should indicate Cumulative is not supported, got: {error_message}"
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_with_variant_filter() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test with specific variant (filter to only one)
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name.clone(),
            metric_name.clone(),
            Some(vec!["gpt4o_mini_initial_prompt".to_string()]),
            TimeWindow::Hour,
            87600, // Look back 10 years (10 years * 365 days * 24 hours)
        )
        .await
        .unwrap();

    for point in &feedback_timeseries {
        println!("{point:?}");
    }

    // Should only have the specified variant
    for point in &feedback_timeseries {
        assert!(
            point.variant_name == "gpt4o_mini_initial_prompt",
            "Should only contain gpt4o_mini_initial_prompt, found: {}",
            point.variant_name
        );
    }

    // Verify we have the expected number of periods for this variant (3 hourly periods)
    assert_eq!(
        feedback_timeseries.len(),
        3,
        "gpt4o_mini_initial_prompt should have 3 hourly periods"
    );

    // Verify the variant is present
    let variant_names: std::collections::HashSet<_> = feedback_timeseries
        .iter()
        .map(|p| p.variant_name.as_str())
        .collect();
    assert!(variant_names.contains(&"gpt4o_mini_initial_prompt"));
    // Should NOT contain the filtered-out variants
    assert!(!variant_names.contains(&"gpt4o_initial_prompt"));
    assert!(!variant_names.contains(&"llama_8b_initial_prompt"));

    // Verify final cumulative value matches what we expect
    let final_point = feedback_timeseries
        .iter()
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(final_point.count, 124);
    assert_float_eq(final_point.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(final_point.variance.unwrap(), 0.094_610_54, Some(1e-6));

    // Test with empty variant list - should return empty result
    let empty_result = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            Some(vec![]),
            TimeWindow::Hour,
            87600, // Look back 10 years
        )
        .await
        .unwrap();
    assert!(
        empty_result.is_empty(),
        "Empty variant list should return empty result"
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_includes_baseline_values() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Use a small max_periods (2) to create a window that starts after some data
    // The fixture data spans from 2025-04-14 22:46 to 2025-04-16 02:35
    // With daily granularity and max_periods=2, we should only see:
    // - 2025-04-15 (most recent - 1)
    // - 2025-04-16 (most recent)
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Day,
            2, // Only last 2 days
        )
        .await
        .unwrap();

    println!("Baseline test - data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!(
            "Point: period={}, variant={}, count={}",
            point.period_end, point.variant_name, point.count
        );
    }

    // We expect baseline entries at the window start (2025-04-15) for all 3 variants
    // PLUS the actual data points that fall within the window
    // Expected structure:
    // - 2025-04-15: 3 baseline entries (one per variant) + actual data for all 3 variants
    // - 2025-04-16: actual data for 1 variant (gpt4o_mini)

    // Count data points at the earliest period (window start)
    let earliest_period = feedback_timeseries
        .iter()
        .map(|p| &p.period_end)
        .min()
        .unwrap();

    let window_start_points: Vec<_> = feedback_timeseries
        .iter()
        .filter(|p| p.period_end == *earliest_period)
        .collect();

    println!("Window start period: {earliest_period}");
    println!("Points at window start: {}", window_start_points.len());

    // All 3 variants should have data at the window start
    // This ensures we have baseline values for all variants
    assert_eq!(
        window_start_points.len(),
        3,
        "Should have baseline data for all 3 variants at window start"
    );

    // Verify each variant has a baseline value
    let variant_names_at_start: std::collections::HashSet<_> = window_start_points
        .iter()
        .map(|p| p.variant_name.as_str())
        .collect();

    assert!(variant_names_at_start.contains("gpt4o_initial_prompt"));
    assert!(variant_names_at_start.contains("gpt4o_mini_initial_prompt"));
    assert!(variant_names_at_start.contains("llama_8b_initial_prompt"));

    // All baseline values should have non-zero counts
    // (because all variants had data before the window)
    for point in &window_start_points {
        assert!(
            point.count > 0,
            "Baseline count for {} should be > 0, got {}",
            point.variant_name,
            point.count
        );
    }
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_includes_silent_variants() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Use a window that only includes 2025-04-16
    // On this day, only gpt4o_mini_initial_prompt has new data
    // But llama_8b_initial_prompt and gpt4o_initial_prompt should still appear
    // with their baseline values from 2025-04-14
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Day,
            1, // Only last 1 day (2025-04-16)
        )
        .await
        .unwrap();

    println!(
        "Silent variants test - data points: {}",
        feedback_timeseries.len()
    );
    for point in &feedback_timeseries {
        println!(
            "Point: period={}, variant={}, count={}",
            point.period_end, point.variant_name, point.count
        );
    }

    // We should have all 3 variants in the result
    let variant_names: std::collections::HashSet<_> = feedback_timeseries
        .iter()
        .map(|p| p.variant_name.as_str())
        .collect();

    assert_eq!(
        variant_names.len(),
        3,
        "Should include all 3 variants even though only 1 has new data in window"
    );

    assert!(
        variant_names.contains("gpt4o_initial_prompt"),
        "Should include gpt4o_initial_prompt even with no new data"
    );
    assert!(
        variant_names.contains("llama_8b_initial_prompt"),
        "Should include llama_8b_initial_prompt even with no new data"
    );
    assert!(
        variant_names.contains("gpt4o_mini_initial_prompt"),
        "Should include gpt4o_mini_initial_prompt with new data"
    );

    // Variants with no new data should have baseline counts
    let gpt4o_initial_prompt_count = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .map(|p| p.count)
        .max()
        .unwrap();

    let llama_8b_initial_prompt_count = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .map(|p| p.count)
        .max()
        .unwrap();

    // These should have their full historical counts (from before the window)
    assert_eq!(
        gpt4o_initial_prompt_count, 42,
        "gpt4o_initial_prompt should have baseline count of 42"
    );
    assert_eq!(
        llama_8b_initial_prompt_count, 38,
        "llama_8b_initial_prompt should have baseline count of 38"
    );
}

#[tokio::test]
async fn test_clickhouse_get_cumulative_feedback_timeseries_baseline_continuity() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Get a narrow window (hourly with max_periods=3)
    // This should show proper cumulative progression
    let feedback_timeseries = clickhouse
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            None,
            TimeWindow::Hour,
            3, // Only last 3 hours
        )
        .await
        .unwrap();

    println!(
        "Baseline continuity test - data points: {}",
        feedback_timeseries.len()
    );
    for point in &feedback_timeseries {
        println!(
            "Point: period={}, variant={}, count={}",
            point.period_end, point.variant_name, point.count
        );
    }

    // For each variant, verify cumulative counts are monotonically increasing
    let variants = [
        "gpt4o_initial_prompt",
        "gpt4o_mini_initial_prompt",
        "llama_8b_initial_prompt",
    ];

    for variant in variants {
        let mut variant_points: Vec<_> = feedback_timeseries
            .iter()
            .filter(|p| p.variant_name == variant)
            .collect();

        // Sort by period
        variant_points.sort_by(|a, b| a.period_end.cmp(&b.period_end));

        if variant_points.len() > 1 {
            // Verify counts are non-decreasing (cumulative)
            for window in variant_points.windows(2) {
                assert!(
                    window[1].count >= window[0].count,
                    "Cumulative count for {} should be non-decreasing: {} -> {}",
                    variant,
                    window[0].count,
                    window[1].count
                );
            }

            // First point should have baseline count (not zero)
            assert!(
                variant_points[0].count > 0,
                "First point for {} should have non-zero baseline count, got {}",
                variant,
                variant_points[0].count
            );
        }
    }
}
