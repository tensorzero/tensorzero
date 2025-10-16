#![expect(clippy::print_stdout)]
use tensorzero_core::db::{clickhouse::test_helpers::get_clickhouse, SelectQueries};

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
    assert_float_eq(metric.variance, 0.10703, None);
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
    assert_float_eq(metric.variance, 0.22807, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "turbo");
    assert_eq!(metric.count, 35);
    assert_float_eq(metric.mean, 0.65714, None);
    assert_float_eq(metric.variance, 0.23193, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "baseline");
    assert_eq!(metric.count, 25);
    assert_float_eq(metric.mean, 0.2, None);
    assert_float_eq(metric.variance, 0.16666667, None);
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
    assert_float_eq(metric.variance, 0.22535211, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-nano");
    assert_eq!(metric.count, 49);
    assert_float_eq(metric.mean, 0.4489796, None);
    assert_float_eq(metric.variance, 0.25255102, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-mini");
    assert_eq!(metric.count, 3);
    assert_float_eq(metric.mean, 1.0, None);
    assert_float_eq(metric.variance, 0.0, None);
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
    assert_float_eq(metric.variance, 443305500.0, None);

    let metric = metrics_by_variant.get(1).unwrap();
    assert_eq!(metric.variant_name, "baseline");
    assert_eq!(metric.count, 48);
    assert_float_eq(metric.mean, 118620.79, None);
    assert_float_eq(metric.variance, 885428200.0, None);

    let metric = metrics_by_variant.get(2).unwrap();
    assert_eq!(metric.variant_name, "gpt-4.1-mini");
    assert_eq!(metric.count, 3);
    assert_float_eq(metric.mean, 65755.3, None);
    assert_float_eq(metric.variance, 22337140.0, None);
}

#[tokio::test]
async fn test_clickhouse_get_feedback_timeseries_minute_level() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test minute-level aggregation (1 minute intervals)
    // Fixture data is from April 2025, so we need to look back far enough
    let feedback_timeseries = clickhouse
        .get_feedback_timeseries(
            function_name,
            metric_name,
            None,
            1,      // 1 minute intervals
            525600, // Look back a year for data (365 days * 24 hours * 60 minutes)
        )
        .await
        .unwrap();

    println!("Minute-level data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Minute: {point:?}");
    }

    // Verify basic properties
    // CROSS JOIN generates all combinations, but INNER JOIN filters to only variants with data
    // Period 1-2: 2 variants each (gpt4o_mini, llama_8b)
    // Period 3-4: 3 variants each (all variants now have data)
    // Total: 2 + 2 + 3 + 3 = 10 points
    assert_eq!(
        feedback_timeseries.len(),
        10,
        "Should have 10 data points across all variants and periods"
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
        assert!(!point.variance.is_nan());
        assert!(point.variance >= 0.0);
    }

    // Verify final cumulative values for each variant (last period for each variant)
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(gpt4o_initial_prompt.variance, 0.255_516_84, Some(1e-6));

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(gpt4o_mini_initial_prompt.variance, 0.094_610_54, Some(1e-6));

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(llama_8b_initial_prompt.variance, 0.231_152_2, Some(1e-6));
}

#[tokio::test]
async fn test_clickhouse_get_feedback_timeseries_hourly() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test hourly aggregation (60 minute intervals)
    let feedback_timeseries = clickhouse
        .get_feedback_timeseries(
            function_name,
            metric_name,
            None,
            60,   // 60 minute intervals
            8760, // Look back a year (365 days * 24 hours)
        )
        .await
        .unwrap();

    println!("Hourly data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Hourly: {point:?}");
    }

    // Verify basic properties
    // CROSS JOIN generates all combinations, but INNER JOIN filters to only variants with data
    // Period 1 (23:00): 2 variants (gpt4o_mini, llama_8b)
    // Period 2 (00:00): 3 variants (gpt4o_initial_prompt now has data)
    // Period 3 (03:00): 3 variants
    // Total: 2 + 3 + 3 = 8 points
    assert_eq!(
        feedback_timeseries.len(),
        8,
        "Should have 8 data points across all variants and periods"
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
        assert!(!point.variance.is_nan());
        assert!(point.variance >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(gpt4o_initial_prompt.variance, 0.255_516_84, Some(1e-6));

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(gpt4o_mini_initial_prompt.variance, 0.094_610_54, Some(1e-6));

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(llama_8b_initial_prompt.variance, 0.231_152_2, Some(1e-6));
}

#[tokio::test]
async fn test_clickhouse_get_feedback_timeseries_daily() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test daily aggregation (1440 minutes = 1 day)
    let feedback_timeseries = clickhouse
        .get_feedback_timeseries(
            function_name,
            metric_name,
            None,
            1440, // 1440 minutes = 1 day
            365,  // Look back a year
        )
        .await
        .unwrap();

    println!("Daily data points: {}", feedback_timeseries.len());
    for point in &feedback_timeseries {
        println!("Daily: {point:?}");
    }

    // Verify basic properties
    // With CROSS JOIN, we get all variants × all periods = 3 variants × 2 periods = 6 points
    assert_eq!(
        feedback_timeseries.len(),
        6,
        "Should have 6 data points (3 variants × 2 periods)"
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
        assert!(!point.variance.is_nan());
        assert!(point.variance >= 0.0);
    }

    // Verify final cumulative values for each variant
    let gpt4o_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_initial_prompt.count, 42);
    assert_float_eq(gpt4o_initial_prompt.mean, 0.523_809_5, Some(1e-6));
    assert_float_eq(gpt4o_initial_prompt.variance, 0.255_516_84, Some(1e-6));

    let gpt4o_mini_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "gpt4o_mini_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(gpt4o_mini_initial_prompt.count, 124);
    assert_float_eq(gpt4o_mini_initial_prompt.mean, 0.104_838_71, Some(1e-6));
    assert_float_eq(gpt4o_mini_initial_prompt.variance, 0.094_610_54, Some(1e-6));

    let llama_8b_initial_prompt = feedback_timeseries
        .iter()
        .filter(|p| p.variant_name == "llama_8b_initial_prompt")
        .max_by_key(|p| p.period_end)
        .unwrap();
    assert_eq!(llama_8b_initial_prompt.count, 38);
    assert_float_eq(llama_8b_initial_prompt.mean, 0.342_105_26, Some(1e-6));
    assert_float_eq(llama_8b_initial_prompt.variance, 0.231_152_2, Some(1e-6));
}

#[tokio::test]
async fn test_clickhouse_get_feedback_timeseries_with_variant_filter() {
    let clickhouse = get_clickhouse().await;
    let function_name = "extract_entities".to_string();
    let metric_name =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match".to_string();

    // Test with specific variant (filter to only one)
    let feedback_timeseries = clickhouse
        .get_feedback_timeseries(
            function_name.clone(),
            metric_name.clone(),
            Some(vec!["gpt4o_mini_initial_prompt".to_string()]),
            60,    // 60 minutes = 1 hour intervals
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
    assert_float_eq(final_point.variance, 0.094_610_54, Some(1e-6));

    // Test with empty variant list - should return empty result
    let empty_result = clickhouse
        .get_feedback_timeseries(
            function_name,
            metric_name,
            Some(vec![]),
            60,    // 60 minutes = 1 hour intervals
            87600, // Look back 10 years
        )
        .await
        .unwrap();
    assert!(
        empty_result.is_empty(),
        "Empty variant list should return empty result"
    );
}
