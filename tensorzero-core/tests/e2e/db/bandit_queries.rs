#![expect(clippy::print_stdout)]
use tensorzero_core::db::{clickhouse::test_helpers::get_clickhouse, FeedbackQueries};

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
