//! Functions endpoint for querying function-level information

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::{Config, MetricConfigType};
use crate::db::TimeWindow;
use crate::db::feedback::{
    FeedbackQueries, GetVariantPerformanceParams, MetricType, MetricWithFeedback,
    VariantPerformanceRow,
};
use crate::error::{Error, ErrorDetails};
use crate::function::get_function;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the metrics endpoint
#[derive(Debug, Deserialize)]
pub struct MetricsQueryParams {
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
}

/// Response containing metrics with feedback statistics
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MetricsWithFeedbackResponse {
    /// Metrics with feedback statistics
    pub metrics: Vec<MetricWithFeedback>,
}

/// HTTP handler for getting metrics with feedback for a function
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_function_metrics_handler",
    skip_all,
    fields(
        function_name = %function_name,
    )
)]
pub async fn get_function_metrics_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<MetricsQueryParams>,
) -> Result<Json<MetricsWithFeedbackResponse>, Error> {
    let response = get_function_metrics(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        &function_name,
        params.variant_name.as_deref(),
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting metrics with feedback
pub async fn get_function_metrics(
    config: &Config,
    clickhouse: &impl FeedbackQueries,
    function_name: &str,
    variant_name: Option<&str>,
) -> Result<MetricsWithFeedbackResponse, Error> {
    // Get function config to determine the inference table
    let function_config = get_function(&config.functions, function_name)?;

    // Query metrics with feedback
    let metrics = clickhouse
        .query_metrics_with_feedback(function_name, &function_config, variant_name)
        .await?;

    // Enrich metric_type from config for metrics that don't have it set
    let metrics =
        metrics
            .into_iter()
            .map(|mut metric| {
                if metric.metric_type.is_none() {
                    // Look up metric type from config
                    metric.metric_type = config.metrics.get(&metric.metric_name).map(|mc| match mc
                        .r#type
                    {
                        MetricConfigType::Boolean => MetricType::Boolean,
                        MetricConfigType::Float => MetricType::Float,
                    });
                }
                metric
            })
            .collect();

    Ok(MetricsWithFeedbackResponse { metrics })
}

/// Query parameters for the variant performances endpoint
#[derive(Debug, Deserialize)]
pub struct VariantPerformancesQueryParams {
    /// The metric name to compute performance for
    pub metric_name: String,
    /// Time granularity for grouping performance data (minute, hour, day, week, month, cumulative)
    pub time_window: TimeWindow,
    /// Optional variant name to filter by
    pub variant_name: Option<String>,
}

/// Response containing variant performance statistics
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VariantPerformancesResponse {
    /// Performance statistics for each (variant, time_period) combination
    pub performances: Vec<VariantPerformanceRow>,
}

/// HTTP handler for getting variant performance statistics for a function and metric
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_variant_performances_handler",
    skip_all,
    fields(
        function_name = %function_name,
        metric_name = %params.metric_name,
    )
)]
pub async fn get_variant_performances_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<VariantPerformancesQueryParams>,
) -> Result<Json<VariantPerformancesResponse>, Error> {
    let response = get_variant_performances(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        &function_name,
        &params.metric_name,
        params.time_window,
        params.variant_name.as_deref(),
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting variant performance statistics.
///
/// Validates function and metric exist in config, then queries ClickHouse for
/// performance statistics grouped by variant and time period.
pub async fn get_variant_performances(
    config: &Config,
    clickhouse: &impl FeedbackQueries,
    function_name: &str,
    metric_name: &str,
    time_window: TimeWindow,
    variant_name: Option<&str>,
) -> Result<VariantPerformancesResponse, Error> {
    // Get function config to determine the function type
    let function_config = get_function(&config.functions, function_name)?;
    let function_type = function_config.config_type();

    // Get metric config to determine the metric type and level
    let metric_config = config.metrics.get(metric_name).ok_or_else(|| {
        Error::new(ErrorDetails::UnknownMetric {
            name: metric_name.to_string(),
        })
    })?;

    // If variant_name is provided, validate that it exists
    if let Some(variant) = variant_name
        && !function_config.variants().contains_key(variant)
    {
        return Err(ErrorDetails::UnknownVariant {
            name: variant.to_string(),
        }
        .into());
    }

    let params = GetVariantPerformanceParams {
        function_name,
        function_type,
        metric_name,
        metric_config,
        time_window,
        variant_name,
    };

    let performances = clickhouse.get_variant_performances(params).await?;
    Ok(VariantPerformancesResponse { performances })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ConfigFileGlob};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_get_function_metrics_function_not_found() {
        let config = Config::default();
        // Empty config with no functions should return error
        // We can't actually test this without a real ClickHouse connection,
        // so we just test that get_function fails
        let result = get_function(&config.functions, "nonexistent_function");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }

    #[tokio::test]
    async fn test_get_function_determines_correct_inference_table_for_chat() {
        let config_str = r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        // Test that we can get the function and the table name is correct
        let function_config = get_function(&config.functions, "test_function").unwrap();
        assert_eq!(function_config.table_name(), "ChatInference");
    }

    #[tokio::test]
    async fn test_get_function_determines_correct_inference_table_for_json() {
        let config_str = r#"
            [functions.test_function]
            type = "json"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"
            json_mode = "implicit_tool"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        // Test that we can get the function and the table name is correct
        let function_config = get_function(&config.functions, "test_function").unwrap();
        assert_eq!(function_config.table_name(), "JsonInference");
    }

    // =================================================================
    // Tests for get_variant_performances
    // =================================================================

    use crate::config::{MetricConfigLevel, MetricConfigType};
    use crate::db::feedback::MockFeedbackQueries;

    fn create_config_with_function_and_metric() -> String {
        r#"
            [functions.test_function]
            type = "chat"

            [functions.test_function.variants.variant_a]
            type = "chat_completion"
            model = "openai::gpt-4"

            [functions.test_function.variants.variant_b]
            type = "chat_completion"
            model = "openai::gpt-4o-mini"

            [metrics.accuracy]
            type = "float"
            optimize = "max"
            level = "inference"
        "#
        .to_string()
    }

    #[tokio::test]
    async fn test_get_variant_performances_function_not_found() {
        let config = Config::default();
        let mut mock_clickhouse = MockFeedbackQueries::new();

        // Should not call clickhouse because function validation fails first
        mock_clickhouse.expect_get_variant_performances().never();

        let result = get_variant_performances(
            &config,
            &mock_clickhouse,
            "nonexistent_function",
            "accuracy",
            TimeWindow::Cumulative,
            None,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent_function"));
    }

    #[tokio::test]
    async fn test_get_variant_performances_metric_not_found() {
        let config_str = create_config_with_function_and_metric();
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        let mut mock_clickhouse = MockFeedbackQueries::new();
        mock_clickhouse.expect_get_variant_performances().never();

        let result = get_variant_performances(
            &config,
            &mock_clickhouse,
            "test_function",
            "nonexistent_metric",
            TimeWindow::Cumulative,
            None,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent_metric"),
            "Error should contain metric name: {err}"
        );
    }

    #[tokio::test]
    async fn test_get_variant_performances_variant_not_found() {
        let config_str = create_config_with_function_and_metric();
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        let mut mock_clickhouse = MockFeedbackQueries::new();
        mock_clickhouse.expect_get_variant_performances().never();

        let result = get_variant_performances(
            &config,
            &mock_clickhouse,
            "test_function",
            "accuracy",
            TimeWindow::Cumulative,
            Some("nonexistent_variant"),
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent_variant"),
            "Error should contain variant name: {err}"
        );
    }

    #[tokio::test]
    async fn test_get_variant_performances_calls_clickhouse() {
        use crate::function::FunctionConfigType;
        use chrono::Utc;

        let config_str = create_config_with_function_and_metric();
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        let mut mock_clickhouse = MockFeedbackQueries::new();
        mock_clickhouse
            .expect_get_variant_performances()
            .withf(|params| {
                assert_eq!(params.function_name, "test_function");
                assert_eq!(params.function_type, FunctionConfigType::Chat);
                assert_eq!(params.metric_name, "accuracy");
                assert_eq!(params.metric_config.r#type, MetricConfigType::Float);
                assert_eq!(params.metric_config.level, MetricConfigLevel::Inference);
                assert_eq!(params.time_window, TimeWindow::Cumulative);
                assert!(params.variant_name.is_none());
                true
            })
            .times(1)
            .returning(|_| {
                Box::pin(async move {
                    Ok(vec![
                        VariantPerformanceRow {
                            period_start: Utc::now(),
                            variant_name: "variant_a".to_string(),
                            count: 10,
                            avg_metric: 0.85,
                            stdev: Some(0.05),
                            ci_error: Some(0.03),
                        },
                        VariantPerformanceRow {
                            period_start: Utc::now(),
                            variant_name: "variant_b".to_string(),
                            count: 15,
                            avg_metric: 0.90,
                            stdev: Some(0.03),
                            ci_error: Some(0.02),
                        },
                    ])
                })
            });

        let result = get_variant_performances(
            &config,
            &mock_clickhouse,
            "test_function",
            "accuracy",
            TimeWindow::Cumulative,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.performances.len(), 2);
        assert_eq!(result.performances[0].variant_name, "variant_a");
        assert_eq!(result.performances[1].variant_name, "variant_b");
    }

    #[tokio::test]
    async fn test_get_variant_performances_with_variant_filter() {
        use chrono::Utc;

        let config_str = create_config_with_function_and_metric();
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(config_str.as_bytes()).unwrap();

        let config = Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests();

        let mut mock_clickhouse = MockFeedbackQueries::new();
        mock_clickhouse
            .expect_get_variant_performances()
            .withf(|params| {
                assert_eq!(params.variant_name, Some("variant_a"));
                true
            })
            .times(1)
            .returning(|_| {
                Box::pin(async move {
                    Ok(vec![VariantPerformanceRow {
                        period_start: Utc::now(),
                        variant_name: "variant_a".to_string(),
                        count: 10,
                        avg_metric: 0.85,
                        stdev: Some(0.05),
                        ci_error: Some(0.03),
                    }])
                })
            });

        let result = get_variant_performances(
            &config,
            &mock_clickhouse,
            "test_function",
            "accuracy",
            TimeWindow::Week,
            Some("variant_a"),
        )
        .await
        .unwrap();

        assert_eq!(result.performances.len(), 1);
        assert_eq!(result.performances[0].variant_name, "variant_a");
    }
}
