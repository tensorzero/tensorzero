//! Feedback endpoint for querying cumulative feedback time series

use axum::extract::{Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::TimeWindow;
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::{CumulativeFeedbackTimeSeriesPoint, FeedbackQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[derive(Debug, Deserialize)]
pub struct GetCumulativeFeedbackTimeseriesParams {
    pub function_name: String,
    pub metric_name: String,
    /// Comma-separated list of variant names to filter by. If not provided, all variants are included.
    pub variant_names: Option<String>,
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetCumulativeFeedbackTimeseriesResponse {
    pub timeseries: Vec<CumulativeFeedbackTimeSeriesPoint>,
}

/// HTTP handler for getting cumulative feedback time series
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_cumulative_feedback_timeseries_handler",
    skip_all,
    fields(
        function_name = %params.function_name,
        metric_name = %params.metric_name,
        time_window = ?params.time_window,
        max_periods = %params.max_periods,
    )
)]
pub async fn get_cumulative_feedback_timeseries_handler(
    State(app_state): AppState,
    Query(params): Query<GetCumulativeFeedbackTimeseriesParams>,
) -> Result<Json<GetCumulativeFeedbackTimeseriesResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );

    let variant_names = params.variant_names.map(|s| {
        s.split(',')
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>()
    });

    let response = get_cumulative_feedback_timeseries(
        &database,
        params.function_name,
        params.metric_name,
        variant_names,
        params.time_window,
        params.max_periods,
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting cumulative feedback time series
pub async fn get_cumulative_feedback_timeseries(
    database: &(dyn FeedbackQueries + Sync),
    function_name: String,
    metric_name: String,
    variant_names: Option<Vec<String>>,
    time_window: TimeWindow,
    max_periods: u32,
) -> Result<GetCumulativeFeedbackTimeseriesResponse, Error> {
    let timeseries = database
        .get_cumulative_feedback_timeseries(
            function_name,
            metric_name,
            variant_names,
            time_window,
            max_periods,
        )
        .await?;

    Ok(GetCumulativeFeedbackTimeseriesResponse { timeseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::feedback::MockFeedbackQueries;
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn test_get_cumulative_feedback_timeseries_calls_clickhouse() {
        let mut mock_db = MockFeedbackQueries::new();

        mock_db
            .expect_get_cumulative_feedback_timeseries()
            .withf(|function, metric, variants, window, periods| {
                function == "my_function"
                    && metric == "accuracy"
                    && *variants == Some(vec!["variant_a".to_string()])
                    && matches!(window, TimeWindow::Hour)
                    && *periods == 24
            })
            .times(1)
            .returning(|_, _, _, _, _| {
                let rows = vec![
                    CumulativeFeedbackTimeSeriesPoint {
                        period_end: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
                        variant_name: "variant_a".to_string(),
                        mean: 0.80,
                        variance: Some(0.01),
                        count: 50,
                        alpha: 0.05,
                        cs_lower: Some(0.75),
                        cs_upper: Some(0.85),
                    },
                    CumulativeFeedbackTimeSeriesPoint {
                        period_end: Utc.with_ymd_and_hms(2024, 1, 1, 13, 0, 0).unwrap(),
                        variant_name: "variant_a".to_string(),
                        mean: 0.82,
                        variance: Some(0.01),
                        count: 100,
                        alpha: 0.05,
                        cs_lower: Some(0.78),
                        cs_upper: Some(0.86),
                    },
                ];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_cumulative_feedback_timeseries(
            &mock_db,
            "my_function".to_string(),
            "accuracy".to_string(),
            Some(vec!["variant_a".to_string()]),
            TimeWindow::Hour,
            24,
        )
        .await
        .unwrap();

        assert_eq!(result.timeseries.len(), 2);
        assert_eq!(result.timeseries[0].variant_name, "variant_a");
        assert_eq!(result.timeseries[0].mean, 0.80);
        assert_eq!(result.timeseries[1].count, 100);
    }

    #[tokio::test]
    async fn test_get_cumulative_feedback_timeseries_without_variant_filter() {
        let mut mock_db = MockFeedbackQueries::new();

        mock_db
            .expect_get_cumulative_feedback_timeseries()
            .withf(|function, metric, variants, window, periods| {
                function == "test_function"
                    && metric == "task_success"
                    && variants.is_none()
                    && matches!(window, TimeWindow::Day)
                    && *periods == 7
            })
            .times(1)
            .returning(|_, _, _, _, _| {
                let rows = vec![CumulativeFeedbackTimeSeriesPoint {
                    period_end: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                    variant_name: "default".to_string(),
                    mean: 0.90,
                    variance: Some(0.005),
                    count: 200,
                    alpha: 0.05,
                    cs_lower: Some(0.87),
                    cs_upper: Some(0.93),
                }];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_cumulative_feedback_timeseries(
            &mock_db,
            "test_function".to_string(),
            "task_success".to_string(),
            None,
            TimeWindow::Day,
            7,
        )
        .await
        .unwrap();

        assert_eq!(result.timeseries.len(), 1);
        assert_eq!(result.timeseries[0].mean, 0.90);
        assert_eq!(result.timeseries[0].alpha, 0.05);
    }

    #[tokio::test]
    async fn test_get_cumulative_feedback_timeseries_empty_result() {
        let mut mock_db = MockFeedbackQueries::new();

        mock_db
            .expect_get_cumulative_feedback_timeseries()
            .times(1)
            .returning(|_, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_cumulative_feedback_timeseries(
            &mock_db,
            "my_function".to_string(),
            "nonexistent_metric".to_string(),
            None,
            TimeWindow::Week,
            4,
        )
        .await
        .unwrap();

        assert!(result.timeseries.is_empty());
    }

    #[tokio::test]
    async fn test_get_cumulative_feedback_timeseries_cumulative_window_returns_error() {
        use crate::error::{Error, ErrorDetails};

        let mut mock_db = MockFeedbackQueries::new();

        // The ClickHouse implementation returns an error for Cumulative time window
        mock_db
            .expect_get_cumulative_feedback_timeseries()
            .withf(|_, _, _, window, _| matches!(window, TimeWindow::Cumulative))
            .times(1)
            .returning(|_, _, _, _, _| {
                Box::pin(async move {
                    Err(Error::new(ErrorDetails::InvalidRequest {
                        message: "Cumulative time window is not supported for feedback timeseries"
                            .to_string(),
                    }))
                })
            });

        let result = get_cumulative_feedback_timeseries(
            &mock_db,
            "my_function".to_string(),
            "quality".to_string(),
            None,
            TimeWindow::Cumulative,
            1,
        )
        .await;

        assert!(
            result.is_err(),
            "Cumulative time window should return an error"
        );
    }
}
