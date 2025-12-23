//! Handler for getting workflow evaluation run statistics by metric name.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::{GetWorkflowEvaluationRunStatisticsResponse, WorkflowEvaluationRunStatistics};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting workflow evaluation run statistics.
#[derive(Debug, Deserialize)]
pub struct GetWorkflowEvaluationRunStatisticsParams {
    /// The run ID to get statistics for
    pub run_id: String,
    /// Optional metric name to filter by
    pub metric_name: Option<String>,
}

/// Handler for `GET /internal/workflow_evaluations/run_statistics`
///
/// Gets aggregated statistics (count, mean, stdev, confidence intervals) for a workflow
/// evaluation run, grouped by metric name.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.get_run_statistics", skip_all)]
pub async fn get_workflow_evaluation_run_statistics_handler(
    State(app_state): AppState,
    Query(params): Query<GetWorkflowEvaluationRunStatisticsParams>,
) -> Result<Json<GetWorkflowEvaluationRunStatisticsResponse>, Error> {
    // Parse the run_id UUID
    let run_id = Uuid::parse_str(&params.run_id).map_err(|e| {
        Error::new(crate::error::ErrorDetails::InvalidRequest {
            message: format!("Invalid UUID '{}': {e}", params.run_id),
        })
    })?;

    let response = get_workflow_evaluation_run_statistics(
        &app_state.clickhouse_connection_info,
        run_id,
        params.metric_name.as_deref(),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting workflow evaluation run statistics
pub async fn get_workflow_evaluation_run_statistics(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_id: Uuid,
    metric_name: Option<&str>,
) -> Result<GetWorkflowEvaluationRunStatisticsResponse, Error> {
    let stats_database = clickhouse
        .get_workflow_evaluation_run_statistics(run_id, metric_name)
        .await?;
    let statistics = stats_database
        .into_iter()
        .map(|stat| WorkflowEvaluationRunStatistics {
            metric_name: stat.metric_name,
            count: stat.count,
            avg_metric: stat.avg_metric,
            stdev: stat.stdev,
            ci_lower: stat.ci_lower,
            ci_upper: stat.ci_upper,
        })
        .collect();
    Ok(GetWorkflowEvaluationRunStatisticsResponse { statistics })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::{
        MockWorkflowEvaluationQueries, WorkflowEvaluationRunStatisticsRow,
    };

    fn create_test_statistics() -> Vec<WorkflowEvaluationRunStatisticsRow> {
        vec![
            WorkflowEvaluationRunStatisticsRow {
                metric_name: "elapsed_ms".to_string(),
                count: 49,
                avg_metric: 91678.72114158163,
                stdev: Some(21054.80078125),
                ci_lower: Some(85783.37692283162),
                ci_upper: Some(97574.06536033163),
            },
            WorkflowEvaluationRunStatisticsRow {
                metric_name: "goated".to_string(),
                count: 1,
                avg_metric: 1.0,
                stdev: None,
                ci_lower: Some(0.20654329147389294),
                ci_upper: Some(1.0),
            },
            WorkflowEvaluationRunStatisticsRow {
                metric_name: "solved".to_string(),
                count: 49,
                avg_metric: 0.4489795918367347,
                stdev: Some(0.5025445456953674),
                ci_lower: Some(0.31852624929636336),
                ci_upper: Some(0.5868513320032188),
            },
        ]
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_statistics_basic() {
        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_statistics()
            .withf(move |id, metric_name| {
                assert_eq!(*id, run_id);
                assert!(metric_name.is_none());
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(create_test_statistics()) }));

        let result = get_workflow_evaluation_run_statistics(&mock_clickhouse, run_id, None)
            .await
            .unwrap();

        assert_eq!(result.statistics.len(), 3);

        // Check elapsed_ms
        let elapsed_ms = result
            .statistics
            .iter()
            .find(|s| s.metric_name == "elapsed_ms")
            .unwrap();
        assert_eq!(elapsed_ms.count, 49);
        assert!((elapsed_ms.avg_metric - 91678.72114158163).abs() < 0.001);

        // Check goated
        let goated = result
            .statistics
            .iter()
            .find(|s| s.metric_name == "goated")
            .unwrap();
        assert_eq!(goated.count, 1);
        assert!(goated.stdev.is_none());

        // Check solved
        let solved = result
            .statistics
            .iter()
            .find(|s| s.metric_name == "solved")
            .unwrap();
        assert_eq!(solved.count, 49);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_statistics_with_metric_filter() {
        let run_id = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_statistics()
            .withf(move |id, metric_name| {
                assert_eq!(*id, run_id);
                assert_eq!(*metric_name, Some("solved"));
                true
            })
            .times(1)
            .returning(|_, _| {
                Box::pin(async move {
                    Ok(vec![WorkflowEvaluationRunStatisticsRow {
                        metric_name: "solved".to_string(),
                        count: 49,
                        avg_metric: 0.4489795918367347,
                        stdev: Some(0.5025445456953674),
                        ci_lower: Some(0.31852624929636336),
                        ci_upper: Some(0.5868513320032188),
                    }])
                })
            });

        let result =
            get_workflow_evaluation_run_statistics(&mock_clickhouse, run_id, Some("solved"))
                .await
                .unwrap();

        assert_eq!(result.statistics.len(), 1);
        assert_eq!(result.statistics[0].metric_name, "solved");
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_statistics_empty() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_statistics()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_workflow_evaluation_run_statistics(&mock_clickhouse, run_id, None)
            .await
            .unwrap();

        assert_eq!(result.statistics.len(), 0);
    }
}
