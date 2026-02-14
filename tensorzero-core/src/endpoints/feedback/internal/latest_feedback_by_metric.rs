//! Feedback endpoint for querying feedback-related information

use std::collections::HashMap;

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::FeedbackQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct LatestFeedbackIdByMetricResponse {
    pub feedback_id_by_metric: HashMap<String, String>,
}

/// HTTP handler for getting the latest feedback ID for each metric for a target
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_latest_feedback_id_by_metric_handler",
    skip_all,
    fields(
        target_id = %target_id,
    )
)]
pub async fn get_latest_feedback_id_by_metric_handler(
    State(app_state): AppState,
    Path(target_id): Path<Uuid>,
) -> Result<Json<LatestFeedbackIdByMetricResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let response = get_latest_feedback_id_by_metric(&database, target_id).await?;
    Ok(Json(response))
}

/// Core business logic for getting the latest feedback ID for each metric
pub async fn get_latest_feedback_id_by_metric(
    database: &(dyn FeedbackQueries + Sync),
    target_id: Uuid,
) -> Result<LatestFeedbackIdByMetricResponse, Error> {
    let latest_feedback_rows = database
        .query_latest_feedback_id_by_metric(target_id)
        .await?;

    let feedback_id_by_metric = latest_feedback_rows
        .into_iter()
        .map(|row| (row.metric_name, row.latest_id))
        .collect();

    Ok(LatestFeedbackIdByMetricResponse {
        feedback_id_by_metric,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::feedback::{LatestFeedbackRow, MockFeedbackQueries};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_get_latest_feedback_id_by_metric_calls_clickhouse() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();
        let accuracy_id = Uuid::now_v7().to_string();
        let quality_id = Uuid::now_v7().to_string();

        let mut expected_map = HashMap::new();
        expected_map.insert("accuracy".to_string(), accuracy_id.clone());
        expected_map.insert("quality".to_string(), quality_id.clone());

        mock_db
            .expect_query_latest_feedback_id_by_metric()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning({
                let accuracy_id = accuracy_id.clone();
                let quality_id = quality_id.clone();
                move |_| {
                    let rows = vec![
                        LatestFeedbackRow {
                            metric_name: "accuracy".to_string(),
                            latest_id: accuracy_id.clone(),
                        },
                        LatestFeedbackRow {
                            metric_name: "quality".to_string(),
                            latest_id: quality_id.clone(),
                        },
                    ];
                    Box::pin(async move { Ok(rows) })
                }
            });

        let result = get_latest_feedback_id_by_metric(&mock_db, target_id)
            .await
            .unwrap();

        assert_eq!(result.feedback_id_by_metric, expected_map);
    }

    #[tokio::test]
    async fn test_get_latest_feedback_id_by_metric_empty_result() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();
        let expected_map = HashMap::new();

        mock_db
            .expect_query_latest_feedback_id_by_metric()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning(move |_| Box::pin(async move { Ok(vec![]) }));

        let result = get_latest_feedback_id_by_metric(&mock_db, target_id)
            .await
            .unwrap();

        assert_eq!(result.feedback_id_by_metric, expected_map);
        assert!(result.feedback_id_by_metric.is_empty());
    }
}
