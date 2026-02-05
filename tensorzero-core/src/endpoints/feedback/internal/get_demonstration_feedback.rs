//! Endpoint for querying demonstration feedback by inference ID

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::{DemonstrationFeedbackRow, FeedbackQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[derive(Debug, Deserialize)]
pub struct GetDemonstrationFeedbackParams {
    pub before: Option<Uuid>,
    pub after: Option<Uuid>,
    pub limit: Option<u32>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetDemonstrationFeedbackResponse {
    pub feedback: Vec<DemonstrationFeedbackRow>,
}

/// HTTP handler for getting demonstration feedback by inference ID
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_demonstration_feedback_handler",
    skip_all,
    fields(
        inference_id = %inference_id,
    )
)]
pub async fn get_demonstration_feedback_handler(
    State(app_state): AppState,
    Path(inference_id): Path<Uuid>,
    Query(params): Query<GetDemonstrationFeedbackParams>,
) -> Result<Json<GetDemonstrationFeedbackResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let response = get_demonstration_feedback(
        &database,
        inference_id,
        params.before,
        params.after,
        params.limit,
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting demonstration feedback by inference ID
pub async fn get_demonstration_feedback(
    database: &(dyn FeedbackQueries + Sync),
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: Option<u32>,
) -> Result<GetDemonstrationFeedbackResponse, Error> {
    let feedback = database
        .query_demonstration_feedback_by_inference_id(inference_id, before, after, limit)
        .await?;

    Ok(GetDemonstrationFeedbackResponse { feedback })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::feedback::MockFeedbackQueries;
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_get_demonstration_feedback_calls_clickhouse() {
        let mut mock_db = MockFeedbackQueries::new();

        let inference_id = Uuid::now_v7();
        let feedback_id = Uuid::now_v7();

        mock_db
            .expect_query_demonstration_feedback_by_inference_id()
            .withf(move |id, before, after, limit| {
                *id == inference_id && before.is_none() && after.is_none() && *limit == Some(10)
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let rows = vec![DemonstrationFeedbackRow {
                    id: feedback_id,
                    inference_id,
                    value: "demonstration value".to_string(),
                    tags: HashMap::new(),
                    timestamp: Utc::now(),
                }];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_demonstration_feedback(&mock_db, inference_id, None, None, Some(10))
            .await
            .unwrap();

        assert_eq!(result.feedback.len(), 1);
        assert_eq!(result.feedback[0].value, "demonstration value");
    }

    #[tokio::test]
    async fn test_get_demonstration_feedback_with_before_pagination() {
        let mut mock_db = MockFeedbackQueries::new();

        let inference_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();

        mock_db
            .expect_query_demonstration_feedback_by_inference_id()
            .withf(move |id, before, after, limit| {
                *id == inference_id
                    && *before == Some(before_id)
                    && after.is_none()
                    && *limit == Some(50)
            })
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result =
            get_demonstration_feedback(&mock_db, inference_id, Some(before_id), None, Some(50))
                .await
                .unwrap();

        assert!(result.feedback.is_empty());
    }

    #[tokio::test]
    async fn test_get_demonstration_feedback_with_after_pagination() {
        let mut mock_db = MockFeedbackQueries::new();

        let inference_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();

        mock_db
            .expect_query_demonstration_feedback_by_inference_id()
            .withf(move |id, before, after, limit| {
                *id == inference_id
                    && before.is_none()
                    && *after == Some(after_id)
                    && *limit == Some(25)
            })
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result =
            get_demonstration_feedback(&mock_db, inference_id, None, Some(after_id), Some(25))
                .await
                .unwrap();

        assert!(result.feedback.is_empty());
    }

    #[tokio::test]
    async fn test_get_demonstration_feedback_empty_result() {
        let mut mock_db = MockFeedbackQueries::new();

        let inference_id = Uuid::now_v7();

        mock_db
            .expect_query_demonstration_feedback_by_inference_id()
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_demonstration_feedback(&mock_db, inference_id, None, None, None)
            .await
            .unwrap();

        assert!(result.feedback.is_empty());
    }

    #[tokio::test]
    async fn test_get_demonstration_feedback_multiple_results() {
        let mut mock_db = MockFeedbackQueries::new();

        let inference_id = Uuid::now_v7();

        mock_db
            .expect_query_demonstration_feedback_by_inference_id()
            .times(1)
            .returning(move |_, _, _, _| {
                let rows = vec![
                    DemonstrationFeedbackRow {
                        id: Uuid::now_v7(),
                        inference_id,
                        value: "demo 1".to_string(),
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    },
                    DemonstrationFeedbackRow {
                        id: Uuid::now_v7(),
                        inference_id,
                        value: "demo 2".to_string(),
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    },
                ];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_demonstration_feedback(&mock_db, inference_id, None, None, None)
            .await
            .unwrap();

        assert_eq!(result.feedback.len(), 2);
    }
}
