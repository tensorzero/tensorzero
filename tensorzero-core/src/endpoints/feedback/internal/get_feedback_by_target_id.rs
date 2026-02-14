//! Feedback endpoint for querying feedback by target ID

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::{FeedbackQueries, FeedbackRow};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[derive(Debug, Deserialize)]
pub struct GetFeedbackByTargetIdParams {
    pub before: Option<Uuid>,
    pub after: Option<Uuid>,
    pub limit: Option<u32>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetFeedbackByTargetIdResponse {
    pub feedback: Vec<FeedbackRow>,
}

/// HTTP handler for getting feedback by target ID
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_feedback_by_target_id_handler",
    skip_all,
    fields(
        target_id = %target_id,
    )
)]
pub async fn get_feedback_by_target_id_handler(
    State(app_state): AppState,
    Path(target_id): Path<Uuid>,
    Query(params): Query<GetFeedbackByTargetIdParams>,
) -> Result<Json<GetFeedbackByTargetIdResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let response = get_feedback_by_target_id(
        &database,
        target_id,
        params.before,
        params.after,
        params.limit,
    )
    .await?;
    Ok(Json(response))
}

/// Core business logic for getting feedback by target ID
pub async fn get_feedback_by_target_id(
    database: &(dyn FeedbackQueries + Sync),
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: Option<u32>,
) -> Result<GetFeedbackByTargetIdResponse, Error> {
    let feedback = database
        .query_feedback_by_target_id(target_id, before, after, limit)
        .await?;

    Ok(GetFeedbackByTargetIdResponse { feedback })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::feedback::{
        BooleanMetricFeedbackRow, CommentFeedbackRow, CommentTargetType, DemonstrationFeedbackRow,
        FloatMetricFeedbackRow, MockFeedbackQueries,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_get_feedback_by_target_id_calls_clickhouse() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();
        let feedback_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_by_target_id()
            .withf(move |id, before, after, limit| {
                *id == target_id && before.is_none() && after.is_none() && *limit == Some(10)
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let rows = vec![FeedbackRow::Boolean(BooleanMetricFeedbackRow {
                    id: feedback_id,
                    target_id,
                    metric_name: "accuracy".to_string(),
                    value: true,
                    tags: HashMap::new(),
                    timestamp: Utc::now(),
                })];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_feedback_by_target_id(&mock_db, target_id, None, None, Some(10))
            .await
            .unwrap();

        assert_eq!(result.feedback.len(), 1);
        match &result.feedback[0] {
            FeedbackRow::Boolean(row) => {
                assert_eq!(row.metric_name, "accuracy");
                assert!(row.value);
            }
            _ => panic!("Expected Boolean feedback row"),
        }
    }

    #[tokio::test]
    async fn test_get_feedback_by_target_id_with_before_pagination() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();
        let before_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_by_target_id()
            .withf(move |id, before, after, limit| {
                *id == target_id
                    && *before == Some(before_id)
                    && after.is_none()
                    && *limit == Some(50)
            })
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result =
            get_feedback_by_target_id(&mock_db, target_id, Some(before_id), None, Some(50))
                .await
                .unwrap();

        assert!(result.feedback.is_empty());
    }

    #[tokio::test]
    async fn test_get_feedback_by_target_id_with_after_pagination() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_by_target_id()
            .withf(move |id, before, after, limit| {
                *id == target_id
                    && before.is_none()
                    && *after == Some(after_id)
                    && *limit == Some(25)
            })
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_feedback_by_target_id(&mock_db, target_id, None, Some(after_id), Some(25))
            .await
            .unwrap();

        assert!(result.feedback.is_empty());
    }

    #[tokio::test]
    async fn test_get_feedback_by_target_id_returns_multiple_types() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_by_target_id()
            .times(1)
            .returning(move |_, _, _, _| {
                let rows = vec![
                    FeedbackRow::Boolean(BooleanMetricFeedbackRow {
                        id: Uuid::now_v7(),
                        target_id,
                        metric_name: "accuracy".to_string(),
                        value: true,
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    }),
                    FeedbackRow::Float(FloatMetricFeedbackRow {
                        id: Uuid::now_v7(),
                        target_id,
                        metric_name: "score".to_string(),
                        value: 0.95,
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    }),
                    FeedbackRow::Comment(CommentFeedbackRow {
                        id: Uuid::now_v7(),
                        target_id,
                        target_type: CommentTargetType::Inference,
                        value: "Great response!".to_string(),
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    }),
                    FeedbackRow::Demonstration(DemonstrationFeedbackRow {
                        id: Uuid::now_v7(),
                        inference_id: target_id,
                        value: "demonstration value".to_string(),
                        tags: HashMap::new(),
                        timestamp: Utc::now(),
                    }),
                ];
                Box::pin(async move { Ok(rows) })
            });

        let result = get_feedback_by_target_id(&mock_db, target_id, None, None, None)
            .await
            .unwrap();

        assert_eq!(result.feedback.len(), 4);

        // Check that we have all 4 types
        let has_boolean = result
            .feedback
            .iter()
            .any(|f| matches!(f, FeedbackRow::Boolean(_)));
        let has_float = result
            .feedback
            .iter()
            .any(|f| matches!(f, FeedbackRow::Float(_)));
        let has_comment = result
            .feedback
            .iter()
            .any(|f| matches!(f, FeedbackRow::Comment(_)));
        let has_demonstration = result
            .feedback
            .iter()
            .any(|f| matches!(f, FeedbackRow::Demonstration(_)));

        assert!(has_boolean, "Expected Boolean feedback");
        assert!(has_float, "Expected Float feedback");
        assert!(has_comment, "Expected Comment feedback");
        assert!(has_demonstration, "Expected Demonstration feedback");
    }

    #[tokio::test]
    async fn test_get_feedback_by_target_id_empty_result() {
        let mut mock_db = MockFeedbackQueries::new();

        let target_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_by_target_id()
            .times(1)
            .returning(move |_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_feedback_by_target_id(&mock_db, target_id, None, None, None)
            .await
            .unwrap();

        assert!(result.feedback.is_empty());
    }
}
