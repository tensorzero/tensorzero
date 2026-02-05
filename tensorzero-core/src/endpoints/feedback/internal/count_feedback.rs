//! Feedback endpoint for counting feedback by target ID

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
pub struct CountFeedbackByTargetIdResponse {
    pub count: u64,
}

/// HTTP handler for counting feedback by target ID
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "count_feedback_by_target_id_handler",
    skip_all,
    fields(
        target_id = %target_id,
    )
)]
pub async fn count_feedback_by_target_id_handler(
    State(app_state): AppState,
    Path(target_id): Path<Uuid>,
) -> Result<Json<CountFeedbackByTargetIdResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let response = count_feedback_by_target_id(&database, target_id).await?;
    Ok(Json(response))
}

/// Core business logic for counting feedback by target ID
pub async fn count_feedback_by_target_id(
    database: &(dyn FeedbackQueries + Sync),
    target_id: Uuid,
) -> Result<CountFeedbackByTargetIdResponse, Error> {
    let count = database.count_feedback_by_target_id(target_id).await?;

    Ok(CountFeedbackByTargetIdResponse { count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::feedback::MockFeedbackQueries;

    #[tokio::test]
    async fn test_count_feedback_by_target_id_returns_count() {
        let mut mock_db = MockFeedbackQueries::new();
        let target_id = Uuid::now_v7();

        mock_db
            .expect_count_feedback_by_target_id()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning(|_| Box::pin(async move { Ok(42) }));

        let response = count_feedback_by_target_id(&mock_db, target_id)
            .await
            .expect("Expected count to be returned");

        assert_eq!(
            response.count, 42,
            "Expected count to be 42 from ClickHouse"
        );
    }

    #[tokio::test]
    async fn test_count_feedback_by_target_id_returns_zero_for_no_feedback() {
        let mut mock_db = MockFeedbackQueries::new();
        let target_id = Uuid::now_v7();

        mock_db
            .expect_count_feedback_by_target_id()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning(|_| Box::pin(async move { Ok(0) }));

        let response = count_feedback_by_target_id(&mock_db, target_id)
            .await
            .expect("Expected zero count response");

        assert_eq!(
            response.count, 0,
            "Expected count to be 0 when no feedback exists"
        );
    }
}
