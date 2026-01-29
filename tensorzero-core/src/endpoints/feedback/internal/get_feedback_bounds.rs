//! Feedback endpoint for querying feedback bounds by target ID

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::feedback::{FeedbackBounds, FeedbackBoundsByType, FeedbackQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetFeedbackBoundsResponse {
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub first_id: Option<Uuid>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub last_id: Option<Uuid>,
    pub by_type: FeedbackBoundsByType,
}

impl From<FeedbackBounds> for GetFeedbackBoundsResponse {
    fn from(bounds: FeedbackBounds) -> Self {
        Self {
            first_id: bounds.first_id,
            last_id: bounds.last_id,
            by_type: bounds.by_type,
        }
    }
}

/// HTTP handler for getting feedback bounds by target ID
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_feedback_bounds_by_target_id_handler",
    skip_all,
    fields(
        target_id = %target_id,
    )
)]
pub async fn get_feedback_bounds_by_target_id_handler(
    State(app_state): AppState,
    Path(target_id): Path<Uuid>,
) -> Result<Json<GetFeedbackBoundsResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let response = get_feedback_bounds_by_target_id(&database, target_id).await?;
    Ok(Json(response))
}

/// Core business logic for getting feedback bounds by target ID
pub async fn get_feedback_bounds_by_target_id(
    database: &(dyn FeedbackQueries + Sync),
    target_id: Uuid,
) -> Result<GetFeedbackBoundsResponse, Error> {
    let bounds = database
        .query_feedback_bounds_by_target_id(target_id)
        .await?;

    Ok(bounds.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TableBounds;
    use crate::db::feedback::MockFeedbackQueries;

    #[tokio::test]
    async fn test_get_feedback_bounds_by_target_id_calls_clickhouse() {
        let mut mock_db = MockFeedbackQueries::new();
        let target_id = Uuid::now_v7();
        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_bounds_by_target_id()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning(move |_| {
                Box::pin(async move {
                    Ok(FeedbackBounds {
                        first_id: Some(first_id),
                        last_id: Some(last_id),
                        by_type: FeedbackBoundsByType {
                            boolean: TableBounds {
                                first_id: Some(first_id),
                                last_id: Some(last_id),
                            },
                            ..Default::default()
                        },
                    })
                })
            });

        let response = get_feedback_bounds_by_target_id(&mock_db, target_id)
            .await
            .expect("Expected feedback bounds to be returned");

        assert_eq!(
            response.first_id,
            Some(first_id),
            "Expected first_id to propagate from ClickHouse bounds"
        );
        assert_eq!(
            response.last_id,
            Some(last_id),
            "Expected last_id to propagate from ClickHouse bounds"
        );
        assert_eq!(
            response.by_type.boolean.first_id,
            Some(first_id),
            "Expected boolean bounds to be included"
        );
    }

    #[tokio::test]
    async fn test_get_feedback_bounds_by_target_id_handles_empty_bounds() {
        let mut mock_db = MockFeedbackQueries::new();
        let target_id = Uuid::now_v7();

        mock_db
            .expect_query_feedback_bounds_by_target_id()
            .withf(move |id| *id == target_id)
            .times(1)
            .returning(|_| {
                Box::pin(async move {
                    Ok(FeedbackBounds {
                        first_id: None,
                        last_id: None,
                        by_type: FeedbackBoundsByType {
                            demonstration: TableBounds {
                                first_id: None,
                                last_id: None,
                            },
                            ..Default::default()
                        },
                    })
                })
            });

        let response = get_feedback_bounds_by_target_id(&mock_db, target_id)
            .await
            .expect("Expected empty bounds response");

        assert!(
            response.first_id.is_none() && response.last_id.is_none(),
            "Expected first_id and last_id to be None when no feedback exists"
        );
        assert!(
            response.by_type.boolean.first_id.is_none(),
            "Expected boolean bounds to be empty"
        );
    }
}
