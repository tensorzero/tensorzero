//! Episodes endpoint for querying episode data.

use axum::extract::{Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::{EpisodeByIdRow, SelectQueries, TableBoundsWithCount};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the episode table endpoint
#[derive(Debug, Deserialize)]
pub struct ListEpisodesParams {
    /// Maximum number of episodes to return
    pub limit: u32,
    /// Return episodes before this episode_id (for pagination)
    pub before: Option<Uuid>,
    /// Return episodes after this episode_id (for pagination)
    pub after: Option<Uuid>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListEpisodesResponse {
    pub episodes: Vec<EpisodeByIdRow>,
}

/// HTTP handler for querying episodes
#[debug_handler(state = AppStateData)]
#[instrument(name = "list_episodes_handler", skip_all)]
pub async fn list_episodes_handler(
    State(app_state): AppState,
    Query(params): Query<ListEpisodesParams>,
) -> Result<Json<ListEpisodesResponse>, Error> {
    let episodes = list_episodes(&app_state.clickhouse_connection_info, params).await?;
    Ok(Json(ListEpisodesResponse { episodes }))
}

/// Core business logic for listing episodes
pub async fn list_episodes(
    clickhouse: &impl SelectQueries,
    params: ListEpisodesParams,
) -> Result<Vec<EpisodeByIdRow>, Error> {
    if params.limit > 100 {
        return Err(ErrorDetails::InvalidRequest {
            message: "Limit cannot exceed 100".to_string(),
        }
        .into());
    }
    clickhouse
        .query_episode_table(params.limit, params.before, params.after)
        .await
}

/// HTTP handler for querying episode table bounds
#[debug_handler(state = AppStateData)]
#[instrument(name = "query_episode_table_bounds_handler", skip_all)]
pub async fn query_episode_table_bounds_handler(
    State(app_state): AppState,
) -> Result<Json<TableBoundsWithCount>, Error> {
    let bounds = query_episode_table_bounds(&app_state.clickhouse_connection_info).await?;
    Ok(Json(bounds))
}

/// Core business logic for querying episode table bounds
pub async fn query_episode_table_bounds(
    clickhouse: &impl SelectQueries,
) -> Result<TableBoundsWithCount, Error> {
    clickhouse.query_episode_table_bounds().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MockSelectQueries;
    use chrono::Utc;

    #[tokio::test]
    async fn test_query_episode_table_calls_clickhouse() {
        let mut mock_clickhouse = MockSelectQueries::new();

        let episode_id = Uuid::now_v7();
        let last_inference_id = Uuid::now_v7();
        let now = Utc::now();

        mock_clickhouse
            .expect_query_episode_table()
            .withf(|limit, before, after| {
                assert_eq!(*limit, 10);
                assert!(before.is_none());
                assert!(after.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(vec![EpisodeByIdRow {
                        episode_id,
                        count: 5,
                        start_time: now,
                        end_time: now,
                        last_inference_id,
                    }])
                })
            });

        let params = ListEpisodesParams {
            limit: 10,
            before: None,
            after: None,
        };

        let result = list_episodes(&mock_clickhouse, params).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].episode_id, episode_id);
        assert_eq!(result[0].count, 5);
        assert_eq!(result[0].last_inference_id, last_inference_id);
    }

    #[tokio::test]
    async fn test_query_episode_table_with_pagination() {
        let mut mock_clickhouse = MockSelectQueries::new();

        let before_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();

        mock_clickhouse
            .expect_query_episode_table()
            .withf(move |limit, before, after| {
                assert_eq!(*limit, 20);
                assert_eq!(*before, Some(before_id));
                assert_eq!(*after, Some(after_id));
                true
            })
            .times(1)
            .returning(|_, _, _| Box::pin(async move { Ok(vec![]) }));

        let params = ListEpisodesParams {
            limit: 20,
            before: Some(before_id),
            after: Some(after_id),
        };

        let result = list_episodes(&mock_clickhouse, params).await.unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_query_episode_table_bounds_calls_clickhouse() {
        let mut mock_clickhouse = MockSelectQueries::new();

        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_clickhouse
            .expect_query_episode_table_bounds()
            .times(1)
            .returning(move || {
                Box::pin(async move {
                    Ok(TableBoundsWithCount {
                        first_id: Some(first_id),
                        last_id: Some(last_id),
                        count: 100,
                    })
                })
            });

        let result = query_episode_table_bounds(&mock_clickhouse).await.unwrap();

        assert_eq!(result.first_id, Some(first_id));
        assert_eq!(result.last_id, Some(last_id));
        assert_eq!(result.count, 100);
    }

    #[tokio::test]
    async fn test_query_episode_table_bounds_empty() {
        let mut mock_clickhouse = MockSelectQueries::new();

        mock_clickhouse
            .expect_query_episode_table_bounds()
            .times(1)
            .returning(|| {
                Box::pin(async move {
                    Ok(TableBoundsWithCount {
                        first_id: None,
                        last_id: None,
                        count: 0,
                    })
                })
            });

        let result = query_episode_table_bounds(&mock_clickhouse).await.unwrap();

        assert!(result.first_id.is_none());
        assert!(result.last_id.is_none());
        assert_eq!(result.count, 0);
    }

    #[tokio::test]
    async fn test_query_episode_table_rejects_limit_over_100() {
        let mock_clickhouse = MockSelectQueries::new();

        let params = ListEpisodesParams {
            limit: 101,
            before: None,
            after: None,
        };

        let result = list_episodes(&mock_clickhouse, params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Limit cannot exceed 100"));
    }
}
