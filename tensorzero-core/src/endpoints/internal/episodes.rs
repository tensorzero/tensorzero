//! Episodes endpoint for querying episode data.

use axum::extract::{Path, Query, State};
use axum::{Json, debug_handler};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use crate::db::inference_stats::InferenceStatsQueries;
use crate::db::{EpisodeByIdRow, SelectQueries, TableBoundsWithCount};
use crate::endpoints::internal::inference_stats::InferenceStatsResponse;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the episode table endpoint
#[derive(Debug, Deserialize)]
pub struct QueryEpisodeTableParams {
    /// Maximum number of episodes to return
    pub limit: u32,
    /// Return episodes before this episode_id (for pagination)
    pub before: Option<Uuid>,
    /// Return episodes after this episode_id (for pagination)
    pub after: Option<Uuid>,
}

/// HTTP handler for querying episodes
#[debug_handler(state = AppStateData)]
#[instrument(name = "query_episode_table_handler", skip_all)]
pub async fn query_episode_table_handler(
    State(app_state): AppState,
    Query(params): Query<QueryEpisodeTableParams>,
) -> Result<Json<Vec<EpisodeByIdRow>>, Error> {
    let episodes = query_episode_table(&app_state.clickhouse_connection_info, params).await?;
    Ok(Json(episodes))
}

/// Core business logic for querying episodes
pub async fn query_episode_table(
    clickhouse: &impl SelectQueries,
    params: QueryEpisodeTableParams,
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

/// HTTP handler for getting inference stats for an episode
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_episode_inference_stats_handler",
    skip_all,
    fields(
        episode_id = %episode_id,
    )
)]
pub async fn get_episode_inference_stats_handler(
    State(app_state): AppState,
    Path(episode_id): Path<Uuid>,
) -> Result<Json<InferenceStatsResponse>, Error> {
    let stats =
        get_episode_inference_stats(&app_state.clickhouse_connection_info, episode_id).await?;
    Ok(Json(stats))
}

/// Core business logic for getting episode inference statistics
pub async fn get_episode_inference_stats(
    clickhouse: &impl InferenceStatsQueries,
    episode_id: Uuid,
) -> Result<InferenceStatsResponse, Error> {
    let inference_count = clickhouse.count_inferences_for_episode(episode_id).await?;

    Ok(InferenceStatsResponse { inference_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MockSelectQueries;
    use crate::db::inference_stats::MockInferenceStatsQueries;
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

        let params = QueryEpisodeTableParams {
            limit: 10,
            before: None,
            after: None,
        };

        let result = query_episode_table(&mock_clickhouse, params).await.unwrap();

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

        let params = QueryEpisodeTableParams {
            limit: 20,
            before: Some(before_id),
            after: Some(after_id),
        };

        let result = query_episode_table(&mock_clickhouse, params).await.unwrap();

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

        let params = QueryEpisodeTableParams {
            limit: 101,
            before: None,
            after: None,
        };

        let result = query_episode_table(&mock_clickhouse, params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Limit cannot exceed 100"));
    }

    #[tokio::test]
    async fn test_get_episode_inference_stats_calls_clickhouse() {
        let mut mock_clickhouse = MockInferenceStatsQueries::new();

        let episode_id = Uuid::now_v7();
        let expected_count = 42;

        mock_clickhouse
            .expect_count_inferences_for_episode()
            .withf(move |id| *id == episode_id)
            .times(1)
            .returning(move |_| Box::pin(async move { Ok(expected_count) }));

        let result = get_episode_inference_stats(&mock_clickhouse, episode_id)
            .await
            .unwrap();

        assert_eq!(result.inference_count, expected_count);
    }
}
