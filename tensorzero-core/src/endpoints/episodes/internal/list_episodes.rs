//! Episodes endpoint for querying episode data.

use axum::extract::{Query, State};
use axum::{Json, debug_handler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::{EpisodeByIdRow, EpisodeQueries, TableBoundsWithCount};
use crate::endpoints::stored_inferences::v1::types::InferenceFilter;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Query parameters for the GET episode table endpoint
#[derive(Debug, Deserialize)]
pub struct ListEpisodesParams {
    /// Maximum number of episodes to return
    pub limit: u32,
    /// Return episodes before this episode_id (for pagination)
    pub before: Option<Uuid>,
    /// Return episodes after this episode_id (for pagination)
    pub after: Option<Uuid>,
    /// Optional function name to filter episodes by
    pub function_name: Option<String>,
}

/// Request body for the POST episode table endpoint
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ListEpisodesRequest {
    /// Maximum number of episodes to return (max 100)
    #[schemars(extend("maximum" = 100, "minimum" = 1))]
    pub limit: u32,
    /// Return episodes before this episode_id (for pagination)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<Uuid>,
    /// Return episodes after this episode_id (for pagination)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<Uuid>,
    /// Optional function name to filter episodes by
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,
    /// Optional filter to apply when querying episodes.
    /// Episodes are returned if they have at least one inference matching the filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<InferenceFilter>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListEpisodesResponse {
    pub episodes: Vec<EpisodeByIdRow>,
}

/// HTTP handler for querying episodes (GET)
#[debug_handler(state = AppStateData)]
#[instrument(name = "list_episodes_handler", skip_all)]
pub async fn list_episodes_handler(
    State(app_state): AppState,
    Query(params): Query<ListEpisodesParams>,
) -> Result<Json<ListEpisodesResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let episodes = list_episodes(
        &database,
        &app_state.config,
        params.limit,
        params.before,
        params.after,
        params.function_name,
        None,
    )
    .await?;
    Ok(Json(ListEpisodesResponse { episodes }))
}

/// HTTP handler for querying episodes (POST with filters)
#[debug_handler(state = AppStateData)]
#[instrument(name = "list_episodes_post_handler", skip_all)]
pub async fn list_episodes_post_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<ListEpisodesRequest>,
) -> Result<Json<ListEpisodesResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let episodes = list_episodes(
        &database,
        &app_state.config,
        request.limit,
        request.before,
        request.after,
        request.function_name,
        request.filters,
    )
    .await?;
    Ok(Json(ListEpisodesResponse { episodes }))
}

/// Core business logic for listing episodes
pub async fn list_episodes(
    database: &impl EpisodeQueries,
    config: &Config,
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
    function_name: Option<String>,
    filters: Option<InferenceFilter>,
) -> Result<Vec<EpisodeByIdRow>, Error> {
    if limit > 100 {
        return Err(ErrorDetails::InvalidRequest {
            message: "Limit cannot exceed 100".to_string(),
        }
        .into());
    }
    database
        .query_episode_table(config, limit, before, after, function_name, filters)
        .await
}

/// HTTP handler for querying episode table bounds
#[debug_handler(state = AppStateData)]
#[instrument(name = "query_episode_table_bounds_handler", skip_all)]
pub async fn query_episode_table_bounds_handler(
    State(app_state): AppState,
) -> Result<Json<TableBoundsWithCount>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let bounds = query_episode_table_bounds(&database).await?;
    Ok(Json(bounds))
}

/// Core business logic for querying episode table bounds
pub async fn query_episode_table_bounds(
    database: &impl EpisodeQueries,
) -> Result<TableBoundsWithCount, Error> {
    database.query_episode_table_bounds().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MockEpisodeQueries;
    use chrono::Utc;

    fn test_config() -> Config {
        Config::default()
    }

    #[tokio::test]
    async fn test_query_episode_table_calls_database() {
        let mut mock_database = MockEpisodeQueries::new();

        let episode_id = Uuid::now_v7();
        let last_inference_id = Uuid::now_v7();
        let now = Utc::now();

        mock_database
            .expect_query_episode_table()
            .withf(|_config, limit, before, after, function_name, filters| {
                assert_eq!(*limit, 10);
                assert!(before.is_none());
                assert!(after.is_none());
                assert!(function_name.is_none());
                assert!(filters.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _, _, _, _| {
                Ok(vec![EpisodeByIdRow {
                    episode_id,
                    count: 5,
                    start_time: now,
                    end_time: now,
                    last_inference_id,
                }])
            });

        let config = test_config();
        let result = list_episodes(&mock_database, &config, 10, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].episode_id, episode_id);
        assert_eq!(result[0].count, 5);
        assert_eq!(result[0].last_inference_id, last_inference_id);
    }

    #[tokio::test]
    async fn test_query_episode_table_with_pagination() {
        let mut mock_database = MockEpisodeQueries::new();

        let before_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();

        mock_database
            .expect_query_episode_table()
            .withf(
                move |_config, limit, before, after, function_name, filters| {
                    assert_eq!(*limit, 20);
                    assert_eq!(*before, Some(before_id));
                    assert_eq!(*after, Some(after_id));
                    assert!(function_name.is_none());
                    assert!(filters.is_none());
                    true
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _| Ok(vec![]));

        let config = test_config();
        let result = list_episodes(
            &mock_database,
            &config,
            20,
            Some(before_id),
            Some(after_id),
            None,
            None,
        )
        .await
        .unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_query_episode_table_with_function_name() {
        let mut mock_database = MockEpisodeQueries::new();

        mock_database
            .expect_query_episode_table()
            .withf(|_config, limit, before, after, function_name, filters| {
                assert_eq!(*limit, 10);
                assert!(before.is_none());
                assert!(after.is_none());
                assert_eq!(function_name.as_deref(), Some("my_function"));
                assert!(filters.is_none());
                true
            })
            .times(1)
            .returning(|_, _, _, _, _, _| Ok(vec![]));

        let config = test_config();
        let result = list_episodes(
            &mock_database,
            &config,
            10,
            None,
            None,
            Some("my_function".to_string()),
            None,
        )
        .await
        .unwrap();

        assert!(result.is_empty(), "Should return empty results");
    }

    #[tokio::test]
    async fn test_query_episode_table_bounds_calls_database() {
        let mut mock_database = MockEpisodeQueries::new();

        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_database
            .expect_query_episode_table_bounds()
            .times(1)
            .returning(move || {
                Ok(TableBoundsWithCount {
                    first_id: Some(first_id),
                    last_id: Some(last_id),
                    count: 100,
                })
            });

        let result = query_episode_table_bounds(&mock_database).await.unwrap();

        assert_eq!(result.first_id, Some(first_id));
        assert_eq!(result.last_id, Some(last_id));
        assert_eq!(result.count, 100);
    }

    #[tokio::test]
    async fn test_query_episode_table_bounds_empty() {
        let mut mock_database = MockEpisodeQueries::new();

        mock_database
            .expect_query_episode_table_bounds()
            .times(1)
            .returning(|| {
                Ok(TableBoundsWithCount {
                    first_id: None,
                    last_id: None,
                    count: 0,
                })
            });

        let result = query_episode_table_bounds(&mock_database).await.unwrap();

        assert!(result.first_id.is_none());
        assert!(result.last_id.is_none());
        assert_eq!(result.count, 0);
    }

    #[tokio::test]
    async fn test_query_episode_table_rejects_limit_over_100() {
        let mock_database = MockEpisodeQueries::new();
        let config = test_config();

        let result = list_episodes(&mock_database, &config, 101, None, None, None, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Limit cannot exceed 100"));
    }
}
