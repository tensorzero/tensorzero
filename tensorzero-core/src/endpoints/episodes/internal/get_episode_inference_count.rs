use crate::{
    config::Config,
    db::inferences::{CountInferencesParams, InferenceQueries},
    error::Error,
    feature_flags::ENABLE_POSTGRES_AS_PRIMARY_DATASTORE,
    utils::gateway::{AppState, AppStateData},
};
use axum::{
    Json, debug_handler,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetEpisodeInferenceCountResponse {
    pub inference_count: u64,
}

/// HTTP handler for getting inference counts for an episode
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "get_episode_inference_count_handler",
    skip_all,
    fields(
        episode_id = %episode_id,
    )
)]
pub async fn get_episode_inference_count_handler(
    State(app_state): AppState,
    Path(episode_id): Path<Uuid>,
) -> Result<Json<GetEpisodeInferenceCountResponse>, Error> {
    let stats = if ENABLE_POSTGRES_AS_PRIMARY_DATASTORE.get() {
        get_episode_inference_count(
            &app_state.config,
            &app_state.postgres_connection_info,
            episode_id,
        )
        .await?
    } else {
        get_episode_inference_count(
            &app_state.config,
            &app_state.clickhouse_connection_info,
            episode_id,
        )
        .await?
    };
    Ok(Json(stats))
}

/// Core business logic for getting episode inference counts
pub async fn get_episode_inference_count(
    config: &Config,
    database: &impl InferenceQueries,
    episode_id: Uuid,
) -> Result<GetEpisodeInferenceCountResponse, Error> {
    let count_params = CountInferencesParams {
        episode_id: Some(&episode_id),
        ..Default::default()
    };
    let inference_count = database.count_inferences(config, &count_params).await?;

    Ok(GetEpisodeInferenceCountResponse { inference_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::inferences::MockInferenceQueries;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_get_episode_inference_count_calls_database() {
        let mut mock_database = MockInferenceQueries::new();
        let config = Arc::new(Config::default());

        let episode_id = Uuid::now_v7();
        let expected_count = 42;

        mock_database
            .expect_count_inferences()
            .withf(move |_, params| params.episode_id == Some(&episode_id))
            .times(1)
            .returning(move |_, _| Box::pin(async move { Ok(expected_count) }));

        let result = get_episode_inference_count(&config, &mock_database, episode_id)
            .await
            .unwrap();

        assert_eq!(result.inference_count, expected_count);
    }
}
