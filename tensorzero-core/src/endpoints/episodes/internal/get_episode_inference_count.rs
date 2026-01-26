use crate::{
    db::inference_count::InferenceCountQueries,
    error::Error,
    feature_flags::ENABLE_POSTGRES_READ,
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
    let stats = if ENABLE_POSTGRES_READ.get() {
        get_episode_inference_count(&app_state.postgres_connection_info, episode_id).await?
    } else {
        get_episode_inference_count(&app_state.clickhouse_connection_info, episode_id).await?
    };
    Ok(Json(stats))
}

/// Core business logic for getting episode inference counts
pub async fn get_episode_inference_count(
    clickhouse: &impl InferenceCountQueries,
    episode_id: Uuid,
) -> Result<GetEpisodeInferenceCountResponse, Error> {
    let inference_count = clickhouse.count_inferences_for_episode(episode_id).await?;

    Ok(GetEpisodeInferenceCountResponse { inference_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::inference_count::MockInferenceCountQueries;

    #[tokio::test]
    async fn test_get_episode_inference_count_calls_clickhouse() {
        let mut mock_clickhouse = MockInferenceCountQueries::new();

        let episode_id = Uuid::now_v7();
        let expected_count = 42;

        mock_clickhouse
            .expect_count_inferences_for_episode()
            .withf(move |id| *id == episode_id)
            .times(1)
            .returning(move |_| Box::pin(async move { Ok(expected_count) }));

        let result = get_episode_inference_count(&mock_clickhouse, episode_id)
            .await
            .unwrap();

        assert_eq!(result.inference_count, expected_count);
    }
}
