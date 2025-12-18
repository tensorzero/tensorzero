use crate::{
    db::inference_stats::InferenceStatsQueries,
    error::Error,
    utils::gateway::{AppState, AppStateData},
};
use axum::{
    Json, debug_handler,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetEpisodeInferenceCountResponse {
    pub inference_count: u64,
}

/// HTTP handler for getting inference stats for an episode
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
    let stats =
        get_episode_inference_count(&app_state.clickhouse_connection_info, episode_id).await?;
    Ok(Json(stats))
}

/// Core business logic for getting episode inference statistics
pub async fn get_episode_inference_count(
    clickhouse: &impl InferenceStatsQueries,
    episode_id: Uuid,
) -> Result<GetEpisodeInferenceCountResponse, Error> {
    let inference_count = clickhouse.count_inferences_for_episode(episode_id).await?;

    Ok(GetEpisodeInferenceCountResponse { inference_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::inference_stats::MockInferenceStatsQueries;

    #[tokio::test]
    async fn test_get_episode_inference_count_calls_clickhouse() {
        let mut mock_clickhouse = MockInferenceStatsQueries::new();

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
