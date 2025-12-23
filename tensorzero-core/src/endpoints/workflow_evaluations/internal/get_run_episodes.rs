//! Handler for getting workflow evaluation run episodes with feedback.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::{
    GetWorkflowEvaluationRunEpisodesWithFeedbackResponse, WorkflowEvaluationRunEpisodeWithFeedback,
};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting workflow evaluation run episodes with feedback.
#[derive(Debug, Deserialize)]
pub struct GetWorkflowEvaluationRunEpisodesParams {
    /// The run ID to get episodes for
    pub run_id: Uuid,
    /// Maximum number of episodes to return (default: 15)
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Offset for pagination (default: 0)
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    15
}

/// Handler for `GET /internal/workflow_evaluations/run_episodes`
///
/// Gets workflow evaluation run episodes with their feedback for a specific run.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.get_run_episodes", skip_all)]
pub async fn get_workflow_evaluation_run_episodes_handler(
    State(app_state): AppState,
    Query(params): Query<GetWorkflowEvaluationRunEpisodesParams>,
) -> Result<Json<GetWorkflowEvaluationRunEpisodesWithFeedbackResponse>, Error> {
    let response = get_workflow_evaluation_run_episodes(
        &app_state.clickhouse_connection_info,
        params.run_id,
        params.limit,
        params.offset,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting workflow evaluation run episodes with feedback
pub async fn get_workflow_evaluation_run_episodes(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_id: Uuid,
    limit: u32,
    offset: u32,
) -> Result<GetWorkflowEvaluationRunEpisodesWithFeedbackResponse, Error> {
    let episodes_database = clickhouse
        .get_workflow_evaluation_run_episodes_with_feedback(run_id, limit, offset)
        .await?;
    let episodes = episodes_database
        .into_iter()
        .map(|episode| WorkflowEvaluationRunEpisodeWithFeedback {
            episode_id: episode.episode_id,
            timestamp: episode.timestamp,
            run_id: episode.run_id,
            tags: episode.tags,
            task_name: episode.task_name,
            feedback_metric_names: episode.feedback_metric_names,
            feedback_values: episode.feedback_values,
        })
        .collect();
    Ok(GetWorkflowEvaluationRunEpisodesWithFeedbackResponse { episodes })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::{
        MockWorkflowEvaluationQueries, WorkflowEvaluationRunEpisodeWithFeedbackRow,
    };

    fn create_test_episode(
        episode_id: Uuid,
        run_id: Uuid,
        task_name: Option<&str>,
        metric_names: Vec<&str>,
        values: Vec<&str>,
    ) -> WorkflowEvaluationRunEpisodeWithFeedbackRow {
        WorkflowEvaluationRunEpisodeWithFeedbackRow {
            episode_id,
            timestamp: Utc::now(),
            run_id,
            tags: HashMap::new(),
            task_name: task_name.map(|s| s.to_string()),
            feedback_metric_names: metric_names.into_iter().map(|s| s.to_string()).collect(),
            feedback_values: values.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_empty() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_episodes_with_feedback()
            .withf(move |id, limit, offset| {
                assert_eq!(*id, run_id);
                assert_eq!(*limit, 10);
                assert_eq!(*offset, 0);
                true
            })
            .times(1)
            .returning(|_, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_workflow_evaluation_run_episodes(&mock_clickhouse, run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.episodes.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_with_feedback() {
        let run_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_episodes_with_feedback()
            .times(1)
            .returning(move |_, _, _| {
                let episode = create_test_episode(
                    episode_id,
                    run_id,
                    None,
                    vec!["elapsed_ms", "solved"],
                    vec!["100.0", "true"],
                );
                Box::pin(async move { Ok(vec![episode]) })
            });

        let result = get_workflow_evaluation_run_episodes(&mock_clickhouse, run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.episodes.len(), 1);
        assert_eq!(result.episodes[0].episode_id, episode_id);
        assert_eq!(result.episodes[0].run_id, run_id);
        assert_eq!(result.episodes[0].task_name, None);
        assert_eq!(
            result.episodes[0].feedback_metric_names,
            vec!["elapsed_ms", "solved"]
        );
        assert_eq!(result.episodes[0].feedback_values, vec!["100.0", "true"]);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_with_task_name() {
        let run_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_episodes_with_feedback()
            .times(1)
            .returning(move |_, _, _| {
                let episode = create_test_episode(
                    episode_id,
                    run_id,
                    Some("my_task"),
                    vec!["metric1"],
                    vec!["value1"],
                );
                Box::pin(async move { Ok(vec![episode]) })
            });

        let result = get_workflow_evaluation_run_episodes(&mock_clickhouse, run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.episodes.len(), 1);
        assert_eq!(result.episodes[0].task_name, Some("my_task".to_string()));
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_multiple() {
        let run_id = Uuid::now_v7();
        let episode_id1 = Uuid::now_v7();
        let episode_id2 = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_episodes_with_feedback()
            .times(1)
            .returning(move |_, _, _| {
                let episode1 =
                    create_test_episode(episode_id1, run_id, None, vec!["metric1"], vec!["val1"]);
                let episode2 = create_test_episode(
                    episode_id2,
                    run_id,
                    Some("task"),
                    vec!["metric2"],
                    vec!["val2"],
                );
                Box::pin(async move { Ok(vec![episode1, episode2]) })
            });

        let result = get_workflow_evaluation_run_episodes(&mock_clickhouse, run_id, 10, 0)
            .await
            .unwrap();

        assert_eq!(result.episodes.len(), 2);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_run_episodes_pagination() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_run_episodes_with_feedback()
            .withf(move |id, limit, offset| {
                assert_eq!(*id, run_id);
                assert_eq!(*limit, 5);
                assert_eq!(*offset, 10);
                true
            })
            .times(1)
            .returning(|_, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_workflow_evaluation_run_episodes(&mock_clickhouse, run_id, 5, 10)
            .await
            .unwrap();

        assert_eq!(result.episodes.len(), 0);
    }
}
