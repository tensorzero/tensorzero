//! Handler for listing workflow evaluation run episodes grouped by task name.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::ListWorkflowEvaluationRunEpisodesByTaskNameResponse;
use crate::db::workflow_evaluation_queries::{
    GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow, WorkflowEvaluationQueries,
};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for listing workflow evaluation run episodes by task name.
#[derive(Debug, Deserialize)]
pub struct ListWorkflowEvaluationRunEpisodesByTaskNameParams {
    /// Comma-separated list of run IDs to filter episodes by.
    pub run_ids: Option<String>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

const DEFAULT_LIMIT: u32 = 15;
const DEFAULT_OFFSET: u32 = 0;

/// Handler for `GET /internal/workflow_evaluations/episodes_by_task_name`
///
/// Returns a list of workflow evaluation run episodes grouped by task_name.
/// Episodes with NULL task_name are grouped individually.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.list_episodes_by_task_name", skip_all)]
pub async fn list_workflow_evaluation_run_episodes_by_task_name_handler(
    State(app_state): AppState,
    Query(params): Query<ListWorkflowEvaluationRunEpisodesByTaskNameParams>,
) -> Result<Json<ListWorkflowEvaluationRunEpisodesByTaskNameResponse>, Error> {
    // Parse run_ids from comma-separated string
    let run_ids = params
        .run_ids
        .map(|s| {
            s.split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| Uuid::parse_str(s.trim()).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let response = list_workflow_evaluation_run_episodes_by_task_name(
        &app_state.clickhouse_connection_info,
        &run_ids,
        params.limit.unwrap_or(DEFAULT_LIMIT),
        params.offset.unwrap_or(DEFAULT_OFFSET),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for listing workflow evaluation run episodes by task name
pub async fn list_workflow_evaluation_run_episodes_by_task_name(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_ids: &[Uuid],
    limit: u32,
    offset: u32,
) -> Result<ListWorkflowEvaluationRunEpisodesByTaskNameResponse, Error> {
    let episodes = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(run_ids, limit, offset)
        .await?;

    // Group episodes by group_key
    let mut groups: std::collections::HashMap<
        String,
        Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>,
    > = std::collections::HashMap::new();

    for episode in episodes {
        groups
            .entry(episode.group_key.clone())
            .or_default()
            .push(episode);
    }

    // Convert to Vec<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>> preserving order
    // We need to maintain the order from the database, so use the first occurrence order
    let mut seen_groups = std::collections::HashSet::new();
    let mut group_order = Vec::new();

    // Re-iterate to get the order from original episodes
    let episodes_for_ordering = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(run_ids, limit, offset)
        .await?;

    for episode in episodes_for_ordering {
        if seen_groups.insert(episode.group_key.clone()) {
            group_order.push(episode.group_key);
        }
    }

    let grouped_episodes: Vec<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>> =
        group_order
            .into_iter()
            .filter_map(|key| groups.remove(&key))
            .collect();

    Ok(ListWorkflowEvaluationRunEpisodesByTaskNameResponse {
        episodes: grouped_episodes,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;

    fn create_test_episode(
        group_key: &str,
        task_name: Option<&str>,
        run_id: Uuid,
    ) -> GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow {
        GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow {
            group_key: group_key.to_string(),
            episode_id: Uuid::now_v7(),
            timestamp: Utc::now(),
            run_id,
            tags: HashMap::new(),
            task_name: task_name.map(|s| s.to_string()),
            feedback_metric_names: vec![],
            feedback_values: vec![],
        }
    }

    #[tokio::test]
    async fn test_list_episodes_by_task_name_empty_run_ids() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_run_episodes_by_task_name()
            .withf(|run_ids, limit, offset| {
                assert!(run_ids.is_empty());
                assert_eq!(*limit, 15);
                assert_eq!(*offset, 0);
                true
            })
            .times(2) // Called twice - once for episodes, once for ordering
            .returning(|_, _, _| Box::pin(async move { Ok(vec![]) }));

        let result =
            list_workflow_evaluation_run_episodes_by_task_name(&mock_clickhouse, &[], 15, 0)
                .await
                .unwrap();

        assert!(result.episodes.is_empty());
    }

    #[tokio::test]
    async fn test_list_episodes_by_task_name_groups_correctly() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();

        mock_clickhouse
            .expect_list_workflow_evaluation_run_episodes_by_task_name()
            .times(2)
            .returning(move |_, _, _| {
                let run_id_clone = run_id;
                Box::pin(async move {
                    Ok(vec![
                        create_test_episode("task1", Some("task1"), run_id_clone),
                        create_test_episode("task1", Some("task1"), run_id_clone),
                        create_test_episode("task2", Some("task2"), run_id_clone),
                    ])
                })
            });

        let result =
            list_workflow_evaluation_run_episodes_by_task_name(&mock_clickhouse, &[run_id], 100, 0)
                .await
                .unwrap();

        assert_eq!(result.episodes.len(), 2); // Two groups
        assert_eq!(result.episodes[0].len(), 2); // First group has 2 episodes
        assert_eq!(result.episodes[1].len(), 1); // Second group has 1 episode
    }

    #[tokio::test]
    async fn test_list_episodes_by_task_name_with_pagination() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();

        mock_clickhouse
            .expect_list_workflow_evaluation_run_episodes_by_task_name()
            .withf(move |run_ids, limit, offset| {
                assert_eq!(run_ids.len(), 1);
                assert_eq!(*limit, 10);
                assert_eq!(*offset, 5);
                true
            })
            .times(2)
            .returning(move |_, _, _| {
                let run_id_clone = run_id;
                Box::pin(async move {
                    Ok(vec![create_test_episode(
                        "task3",
                        Some("task3"),
                        run_id_clone,
                    )])
                })
            });

        let result =
            list_workflow_evaluation_run_episodes_by_task_name(&mock_clickhouse, &[run_id], 10, 5)
                .await
                .unwrap();

        assert_eq!(result.episodes.len(), 1);
    }
}
