use std::collections::HashMap;

use axum::{
    Json, debug_handler,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::Config,
    db::{
        clickhouse::ClickHouseConnectionInfo,
        workflow_evaluation_queries::WorkflowEvaluationQueries,
    },
    endpoints::validate_tags,
    error::{Error, ErrorDetails},
    utils::{
        gateway::{AppState, AppStateData, StructuredJson},
        uuid::{
            WORKFLOW_EVALUATION_THRESHOLD, compare_timestamps,
            generate_workflow_evaluation_run_episode_id, validate_tensorzero_uuid,
        },
    },
};

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowEvaluationRunParams {
    pub variants: HashMap<String, String>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub internal: bool, // For internal use only
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowEvaluationRunResponse {
    pub run_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn workflow_evaluation_run_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<WorkflowEvaluationRunParams>,
) -> Result<Json<WorkflowEvaluationRunResponse>, Error> {
    workflow_evaluation_run(app_state, params).await.map(Json)
}

/// Creates a new workflow evaluation run.
pub async fn workflow_evaluation_run(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    params: WorkflowEvaluationRunParams,
) -> Result<WorkflowEvaluationRunResponse, Error> {
    validate_tags(&params.tags, params.internal)?;
    validate_variant_pins(&params.variants, &config)?;
    let run_id = Uuid::now_v7();
    clickhouse_connection_info
        .insert_workflow_evaluation_run(
            run_id,
            &params.variants,
            &params.tags,
            params.project_name.as_deref(),
            params.display_name.as_deref(),
            &config.hash,
        )
        .await?;
    Ok(WorkflowEvaluationRunResponse { run_id })
}

#[derive(Debug, Deserialize, Serialize)]
pub struct WorkflowEvaluationRunEpisodePathParams {
    pub run_id: Uuid,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct WorkflowEvaluationRunEpisodeParams {
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct WorkflowEvaluationRunEpisodeResponse {
    pub episode_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn workflow_evaluation_run_episode_handler(
    State(app_state): AppState,
    Path(path_params): Path<WorkflowEvaluationRunEpisodePathParams>,
    StructuredJson(params): StructuredJson<WorkflowEvaluationRunEpisodeParams>,
) -> Result<Json<WorkflowEvaluationRunEpisodeResponse>, Error> {
    workflow_evaluation_run_episode(app_state, path_params.run_id, params)
        .await
        .map(Json)
}

pub async fn workflow_evaluation_run_episode(
    AppStateData {
        clickhouse_connection_info,
        config,
        ..
    }: AppStateData,
    run_id: Uuid,
    params: WorkflowEvaluationRunEpisodeParams,
) -> Result<WorkflowEvaluationRunEpisodeResponse, Error> {
    validate_tags(&params.tags, false)?;
    let episode_id = generate_workflow_evaluation_run_episode_id();
    let run_id_str = run_id.to_string();

    // IMPORTANT: We write both the old and new tag names for backward compatibility.
    // - tensorzero::dynamic_evaluation_run_id (OLD): Historical tag name, kept for existing queries/UI
    // - tensorzero::workflow_evaluation_run_id (NEW): Updated tag name following feature rename
    //
    // The UI and all queries currently use the old tag name. In a future migration, we will:
    // 1. Update UI and queries to use the new tag name
    // 2. Backfill old inferences to add the new tag
    // 3. Remove the old tag from new writes
    //
    // Similar to how database tables kept "DynamicEvaluation" names but code uses "workflow_evaluation",
    // we're maintaining both tag names during the transition period.
    let mut tags = params.tags;
    tags.insert(
        "tensorzero::dynamic_evaluation_run_id".to_string(),
        run_id_str.clone(),
    );
    tags.insert(
        "tensorzero::workflow_evaluation_run_id".to_string(),
        run_id_str,
    );
    clickhouse_connection_info
        .insert_workflow_evaluation_run_episode(
            run_id,
            episode_id,
            params.task_name.as_deref(),
            &tags,
            &config.hash,
        )
        .await?;
    Ok(WorkflowEvaluationRunEpisodeResponse { episode_id })
}

pub fn validate_variant_pins(
    variant_pins: &HashMap<String, String>,
    config: &Config,
) -> Result<(), Error> {
    for (function_name, variant_name) in variant_pins {
        let function_config = config.get_function(function_name)?;
        function_config
            .variants()
            .get(variant_name)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Variant {variant_name} for function {function_name} not found.",
                    ),
                })
            })?;
    }
    Ok(())
}

/// For workflow evaluation runs, we generate episode IDs that are WORKFLOW_EVALUATION_OFFSET in the future.
/// If we come across an episode ID that is at least WORKFLOW_EVALUATION_THRESHOLD, we need to look up the
/// appropriate workflow evaluation run and then apply the `variant_name` if unset and the tags if unset.
/// We'll warn if the variant name is set in two places and then take the inference-level one.
pub async fn validate_inference_episode_id_and_apply_workflow_evaluation_run(
    episode_id: Uuid,
    function_name: Option<&String>,
    variant_name: &mut Option<String>,
    tags: &mut HashMap<String, String>,
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<(), Error> {
    let episode_id_timestamp = episode_id.get_timestamp().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidUuid {
            raw_uuid: episode_id.to_string(),
        })
    })?;
    // If the episode ID timestamp is before the workflow evaluation threshold,
    // it's a regular episode ID and we validate it normally.
    // Otherwise, it's a workflow evaluation run ID that needs special handling.
    if compare_timestamps(episode_id_timestamp, WORKFLOW_EVALUATION_THRESHOLD) {
        return validate_tensorzero_uuid(episode_id, "Episode");
    }
    let workflow_evaluation_run = clickhouse
        .get_workflow_evaluation_run_by_episode_id(episode_id)
        .await?;
    let Some(workflow_evaluation_run) = workflow_evaluation_run else {
        return Err(Error::new(ErrorDetails::InvalidWorkflowEvaluationRun {
            episode_id,
        }));
    };
    if let Some(function_name) = function_name {
        let workflow_run_variant_name = workflow_evaluation_run.variant_pins.get(function_name);

        match (variant_name.as_ref(), workflow_run_variant_name) {
            (Some(_), Some(_)) => {
                tracing::warn!("Variant name set in both inference and workflow evaluation run");
            }
            // If the inference pinned the `variant_name` and the workflow run did not, leave as is
            (Some(_), None) => {}
            // If the workflow run pinned the `variant_name` and the inference did not, use the workflow run `variant_name`
            (None, Some(workflow_run_variant_name)) => {
                *variant_name = Some(workflow_run_variant_name.clone());
            }
            // If both are unset, leave as is
            (None, None) => {}
        }
    }

    // Apply experiment tags from the workflow evaluation run if they don't already exist in the inference tags
    for (key, value) in workflow_evaluation_run.tags {
        // Only insert if the key doesn't already exist in the tags
        // This ensures inference-level tags have higher priority
        tags.entry(key).or_insert(value);
    }

    Ok(())
}

// ============================================================================
// DEPRECATED HANDLERS - For backward compatibility
// ============================================================================

/// DEPRECATED: Use the POST `/workflow_evaluation_run` endpoint instead.
#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<WorkflowEvaluationRunParams>,
) -> Result<Json<WorkflowEvaluationRunResponse>, Error> {
    tracing::warn!(
        "DEPRECATED: The `/dynamic_evaluation_run` endpoint is deprecated. Please use `/workflow_evaluation_run` instead. Support for `/dynamic_evaluation_run` will be removed in a future version."
    );
    workflow_evaluation_run_handler(State(app_state), StructuredJson(params)).await
}

/// DEPRECATED: Use the POST `/workflow_evaluation_run/{run_id}/episode` endpoint instead.
#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_episode_handler(
    State(app_state): AppState,
    Path(path_params): Path<WorkflowEvaluationRunEpisodePathParams>,
    StructuredJson(params): StructuredJson<WorkflowEvaluationRunEpisodeParams>,
) -> Result<Json<WorkflowEvaluationRunEpisodeResponse>, Error> {
    tracing::warn!(
        run_id = %path_params.run_id,
        "DEPRECATED: The `/dynamic_evaluation_run/{{run_id}}/episode` endpoint is deprecated. Please use `/workflow_evaluation_run/{{run_id}}/episode` instead. Support for `/dynamic_evaluation_run/{{run_id}}/episode` will be removed in a future version."
    );
    workflow_evaluation_run_episode_handler(
        State(app_state),
        Path(path_params),
        StructuredJson(params),
    )
    .await
}
