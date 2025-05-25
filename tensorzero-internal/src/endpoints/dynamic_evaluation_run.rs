use std::collections::HashMap;

use axum::{
    debug_handler,
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    clickhouse::{escape_string_for_clickhouse_literal, ClickHouseConnectionInfo},
    config_parser::Config,
    endpoints::validate_tags,
    error::{Error, ErrorDetails},
    gateway_util::{AppState, AppStateData, StructuredJson},
    uuid_util::{
        compare_timestamps, generate_dynamic_evaluation_run_episode_id, validate_tensorzero_uuid,
        DYNAMIC_EVALUATION_THRESHOLD,
    },
};

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunInfo {
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DynamicEvaluationRunParams {
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
pub struct DynamicEvaluationRunResponse {
    pub run_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<DynamicEvaluationRunParams>,
) -> Result<Json<DynamicEvaluationRunResponse>, Error> {
    dynamic_evaluation_run(app_state, params).await.map(Json)
}

/// Creates a new dynamic evaluation run.
pub async fn dynamic_evaluation_run(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    params: DynamicEvaluationRunParams,
) -> Result<DynamicEvaluationRunResponse, Error> {
    validate_tags(&params.tags, params.internal)?;
    validate_variant_pins(&params.variants, &config)?;
    let run_id = Uuid::now_v7();
    write_dynamic_evaluation_run(
        clickhouse_connection_info,
        run_id,
        params.variants,
        params.tags,
        params.project_name,
        params.display_name,
    )
    .await?;
    Ok(DynamicEvaluationRunResponse { run_id })
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunEpisodePathParams {
    pub run_id: Uuid,
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct DynamicEvaluationRunEpisodeParams {
    // This has been deprecated in favor of `task_name`.
    #[serde(default)]
    pub datapoint_name: Option<String>,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunEpisodeResponse {
    pub episode_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_episode_handler(
    State(app_state): AppState,
    Path(path_params): Path<DynamicEvaluationRunEpisodePathParams>,
    StructuredJson(params): StructuredJson<DynamicEvaluationRunEpisodeParams>,
) -> Result<Json<DynamicEvaluationRunEpisodeResponse>, Error> {
    dynamic_evaluation_run_episode(app_state, path_params.run_id, params)
        .await
        .map(Json)
}

pub async fn dynamic_evaluation_run_episode(
    AppStateData {
        clickhouse_connection_info,
        ..
    }: AppStateData,
    run_id: Uuid,
    params: DynamicEvaluationRunEpisodeParams,
) -> Result<DynamicEvaluationRunEpisodeResponse, Error> {
    validate_tags(&params.tags, false)?;
    let episode_id = generate_dynamic_evaluation_run_episode_id();
    let run_id_str = run_id.to_string();
    // We add the dynamic evaluation run ID to the tags so that we can look it up per-inference later
    let mut tags = params.tags;
    tags.insert(
        "tensorzero::dynamic_evaluation_run_id".to_string(),
        run_id_str,
    );
    let task_name = get_task_name(params.task_name, params.datapoint_name)?;
    write_dynamic_evaluation_run_episode(
        &clickhouse_connection_info,
        task_name.as_deref(),
        tags,
        run_id,
        episode_id,
    )
    .await?;
    Ok(DynamicEvaluationRunEpisodeResponse { episode_id })
}

pub fn validate_variant_pins(
    variant_pins: &HashMap<String, String>,
    config: &Config<'_>,
) -> Result<(), Error> {
    for (function_name, variant_name) in variant_pins.iter() {
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

fn to_map_literal(map: &HashMap<String, String>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| {
            format!(
                "'{}':'{}'",
                escape_string_for_clickhouse_literal(k),
                escape_string_for_clickhouse_literal(v)
            )
        })
        .collect();
    format!("{{{}}}", items.join(","))
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunRow {
    pub run_id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    pub project_name: Option<String>,
    pub run_display_name: Option<String>,
}

async fn write_dynamic_evaluation_run(
    clickhouse: ClickHouseConnectionInfo,
    run_id: Uuid,
    variant_pins: HashMap<String, String>,
    tags: HashMap<String, String>,
    project_name: Option<String>,
    run_display_name: Option<String>,
) -> Result<(), Error> {
    let query = r#"
    INSERT INTO DynamicEvaluationRun (
        run_id_uint,
        variant_pins,
        tags,
        project_name,
        run_display_name
    )
    VALUES (
        toUInt128({run_id:UUID}),
        {variant_pins:Map(String, String)},
        {tags:Map(String, String)},
        {project_name:Nullable(String)},
        {run_display_name:Nullable(String)}
    )
    "#;
    let mut params = HashMap::new();
    let variant_pins_str = to_map_literal(&variant_pins);
    let tags_str = to_map_literal(&tags);
    let run_id_str = run_id.to_string();
    params.insert("run_id", run_id_str.as_str());
    params.insert("variant_pins", variant_pins_str.as_str());
    params.insert("tags", tags_str.as_str());
    params.insert("project_name", project_name.as_deref().unwrap_or("\\N")); // Use \\N to indicate NULL
    params.insert(
        "run_display_name",
        run_display_name.as_deref().unwrap_or("\\N"),
    ); // Use \\N to indicate NULL
    clickhouse
        .run_query_synchronous(query.to_string(), Some(&params))
        .await?;
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunEpisodeRow {
    pub run_id: Uuid,
    pub episode_id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub datapoint_name: Option<String>,
    pub tags: HashMap<String, String>,
}

async fn write_dynamic_evaluation_run_episode(
    clickhouse: &ClickHouseConnectionInfo,
    datapoint_name: Option<&str>,
    tags: HashMap<String, String>,
    run_id: Uuid,
    episode_id: Uuid,
) -> Result<(), Error> {
    let query = r#"
    INSERT INTO DynamicEvaluationRunEpisode
    (
        run_id,
        episode_id_uint,
        variant_pins,
        datapoint_name,
        tags
    )
    SELECT
        {run_id:UUID} as run_id,
        toUInt128({episode_id:UUID}) as episode_id_uint,
        variant_pins,
        {datapoint_name:Nullable(String)} as datapoint_name,
        mapUpdate(tags, {tags:Map(String, String)}) as tags -- merge the tags in the params on top of tags in the dynamic evaluation run
    FROM DynamicEvaluationRun
    WHERE run_id_uint = toUInt128({run_id:UUID})
    "#;
    let mut query_params = HashMap::new();
    let run_id_str = run_id.to_string();
    let episode_id_str = episode_id.to_string();
    query_params.insert("run_id", run_id_str.as_str());
    query_params.insert("episode_id", episode_id_str.as_str());
    query_params.insert("datapoint_name", datapoint_name.unwrap_or("\\N")); // Use \\N to indicate NULL
    let tags_str = to_map_literal(&tags);
    query_params.insert("tags", tags_str.as_str());
    clickhouse
        .run_query_synchronous(query.to_string(), Some(&query_params))
        .await?;
    Ok(())
}

/// For dynamic evaluation runs, we generate episode IDs that are DYNAMIC_EVALUATION_OFFSET in the future.
/// If we come across an episode ID that is at least DYNAMIC_EVALUATION_THRESHOLD, we need to look up the
/// appropriate DynamicEvaluationRun and then apply the variant_name if unset and the tags if unset.
/// We'll warn if the variant name is set in two places and then take the inference-level one.
pub async fn validate_inference_episode_id_and_apply_dynamic_evaluation_run(
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
    // If the episode ID timestamp is before the dynamic evaluation threshold,
    // it's a regular episode ID and we validate it normally.
    // Otherwise, it's a dynamic evaluation run ID that needs special handling.
    if compare_timestamps(episode_id_timestamp, DYNAMIC_EVALUATION_THRESHOLD) {
        return validate_tensorzero_uuid(episode_id, "Episode");
    }
    let dynamic_evaluation_run = lookup_dynamic_evaluation_run(clickhouse, episode_id).await?;
    let Some(dynamic_evaluation_run) = dynamic_evaluation_run else {
        return Err(Error::new(ErrorDetails::InvalidDynamicEvaluationRun {
            episode_id,
        }));
    };
    if let Some(function_name) = function_name {
        let dynamic_run_variant_name = dynamic_evaluation_run.variant_pins.get(function_name);

        match (variant_name.as_ref(), dynamic_run_variant_name) {
            (Some(_), Some(_)) => {
                tracing::warn!("Variant name set in both inference and dynamic evaluation run");
            }
            // If the inference pinned the variant_name and the dynamic run did not, leave as is
            (Some(_), None) => {}
            // If the dynamic run pinned the variant_name and the inference did not, use the dynamic run variant_name
            (None, Some(dynamic_run_variant_name)) => {
                *variant_name = Some(dynamic_run_variant_name.clone());
            }
            // If both are unset, leave as is
            (None, None) => {}
        }
    }

    // Apply experiment tags from the dynamic evaluation run if they don't already exist in the inference tags
    for (key, value) in dynamic_evaluation_run.tags {
        // Only insert if the key doesn't already exist in the tags
        // This ensures inference-level tags have higher priority
        tags.entry(key).or_insert(value);
    }

    Ok(())
}

async fn lookup_dynamic_evaluation_run(
    clickhouse: &ClickHouseConnectionInfo,
    episode_id: Uuid,
) -> Result<Option<DynamicEvaluationRunInfo>, Error> {
    let query = r#"
    SELECT variant_pins, tags FROM DynamicEvaluationRunEpisode WHERE episode_id_uint = toUInt128({episode_id:UUID}) FORMAT JSONEachRow
    "#;
    let episode_id_str = episode_id.to_string();
    let params = HashMap::from([("episode_id", episode_id_str.as_str())]);
    let result = clickhouse
        .run_query_synchronous(query.to_string(), Some(&params))
        .await?;
    if result.is_empty() {
        return Ok(None);
    }
    let dynamic_evaluation_run: DynamicEvaluationRunInfo =
        serde_json::from_str(&result).map_err(|_| {
            Error::new(ErrorDetails::Serialization {
                message: "Failed to deserialize dynamic evaluation run".to_string(),
            })
        })?;
    Ok(Some(dynamic_evaluation_run))
}

fn get_task_name(
    task_name: Option<String>,
    datapoint_name: Option<String>,
) -> Result<Option<String>, Error> {
    match (task_name, datapoint_name) {
        (Some(task_name), None) => Ok(Some(task_name)),
        (None, Some(datapoint_name)) => {
            tracing::warn!("`datapoint_name` is deprecated in favor of `task_name`. Please change your usage to `task_name`");
            Ok(Some(datapoint_name))
        }
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Err(Error::new(ErrorDetails::InvalidRequest {
            message: "task_name and datapoint_name cannot both be provided".to_string(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;

    #[test]
    #[traced_test]
    fn test_get_task_name() {
        let task_name = get_task_name(Some("task_name".to_string()), None);
        assert_eq!(task_name, Ok(Some("task_name".to_string())));

        assert_eq!(
            get_task_name(
                Some("task_name".to_string()),
                Some("datapoint_name".to_string())
            ),
            Err(Error::new(ErrorDetails::InvalidRequest {
                message: "task_name and datapoint_name cannot both be provided".to_string(),
            }))
        );
        assert!(!logs_contain(
            "`datapoint_name` is deprecated in favor of `task_name`. Please change your usage to `task_name`"
        ));
    }

    #[test]
    #[traced_test]
    fn test_get_task_name_deprecation_warning() {
        let task_name = get_task_name(None, Some("datapoint_name".to_string()));
        assert_eq!(task_name, Ok(Some("datapoint_name".to_string())));
        assert!(logs_contain(
            "`datapoint_name` is deprecated in favor of `task_name`. Please change your usage to `task_name`"
        ));
    }
}
