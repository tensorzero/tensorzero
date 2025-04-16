use std::collections::HashMap;

use axum::{debug_handler, extract::State, Json};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    clickhouse::ClickHouseConnectionInfo,
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
    pub experiment_tags: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Params {
    pub variants: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamicEvaluationRunResponse {
    pub episode_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<DynamicEvaluationRunResponse>, Error> {
    dynamic_evaluation_run(app_state, params).await.map(Json)
}

pub async fn dynamic_evaluation_run(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    params: Params,
) -> Result<DynamicEvaluationRunResponse, Error> {
    validate_tags(&params.tags, false)?;
    validate_variant_pins(&params.variants, &config)?;
    let episode_id = generate_dynamic_evaluation_run_episode_id();
    write_dynamic_evaluation_run(
        clickhouse_connection_info,
        episode_id,
        params.variants,
        params.tags,
    )
    .await?;
    Ok(DynamicEvaluationRunResponse { episode_id })
}

fn validate_variant_pins(
    variant_pins: &HashMap<String, String>,
    config: &Config<'_>,
) -> Result<(), Error> {
    for (function_name, variant_name) in variant_pins.iter() {
        let function_config = config.get_function(function_name)?;
        let variant_config = function_config.variants().get(variant_name);
        if variant_config.is_none() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Variant {} for function {} not found.",
                    variant_name, function_name
                ),
            }));
        }
    }
    Ok(())
}

fn to_map_literal(map: &HashMap<String, String>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("'{}':'{}'", k, v))
        .collect();
    format!("{{{}}}", items.join(","))
}

async fn write_dynamic_evaluation_run(
    clickhouse: ClickHouseConnectionInfo,
    episode_id: Uuid,
    variant_pins: HashMap<String, String>,
    tags: HashMap<String, String>,
) -> Result<(), Error> {
    // The short key is the least significant 64 bits of the episode ID.
    // These are randomly generated, so we can use them as a unique identifier for the dynamic evaluation run.
    let short_key = episode_id.as_u64_pair().1;
    let query = r#"
    INSERT INTO DynamicEvaluationRun (short_key, episode_id, variant_pins, experiment_tags)
    VALUES ({short_key:UInt64}, {episode_id:UUID}, {variant_pins:Map(String, String)}, {experiment_tags:Map(String, String)})
    "#;
    let mut params = HashMap::new();
    let variant_pins_str = to_map_literal(&variant_pins);
    let tags_str = to_map_literal(&tags);
    let short_key_str = short_key.to_string();
    let episode_id_str = episode_id.to_string();
    params.insert("short_key", short_key_str.as_str());
    params.insert("episode_id", episode_id_str.as_str());
    params.insert("variant_pins", variant_pins_str.as_str());
    params.insert("experiment_tags", tags_str.as_str());
    clickhouse
        .run_query_synchronous(query.to_string(), Some(&params))
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
    for (key, value) in dynamic_evaluation_run.experiment_tags {
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
    SELECT variant_pins, experiment_tags FROM DynamicEvaluationRun WHERE short_key = {short_key:UInt64} AND episode_id = {episode_id:UUID} FORMAT JSONEachRow
    "#;
    let mut params = HashMap::new();
    let short_key_str = episode_id.as_u64_pair().1.to_string();
    let episode_id_str = episode_id.to_string();
    params.insert("short_key", short_key_str.as_str());
    params.insert("episode_id", episode_id_str.as_str());
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
