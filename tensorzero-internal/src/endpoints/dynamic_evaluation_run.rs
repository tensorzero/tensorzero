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
    uuid_util::generate_dynamic_evaluation_run_episode_id,
};

#[derive(Debug, Deserialize)]
pub struct Params {
    pub variants: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct DynamicEvaluationRun {
    pub episode_id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct DynamicEvaluationRunResponse {
    pub episode_id: Uuid,
}

#[debug_handler(state = AppStateData)]
pub async fn dynamic_evaluation_run_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Json<DynamicEvaluationRunResponse>, Error> {
    dynamic_evaluation_run(app_state, params).await
}

pub async fn dynamic_evaluation_run(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    params: Params,
) -> Result<Json<DynamicEvaluationRunResponse>, Error> {
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
    Ok(Json(DynamicEvaluationRunResponse { episode_id }))
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

async fn write_dynamic_evaluation_run(
    clickhouse: ClickHouseConnectionInfo,
    run_id: Uuid,
    variant_pins: HashMap<String, String>,
    tags: HashMap<String, String>,
) -> Result<(), Error> {
    let dynamic_evaluation_run = DynamicEvaluationRun {
        episode_id: run_id,
        variant_pins,
        tags,
    };
    clickhouse
        .write(&[dynamic_evaluation_run], "DynamicEvaluationRun")
        .await?;
    Ok(())
}
