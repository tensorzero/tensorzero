use axum::{
    body::Body,
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    config::Config,
    db::clickhouse::{
        query_builder::{InferenceFilter, InferenceOutputSource, ListInferencesParams, OrderBy},
        ClickHouseConnectionInfo, ClickhouseFormat,
    },
    endpoints::{inference::InferenceCredentials, stored_inference::render_samples},
    error::Error,
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        JobHandle, OptimizationJobHandle, OptimizationJobInfo, Optimizer,
        UninitializedOptimizerInfo,
    },
    serde_util::deserialize_option_u64,
    stored_inference::RenderedSample,
    utils::gateway::{AppState, AppStateData, StructuredJson},
};

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationWorkflowParams {
    pub function_name: String,
    pub template_variant_name: String,
    pub query_variant_name: Option<String>,
    pub filters: Option<InferenceFilter>,
    pub output_source: InferenceOutputSource,
    pub order_by: Option<Vec<OrderBy>>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub limit: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub offset: Option<u64>,
    pub val_fraction: Option<f64>,
    #[serde(default)]
    pub format: ClickhouseFormat,
    pub optimizer_config: UninitializedOptimizerInfo,
}

/// TODO: We should deprecate this method/endpoint once the new `launch_optimization_workflow` is more stable and exposed externally.
pub async fn launch_optimization_workflow_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        ..
    }): AppState,
    StructuredJson(params): StructuredJson<LaunchOptimizationWorkflowParams>,
) -> Result<Response<Body>, Error> {
    let job_handle =
        launch_optimization_workflow(&http_client, config, &clickhouse_connection_info, params)
            .await?;
    let encoded_job_handle = job_handle.to_base64_urlencoded()?;
    Ok(encoded_job_handle.into_response())
}

/// Starts an optimization job.
/// This function will query inferences from the database,
/// render them by fetching any network resources needed and
/// templating them with the template variant,
/// and launch the optimization job specified.
pub async fn launch_optimization_workflow(
    http_client: &TensorzeroHttpClient,
    config: Arc<Config>,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    params: LaunchOptimizationWorkflowParams,
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationWorkflowParams {
        function_name,
        template_variant_name,
        query_variant_name,
        filters,
        output_source,
        order_by,
        limit,
        offset,
        val_fraction,
        format,
        optimizer_config,
    } = params;
    // Query the database for the stored inferences
    let stored_inferences = clickhouse_connection_info
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: function_name.clone(),
                variant_name: query_variant_name,
                filters,
                output_source,
                limit,
                offset,
                format,
                order_by,
            },
        )
        .await?;
    let variants = HashMap::from([(function_name, template_variant_name)]);
    // Template the inferences and fetch any network resources needed
    let rendered_inferences = render_samples(config.clone(), stored_inferences, variants).await?;

    // Drop any examples with output that is None
    let rendered_inferences = rendered_inferences
        .into_iter()
        .filter(|example| example.output.is_some())
        .collect::<Vec<_>>();

    // Split the inferences into train and val sets
    let (train_examples, val_examples) =
        super::helpers::split_examples(rendered_inferences, val_fraction)?;
    let default_credentials = &config.models.default_credentials;

    // Launch the optimization job
    optimizer_config
        .load(default_credentials)
        .await?
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
            clickhouse_connection_info,
            &config,
        )
        .await
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationParams {
    pub train_samples: Vec<RenderedSample>,
    pub val_samples: Option<Vec<RenderedSample>>,
    pub optimization_config: UninitializedOptimizerInfo,
    // TODO: add a way to do {"type": "tensorzero", "name": "foo"} to grab an optimizer configured in
    // tensorzero.toml
}

/// Launch an optimization job.
/// This function already takes the data as an argument so it gives the caller more control
/// about preparing the data prior to launching the optimization job than the workflow method above.
pub async fn launch_optimization(
    http_client: &TensorzeroHttpClient,
    params: LaunchOptimizationParams,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    config: Arc<Config>,
    // For the TODO above: will need to pass config in here
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationParams {
        train_samples: train_examples,
        val_samples: val_examples,
        optimization_config: optimizer_config,
    } = params;
    let optimizer = optimizer_config
        .load(&config.models.default_credentials)
        .await?;
    optimizer
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
            clickhouse_connection_info,
            &config,
        )
        .await
}

pub async fn poll_optimization_handler(
    State(AppStateData {
        http_client,
        config,
        ..
    }): AppState,
    Path(job_handle): Path<String>,
) -> Result<Response<Body>, Error> {
    let job_handle = OptimizationJobHandle::from_base64_urlencoded(&job_handle)?;
    let default_credentials = &config.models.default_credentials;
    let info = poll_optimization(&http_client, &job_handle, default_credentials).await?;
    Ok(Json(info).into_response())
}

/// Poll an existing optimization job.
/// This should return the status of the job.
pub async fn poll_optimization(
    http_client: &TensorzeroHttpClient,
    job_handle: &OptimizationJobHandle,
    default_credentials: &ProviderTypeDefaultCredentials,
) -> Result<OptimizationJobInfo, Error> {
    job_handle
        .poll(
            http_client,
            &InferenceCredentials::default(),
            default_credentials,
        )
        .await
}
