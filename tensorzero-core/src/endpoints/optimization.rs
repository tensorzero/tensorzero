use std::{collections::HashMap, sync::Arc};

use axum::{
    body::Body,
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::{
    clickhouse::{
        query_builder::{InferenceFilterTreeNode, InferenceOutputSource, ListInferencesParams},
        ClickHouseConnectionInfo, ClickhouseFormat,
    },
    config_parser::Config,
    endpoints::{inference::InferenceCredentials, stored_inference::render_samples},
    error::{Error, ErrorDetails},
    gateway_util::{AppState, AppStateData, StructuredJson},
    optimization::{
        JobHandle, OptimizationJobHandle, OptimizationJobInfo, Optimizer,
        UninitializedOptimizerInfo,
    },
    serde_util::deserialize_option_u64,
    stored_inference::RenderedSample,
};

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationWorkflowParams {
    pub function_name: String,
    pub template_variant_name: String,
    pub query_variant_name: Option<String>,
    pub filters: Option<InferenceFilterTreeNode>,
    pub output_source: InferenceOutputSource,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub limit: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub offset: Option<u64>,
    pub val_fraction: Option<f64>,
    #[serde(default)]
    pub format: ClickhouseFormat,
    pub optimizer_config: UninitializedOptimizerInfo,
}

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
    http_client: &reqwest::Client,
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
                function_name: &function_name,
                variant_name: query_variant_name.as_deref(),
                filters: filters.as_ref(),
                output_source,
                limit,
                offset,
                format,
            },
        )
        .await?;
    let variants = HashMap::from([(function_name.clone(), template_variant_name.clone())]);
    // Template the inferences and fetch any network resources needed
    let rendered_inferences = render_samples(config, stored_inferences, variants).await?;

    // Drop any examples with output that is None
    let rendered_inferences = rendered_inferences
        .into_iter()
        .filter(|example| example.output.is_some())
        .collect::<Vec<_>>();

    // Split the inferences into train and val sets
    let (train_examples, val_examples) = split_examples(rendered_inferences, val_fraction)?;

    // Launch the optimization job
    optimizer_config
        .load()
        .await?
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
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
    http_client: &reqwest::Client,
    params: LaunchOptimizationParams,
    // For the TODO above: will need to pass config in here
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationParams {
        train_samples: train_examples,
        val_samples: val_examples,
        optimization_config: optimizer_config,
    } = params;
    let optimizer = optimizer_config.load().await?;
    optimizer
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
        )
        .await
}

pub async fn poll_optimization_handler(
    State(AppStateData { http_client, .. }): AppState,
    Path(job_handle): Path<String>,
) -> Result<Response<Body>, Error> {
    let job_handle = OptimizationJobHandle::from_base64_urlencoded(&job_handle)?;
    let info = poll_optimization(&http_client, &job_handle).await?;
    Ok(Json(info).into_response())
}

/// Poll an existing optimization job.
/// This should return the status of the job.
pub async fn poll_optimization(
    http_client: &reqwest::Client,
    job_handle: &OptimizationJobHandle,
) -> Result<OptimizationJobInfo, Error> {
    job_handle
        .poll(http_client, &InferenceCredentials::default())
        .await
}

/// Randomly split examples into train and val sets.
/// Returns a tuple of (train_examples, val_examples).
/// val_examples is None if val_fraction is None.
fn split_examples<T>(
    stored_inferences: Vec<T>,
    val_fraction: Option<f64>,
) -> Result<(Vec<T>, Option<Vec<T>>), Error> {
    if let Some(val_fraction) = val_fraction {
        if val_fraction <= 0.0 || val_fraction >= 1.0 {
            // If val_fraction is not in (0, 1), treat as no split
            return Err(Error::new(ErrorDetails::InvalidValFraction {
                val_fraction,
            }));
        }
        let mut rng = rand::rng();
        let mut examples = stored_inferences;
        let n = examples.len();
        let n_val = ((n as f64) * val_fraction).round() as usize;
        // Shuffle the examples
        examples.as_mut_slice().shuffle(&mut rng);

        // Split examples into val and train sets
        let mut val = Vec::with_capacity(n_val);
        let mut train = Vec::with_capacity(n - n_val);

        // Move elements from examples into val and train
        for (i, example) in examples.into_iter().enumerate() {
            if i < n_val {
                val.push(example);
            } else {
                train.push(example);
            }
        }

        Ok((train, Some(val)))
    } else {
        Ok((stored_inferences, None))
    }
}
