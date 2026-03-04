//! HTTP endpoints for optimizer operations
//!
//! These endpoints handle launching and polling optimization jobs.

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    response::{IntoResponse, Response},
};
use rand::seq::SliceRandom;

use tensorzero_core::{
    config::{Config, provider_types::ProviderTypesConfig},
    db::{
        clickhouse::query_builder::{InferenceFilter, OrderBy},
        datasets::GetDatapointsParams,
        delegating_connection::DelegatingDatabaseQueries,
        inferences::{InferenceOutputSource, ListInferencesParams},
    },
    endpoints::{inference::InferenceCredentials, stored_inferences::render_samples},
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{OptimizationJobHandle, OptimizationJobInfo, UninitializedOptimizerInfo},
    stored_inference::RenderedSample,
    utils::gateway::{AppState, AppStateData, StructuredJson},
};

use crate::{JobHandle, Optimizer};

// TODO(shuyangli): revisit this and see if it should be u32::MAX.
const DEFAULT_LIST_INFERENCES_QUERY_LIMIT_MAX_FOR_OPTIMIZATIONS: u32 = u32::MAX;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct LaunchOptimizationWorkflowParams {
    pub function_name: String,
    pub template_variant_name: String,
    #[serde(flatten)]
    pub data_source: OptimizationDataSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_fraction: Option<f64>,
    pub optimizer_config: UninitializedOptimizerInfo,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub enum OptimizationDataSource {
    Inferences {
        output_source: InferenceOutputSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        query_variant_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<InferenceFilter>,
        #[serde(skip_serializing_if = "Option::is_none")]
        order_by: Option<Vec<OrderBy>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        offset: Option<u32>,
    },
    Dataset {
        dataset_name: String,
    },
}

pub async fn launch_optimization_workflow_handler(
    State(app_state): AppState,
    StructuredJson(params): StructuredJson<LaunchOptimizationWorkflowParams>,
) -> Result<Response<Body>, Error> {
    let db: Arc<dyn DelegatingDatabaseQueries + Send + Sync> =
        Arc::new(app_state.get_delegating_database());
    let job_handle =
        launch_optimization_workflow(&app_state.http_client, app_state.config, &db, params).await?;
    let encoded_job_handle = job_handle.to_base64_urlencoded()?;
    Ok(encoded_job_handle.into_response())
}

/// Starts an optimization job.
/// This function will query inferences or datapoints from the database,
/// render them by fetching any network resources needed and
/// templating them with the template variant,
/// and launch the optimization job specified.
pub async fn launch_optimization_workflow(
    http_client: &TensorzeroHttpClient,
    config: Arc<Config>,
    db: &Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    params: LaunchOptimizationWorkflowParams,
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationWorkflowParams {
        function_name,
        template_variant_name,
        data_source,
        val_fraction,
        optimizer_config,
    } = params;

    let variants = HashMap::from([(function_name.clone(), template_variant_name.clone())]);

    let rendered_samples = match data_source {
        OptimizationDataSource::Inferences {
            output_source,
            query_variant_name,
            filters,
            order_by,
            limit,
            offset,
        } => {
            let stored_inferences = db
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: Some(&function_name),
                        ids: None,
                        variant_name: query_variant_name.as_deref(),
                        episode_id: None,
                        filters: filters.as_ref(),
                        output_source,
                        limit: limit
                            .unwrap_or(DEFAULT_LIST_INFERENCES_QUERY_LIMIT_MAX_FOR_OPTIMIZATIONS),
                        offset: offset.unwrap_or(0),
                        pagination: None,
                        order_by: order_by.as_deref(),
                        search_query_experimental: None,
                    },
                )
                .await?;
            render_samples(config.clone(), stored_inferences, variants, None).await?
        }
        OptimizationDataSource::Dataset { dataset_name } => {
            let stored_datapoints = db
                .get_datapoints(&GetDatapointsParams {
                    dataset_name: Some(dataset_name),
                    function_name: Some(function_name.clone()),
                    ids: None,
                    limit: u32::MAX,
                    offset: 0,
                    allow_stale: false,
                    filter: None,
                    order_by: None,
                    search_query_experimental: None,
                })
                .await?;
            render_samples(config.clone(), stored_datapoints, variants, None).await?
        }
    };

    // Drop any examples with output that is None
    let rendered_samples = rendered_samples
        .into_iter()
        .filter(|example| example.output.is_some())
        .collect::<Vec<_>>();

    // Split the samples into train and val sets
    let (train_examples, val_examples) = split_examples(rendered_samples, val_fraction)?;

    // Launch the optimization job
    optimizer_config
        .load()
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
            db,
            config.clone(),
        )
        .await
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    db: Arc<dyn DelegatingDatabaseQueries + Send + Sync>,
    config: Arc<Config>,
    // For the TODO above: will need to pass config in here
) -> Result<OptimizationJobHandle, Error> {
    let LaunchOptimizationParams {
        train_samples: train_examples,
        val_samples: val_examples,
        optimization_config: optimizer_config,
    } = params;
    let optimizer = optimizer_config.load();
    optimizer
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
            &db,
            config.clone(),
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
    let info = poll_optimization(
        &http_client,
        &job_handle,
        default_credentials,
        &config.provider_types,
    )
    .await?;
    Ok(Json(info).into_response())
}

/// Poll an existing optimization job.
/// This should return the status of the job.
pub async fn poll_optimization(
    http_client: &TensorzeroHttpClient,
    job_handle: &OptimizationJobHandle,
    default_credentials: &ProviderTypeDefaultCredentials,
    provider_types: &ProviderTypesConfig,
) -> Result<OptimizationJobInfo, Error> {
    job_handle
        .poll(
            http_client,
            &InferenceCredentials::default(),
            default_credentials,
            provider_types,
        )
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
