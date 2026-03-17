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
    optimization::{
        OptimizationJobHandle, OptimizationJobInfo, UninitializedOptimizerInfo,
        gepa::{GepaGetResponse, GepaLaunchRequest, GepaLaunchResponse, GepaProgress},
    },
    stored_inference::RenderedSample,
    utils::gateway::{AppState, AppStateData, StructuredJson},
};

use crate::gepa::durable::types::{GepaToolOutput, GepaToolParams};
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
#[derive(Debug, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub enum OptimizationDataSource {
    Inferences(InferencesDataSource),
    Dataset(DatasetDataSource),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct InferencesDataSource {
    pub output_source: InferenceOutputSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_variant_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<InferenceFilter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_by: Option<Vec<OrderBy>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DatasetDataSource {
    pub dataset_name: String,
}

impl OptimizationDataSource {
    /// Build an `OptimizationDataSource` from flat fields, validating mutual exclusivity
    /// between inference-query fields and `dataset_name`.
    pub fn from_flat_fields(
        output_source: Option<InferenceOutputSource>,
        dataset_name: Option<String>,
        query_variant_name: Option<String>,
        filters: Option<InferenceFilter>,
        order_by: Option<Vec<OrderBy>>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Self, String> {
        match (output_source, dataset_name) {
            (Some(output_source), None) => {
                Ok(OptimizationDataSource::Inferences(InferencesDataSource {
                    output_source,
                    query_variant_name,
                    filters,
                    order_by,
                    limit,
                    offset,
                }))
            }
            (None, Some(dataset_name)) => {
                let inference_fields = [
                    query_variant_name.as_ref().map(|_| "query_variant_name"),
                    filters.as_ref().map(|_| "filters"),
                    order_by.as_ref().map(|_| "order_by"),
                    limit.map(|_| "limit"),
                    offset.map(|_| "offset"),
                ];
                let present: Vec<&str> = inference_fields.into_iter().flatten().collect();
                if !present.is_empty() {
                    return Err(format!(
                        "inference-specific fields [{}] cannot be used with `dataset_name`",
                        present.join(", ")
                    ));
                }
                Ok(OptimizationDataSource::Dataset(DatasetDataSource {
                    dataset_name,
                }))
            }
            (Some(_), Some(_)) => {
                Err("provide either `output_source` or `dataset_name`, not both".to_string())
            }
            (None, None) => {
                Err("you must provide either `output_source` or `dataset_name`".to_string())
            }
        }
    }
}

/// Custom `Deserialize` that rejects payloads containing both `output_source` and `dataset_name`.
impl<'de> Deserialize<'de> for OptimizationDataSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            output_source: Option<InferenceOutputSource>,
            dataset_name: Option<String>,
            query_variant_name: Option<String>,
            filters: Option<InferenceFilter>,
            order_by: Option<Vec<OrderBy>>,
            limit: Option<u32>,
            offset: Option<u32>,
        }

        let helper = Helper::deserialize(deserializer)?;

        OptimizationDataSource::from_flat_fields(
            helper.output_source,
            helper.dataset_name,
            helper.query_variant_name,
            helper.filters,
            helper.order_by,
            helper.limit,
            helper.offset,
        )
        .map_err(serde::de::Error::custom)
    }
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
        OptimizationDataSource::Inferences(inferences) => {
            let stored_inferences = db
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: Some(&function_name),
                        ids: None,
                        variant_name: inferences.query_variant_name.as_deref(),
                        episode_id: None,
                        filters: inferences.filters.as_ref(),
                        output_source: inferences.output_source,
                        limit: inferences
                            .limit
                            .unwrap_or(DEFAULT_LIST_INFERENCES_QUERY_LIMIT_MAX_FOR_OPTIMIZATIONS),
                        offset: inferences.offset.unwrap_or(0),
                        pagination: None,
                        order_by: inferences.order_by.as_deref(),
                        search_query_experimental: None,
                    },
                )
                .await?;
            render_samples(config.clone(), stored_inferences, variants, None).await?
        }
        OptimizationDataSource::Dataset(dataset) => {
            let stored_datapoints = db
                .get_datapoints(&GetDatapointsParams {
                    dataset_name: Some(dataset.dataset_name),
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

pub async fn gepa_launch_handler(
    State(app_state): AppState,
    StructuredJson(req): StructuredJson<GepaLaunchRequest>,
) -> Result<Response<Body>, Error> {
    let spawn_client = app_state.spawn_client.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: "GEPA requires Postgres and a durable task queue to be configured".to_string(),
        })
    })?;

    // Validate mutually exclusive fields before spawning
    req.dataset().map_err(|msg| {
        Error::new(ErrorDetails::InvalidRequest {
            message: msg.to_string(),
        })
    })?;
    req.evaluation().map_err(|msg| {
        Error::new(ErrorDetails::InvalidRequest {
            message: msg.to_string(),
        })
    })?;

    // Inline evaluators mode is not yet supported — require evaluation_name
    if req.evaluation_name.is_none() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Inline `evaluators` mode is not yet supported; provide `evaluation_name`"
                .to_string(),
        }));
    }

    // Convert GepaLaunchRequest → GepaToolParams
    let tool_params = GepaToolParams {
        function_name: req.function_name,
        dataset_name: req.dataset_name,
        train_dataset_name: req.train_dataset_name,
        val_dataset_name: req.val_dataset_name,
        evaluation_name: req.evaluation_name,
        evaluators: req.evaluators,
        analysis_model: req.analysis_model,
        mutation_model: req.mutation_model,
        initial_variants: req.initial_variants,
        max_iterations: req.max_iterations,
        variant_prefix: req.variant_prefix,
        batch_size: req.batch_size,
        seed: req.seed,
        include_inference_for_mutation: req.include_inference_for_mutation,
        max_concurrency: req.max_concurrency,
        max_datapoints: req.max_datapoints,
    };

    let llm_params = serde_json::to_value(&tool_params).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize GEPA params: {e}"),
        })
    })?;

    let episode_id = uuid::Uuid::now_v7();

    let spawn_result = spawn_client
        .spawn_tool_by_name(
            "standalone_gepa",
            llm_params,
            serde_json::json!(null),
            episode_id,
            durable_tools_spawn::SpawnOptions::default(),
        )
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to spawn GEPA task: {e}"),
            })
        })?;

    Ok(Json(GepaLaunchResponse {
        task_id: spawn_result.task_id.to_string(),
    })
    .into_response())
}

pub async fn gepa_get_handler(
    State(app_state): AppState,
    Path(task_id): Path<String>,
) -> Result<Response<Body>, Error> {
    let spawn_client = app_state.spawn_client.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: "GEPA requires Postgres and a durable task queue to be configured".to_string(),
        })
    })?;

    let task_id: uuid::Uuid = task_id.parse().map_err(|_| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Invalid task_id: `{task_id}` is not a valid UUID"),
        })
    })?;

    let poll_result = spawn_client.get_task_result(task_id).await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to poll GEPA task: {e}"),
        })
    })?;

    use durable_tools_spawn::TaskStatus;

    let response = match poll_result.status {
        TaskStatus::Completed => {
            if let Some(result_value) = poll_result.result {
                let output: GepaToolOutput = serde_json::from_value(result_value).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize GEPA result: {e}"),
                    })
                })?;
                GepaGetResponse::Completed {
                    variants: output.variants,
                    statistics: output.statistics,
                }
            } else {
                GepaGetResponse::Error {
                    error: "Task completed but no result payload found".to_string(),
                }
            }
        }
        TaskStatus::Failed | TaskStatus::Cancelled => {
            let error = poll_result
                .error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "Unknown error".to_string());
            GepaGetResponse::Error { error }
        }
        TaskStatus::Pending | TaskStatus::Running | TaskStatus::Sleeping => {
            let max_iterations = poll_result
                .params
                .as_ref()
                .and_then(|p| p.get("llm_params"))
                .and_then(|p| p.get("max_iterations"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0);

            let progress = match spawn_client.get_task_progress(task_id).await {
                Ok(Some(checkpoint_name)) => parse_gepa_progress(&checkpoint_name, max_iterations),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("Failed to get GEPA progress: {e}");
                    None
                }
            };
            GepaGetResponse::Pending { progress }
        }
    };

    Ok(Json(response).into_response())
}

fn parse_gepa_progress(checkpoint_name: &str, max_iterations: u32) -> Option<GepaProgress> {
    match checkpoint_name {
        "setup" => {
            return Some(GepaProgress {
                current_iteration: 0,
                max_iterations,
                current_step: "setup".to_string(),
            });
        }
        "init_eval" => {
            return Some(GepaProgress {
                current_iteration: 0,
                max_iterations,
                current_step: "init_eval".to_string(),
            });
        }
        _ => {}
    }

    // Iteration steps: "iter_{n}_{step}" where step can contain underscores
    // e.g. "iter_3_eval_analyze_mutate" → iteration 3, step "eval_analyze_mutate"
    if let Some(rest) = checkpoint_name.strip_prefix("iter_")
        && let Some((n_str, step)) = rest.split_once('_')
        && let Ok(iteration) = n_str.parse::<u32>()
    {
        return Some(GepaProgress {
            current_iteration: iteration,
            max_iterations,
            current_step: step.to_string(),
        });
    }

    None
}

/// Randomly split examples into train and val sets.
/// Returns a tuple of (train_examples, val_examples).
/// val_examples is None if val_fraction is None.
fn split_examples<T>(
    mut examples: Vec<T>,
    val_fraction: Option<f64>,
) -> Result<(Vec<T>, Option<Vec<T>>), Error> {
    let Some(val_fraction) = val_fraction else {
        return Ok((examples, None));
    };

    if val_fraction <= 0.0 || val_fraction >= 1.0 {
        return Err(Error::new(ErrorDetails::InvalidValFraction {
            val_fraction,
        }));
    }

    let n_val = ((examples.len() as f64) * val_fraction).round() as usize;
    examples.as_mut_slice().shuffle(&mut rand::rng());

    // split_off takes everything from n_val onwards, leaving 0..n_val in `examples`
    let train = examples.split_off(n_val);
    let val = examples;

    Ok((train, Some(val)))
}
