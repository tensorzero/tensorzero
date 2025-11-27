#![recursion_limit = "256"]
#![deny(clippy::all)]
use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};
use tensorzero_core::config::ConfigLoadInfo;
use tensorzero_core::endpoints::datasets::StaleDatasetResponse;
use url::Url;

use evaluations::stats::{EvaluationInfo, EvaluationUpdate};
use evaluations::{run_evaluation_core_streaming, EvaluationCoreArgs, EvaluationVariant};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use serde::Serialize;
use serde_json::Value;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientExt, ClientInferenceParams, InferenceOutput,
    OptimizationJobHandle, QUANTILES,
};
use tensorzero_core::{
    cache::CacheEnabledMode,
    config::{Config, ConfigFileGlob},
    db::clickhouse::ClickHouseConnectionInfo,
};
use uuid::Uuid;

#[macro_use]
mod napi_bridge;
mod database;
mod postgres;

#[macro_use]
extern crate napi_derive;

#[derive(Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluationRunEvent {
    Start(EvaluationRunStartEvent),
    Success(EvaluationRunSuccessEvent),
    Error(EvaluationRunErrorEvent),
    FatalError(EvaluationRunFatalErrorEvent),
    Complete(EvaluationRunCompleteEvent),
}

#[derive(Serialize, ts_rs::TS)]
pub struct EvaluationRunStartEvent {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
    pub evaluation_name: String,
    pub dataset_name: Option<String>,
    pub variant_name: String,
}

#[derive(Serialize, ts_rs::TS)]
pub struct EvaluationRunSuccessEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint: Value,
    pub response: Value,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

#[derive(Serialize, ts_rs::TS)]
pub struct EvaluationRunErrorEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint_id: Uuid,
    pub message: String,
}

#[derive(Serialize, ts_rs::TS)]
pub struct EvaluationRunFatalErrorEvent {
    pub evaluation_run_id: Option<Uuid>,
    pub message: String,
}

#[derive(Serialize, ts_rs::TS)]
pub struct EvaluationRunCompleteEvent {
    pub evaluation_run_id: Uuid,
}

impl EvaluationRunSuccessEvent {
    fn try_from_info(evaluation_run_id: Uuid, info: EvaluationInfo) -> Result<Self, napi::Error> {
        let datapoint = serde_json::to_value(info.datapoint)
            .map_err(|e| napi::Error::from_reason(format!("Failed to serialize datapoint: {e}")))?;
        let response = serde_json::to_value(info.response).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize inference response: {e}"))
        })?;

        Ok(Self {
            evaluation_run_id,
            datapoint,
            response,
            evaluations: info.evaluations,
            evaluator_errors: info.evaluator_errors,
        })
    }
}

fn send_event(
    callback: &ThreadsafeFunction<String>,
    event: &EvaluationRunEvent,
) -> Result<(), napi::Error> {
    let payload = serde_json::to_string(event)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize event: {e}")))?;
    let status = callback.call(Ok(payload), ThreadsafeFunctionCallMode::NonBlocking);
    if status == napi::Status::Ok {
        Ok(())
    } else {
        Err(napi::Error::from_status(status))
    }
}

#[napi(object)]
pub struct RunEvaluationStreamingParams {
    pub gateway_url: String,
    pub clickhouse_url: String,
    pub config_path: String,
    pub evaluation_name: String,
    pub dataset_name: Option<String>,
    pub datapoint_ids: Option<Vec<String>>,
    pub variant_name: String,
    pub concurrency: u32,
    pub inference_cache: String,
    pub max_datapoints: Option<u32>,
    /// JSON string mapping evaluator names to precision limit thresholds.
    /// Example: '{"exact_match": 0.13, "llm_judge": 0.16}'
    pub precision_targets: Option<String>,
}

#[napi]
pub async fn run_evaluation_streaming(
    params: RunEvaluationStreamingParams,
    callback: ThreadsafeFunction<String>,
) -> Result<(), napi::Error> {
    let url = Url::parse(&params.gateway_url)
        .map_err(|e| napi::Error::from_reason(format!("Invalid gateway URL: {e}")))?;

    let config_glob =
        ConfigFileGlob::new_from_path(Path::new(&params.config_path)).map_err(|e| {
            napi::Error::from_reason(format!(
                "Failed to resolve config glob from {}: {e}",
                params.config_path
            ))
        })?;

    let ConfigLoadInfo {
        config,
        snapshot: _,
    } = Config::load_from_path_optional_verify_credentials(&config_glob, false)
        .await
        .map_err(|e| {
            napi::Error::from_reason(format!(
                "Failed to load configuration from {}: {e}",
                params.config_path
            ))
        })?;
    let config = Arc::new(config);

    let tensorzero_client = ClientBuilder::new(ClientBuilderMode::HTTPGateway { url })
        .build()
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to build TensorZero client: {e}")))?;

    let clickhouse_client = ClickHouseConnectionInfo::new(
        &params.clickhouse_url,
        config.gateway.observability.batch_writes.clone(),
    )
    .await
    .map_err(|e| napi::Error::from_reason(format!("Failed to connect to ClickHouse: {e}")))?;

    let cache_mode = match params.inference_cache.as_str() {
        "on" => CacheEnabledMode::On,
        "off" => CacheEnabledMode::Off,
        "read_only" => CacheEnabledMode::ReadOnly,
        "write_only" => CacheEnabledMode::WriteOnly,
        other => {
            let _ = callback.abort();
            return Err(napi::Error::from_reason(format!(
                "Invalid inference cache setting '{other}'"
            )));
        }
    };

    let concurrency = usize::try_from(params.concurrency).map_err(|_| {
        napi::Error::from_reason(format!(
            "Concurrency {} is larger than supported on this platform",
            params.concurrency
        ))
    })?;

    let datapoint_ids: Vec<Uuid> = params
        .datapoint_ids
        .unwrap_or_default()
        .iter()
        .map(|s| {
            Uuid::parse_str(s).map_err(|e| {
                napi::Error::from_reason(format!("Invalid UUID in datapoint_ids: {e}"))
            })
        })
        .collect::<Result<Vec<Uuid>, napi::Error>>()?;

    let evaluation_run_id = Uuid::now_v7();

    // Parse precision_targets from JSON string to HashMap
    let precision_targets = if let Some(limits_json_str) = params.precision_targets {
        let limits_map: std::collections::HashMap<String, f64> =
            serde_json::from_str(&limits_json_str).map_err(|e| {
                napi::Error::from_reason(format!("Invalid precision_targets JSON: {e}"))
            })?;
        // Convert f64 to f32
        limits_map.into_iter().map(|(k, v)| (k, v as f32)).collect()
    } else {
        HashMap::new()
    };

    let core_args = EvaluationCoreArgs {
        tensorzero_client,
        clickhouse_client: clickhouse_client.clone(),
        config: config.clone(),
        dataset_name: params.dataset_name.clone(),
        datapoint_ids: Some(datapoint_ids.clone()),
        variant: EvaluationVariant::Name(params.variant_name.clone()),
        evaluation_name: params.evaluation_name.clone(),
        evaluation_run_id,
        inference_cache: cache_mode,
        concurrency,
    };

    let result =
        match run_evaluation_core_streaming(core_args, params.max_datapoints, precision_targets)
            .await
        {
            Ok(result) => result,
            Err(error) => {
                let _ = callback.abort();
                return Err(napi::Error::from_reason(format!(
                    "Failed to start evaluation run: {error}"
                )));
            }
        };

    let start_event = EvaluationRunEvent::Start(EvaluationRunStartEvent {
        evaluation_run_id,
        num_datapoints: result.run_info.num_datapoints,
        evaluation_name: params.evaluation_name.clone(),
        dataset_name: params.dataset_name.clone(),
        variant_name: params.variant_name.clone(),
    });

    send_event(&callback, &start_event)?;

    let mut receiver = result.receiver;

    while let Some(update) = receiver.recv().await {
        let event = match update {
            EvaluationUpdate::RunInfo(_) => continue,
            EvaluationUpdate::Success(info) => EvaluationRunEvent::Success(
                EvaluationRunSuccessEvent::try_from_info(evaluation_run_id, info)?,
            ),
            EvaluationUpdate::Error(error) => EvaluationRunEvent::Error(EvaluationRunErrorEvent {
                evaluation_run_id,
                datapoint_id: error.datapoint_id,
                message: error.message,
            }),
        };

        send_event(&callback, &event)?;
    }

    let join_handle = clickhouse_client.batcher_join_handle();
    drop(clickhouse_client);

    if let Some(handle) = join_handle {
        if let Err(error) = handle.await {
            let fatal_event = EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                evaluation_run_id: Some(evaluation_run_id),
                message: format!(
                    "Error waiting for evaluations ClickHouse batch writer to finish: {error}"
                ),
            });
            let _ = send_event(&callback, &fatal_event);
            let _ = callback.abort();
            return Err(napi::Error::from_reason(format!(
                "Error waiting for evaluations ClickHouse batch writer to finish: {error}"
            )));
        }
    }

    let complete_event =
        EvaluationRunEvent::Complete(EvaluationRunCompleteEvent { evaluation_run_id });
    send_event(&callback, &complete_event)?;

    let _ = callback.abort();

    Ok(())
}

#[napi(js_name = "TensorZeroClient")]
pub struct TensorZeroClient {
    client: Client,
}

#[napi]
impl TensorZeroClient {
    #[napi(factory)]
    pub async fn build_embedded(
        config_path: String,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        timeout: Option<f64>,
    ) -> Result<Self, napi::Error> {
        let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(Path::new(&config_path).to_path_buf()),
            clickhouse_url,
            postgres_url,
            timeout: timeout.map(Duration::from_secs_f64),
            verify_credentials: false,
            allow_batch_writes: false,
        })
        .build()
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { client })
    }

    #[napi(factory)]
    pub async fn build_http(gateway_url: String) -> Result<Self, napi::Error> {
        let url = Url::parse(&gateway_url).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway { url })
            .build()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { client })
    }

    #[napi]
    pub async fn experimental_launch_optimization_workflow(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        let params: tensorzero::LaunchOptimizationWorkflowParams =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let job_handle = self
            .client
            .experimental_launch_optimization_workflow(params)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let job_handle_str = serde_json::to_string(&job_handle).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize job handle: {e}"))
        })?;
        Ok(job_handle_str)
    }

    #[napi]
    pub async fn experimental_poll_optimization(
        &self,
        job_handle: String,
    ) -> Result<String, napi::Error> {
        let job_handle: OptimizationJobHandle = serde_json::from_str(&job_handle)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let info = self
            .client
            .experimental_poll_optimization(&job_handle)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let info_str =
            serde_json::to_string(&info).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(info_str)
    }

    #[napi]
    pub async fn stale_dataset(&self, dataset_name: String) -> Result<String, napi::Error> {
        let result = self
            .client
            .delete_dataset(dataset_name)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let shim_result = StaleDatasetResponse {
            num_staled_datapoints: result.num_deleted_datapoints,
        };
        let result_str = serde_json::to_string(&shim_result).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize stale dataset result: {e}"))
        })?;
        Ok(result_str)
    }

    #[napi]
    pub async fn inference(&self, params: String) -> Result<String, napi::Error> {
        let params: ClientInferenceParams =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        if params.stream.unwrap_or(false) {
            return Err(napi::Error::from_reason(
                "Streaming inference is not supported",
            ));
        }
        let result = self
            .client
            .inference(params)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let InferenceOutput::NonStreaming(result) = result else {
            return Err(napi::Error::from_reason("Streaming inference is not supported. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports"));
        };
        let result_str = serde_json::to_string(&result).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize inference result: {e}"))
        })?;
        Ok(result_str)
    }

    #[napi]
    pub async fn get_variant_sampling_probabilities(
        &self,
        function_name: String,
    ) -> Result<String, napi::Error> {
        let probabilities = self
            .client
            .get_variant_sampling_probabilities(&function_name)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let probabilities_str = serde_json::to_string(&probabilities).map_err(|e| {
            napi::Error::from_reason(format!(
                "Failed to serialize variant sampling probabilities: {e}"
            ))
        })?;
        Ok(probabilities_str)
    }
}

#[napi]
pub async fn get_config(config_path: Option<String>) -> Result<String, napi::Error> {
    let config_path = config_path
        .as_ref()
        .map(Path::new)
        .map(|path| path.to_path_buf());
    let config = tensorzero::get_config_no_verify_credentials(config_path)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to get config: {e}")))?;
    let config_str =
        serde_json::to_string(&config).map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(config_str)
}

#[napi]
pub fn get_quantiles() -> Vec<f64> {
    QUANTILES.to_vec()
}
