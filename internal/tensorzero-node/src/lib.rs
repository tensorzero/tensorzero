#![deny(clippy::all)]
use std::{path::Path, time::Duration};

use tensorzero::{Client, ClientBuilder, ClientBuilderMode, OptimizationJobHandle};

#[macro_use]
extern crate napi_derive;

#[napi(js_name = "TensorZeroClient")]
pub struct TensorZeroClient {
    client: Client,
}

#[napi]
impl TensorZeroClient {
    #[napi(factory)]
    pub async fn build(
        config_path: String,
        clickhouse_url: Option<String>,
        timeout: Option<f64>,
    ) -> Result<Self, napi::Error> {
        let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(Path::new(&config_path).to_path_buf()),
            clickhouse_url,
            timeout: timeout.map(Duration::from_secs_f64),
            verify_credentials: false,
        })
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
            .experimental_poll_optimization(job_handle)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let info_str =
            serde_json::to_string(&info).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(info_str)
    }

    #[napi]
    pub fn get_function_config(&self, function_name: String) -> Result<String, napi::Error> {
        let function_config = self
            .client
            .get_function_config(&function_name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let function_config_str = serde_json::to_string(&function_config)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(function_config_str)
    }

    #[napi]
    pub fn list_functions(&self) -> Result<Vec<String>, napi::Error> {
        let functions = self
            .client
            .list_functions()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(functions.into_iter().map(|s| s.to_string()).collect())
    }

    #[napi]
    pub fn get_metric_config(&self, metric_name: String) -> Result<String, napi::Error> {
        let metric_config = self
            .client
            .get_metric_config(&metric_name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let metric_config_str = serde_json::to_string(&metric_config)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(metric_config_str)
    }

    #[napi]
    pub fn list_metrics(&self) -> Result<Vec<String>, napi::Error> {
        let metrics = self
            .client
            .list_metrics()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(metrics.into_iter().map(|s| s.to_string()).collect())
    }

    #[napi]
    pub fn list_evaluations(&self) -> Result<Vec<String>, napi::Error> {
        let evaluations = self
            .client
            .list_evaluations()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(evaluations.into_iter().map(|s| s.to_string()).collect())
    }

    #[napi]
    pub fn get_evaluation_config(&self, evaluation_name: String) -> Result<String, napi::Error> {
        let evaluation_config = self
            .client
            .get_evaluation_config(&evaluation_name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let evaluation_config_str = serde_json::to_string(&evaluation_config)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(evaluation_config_str)
    }

    #[napi]
    pub fn get_config(&self) -> Result<String, napi::Error> {
        let config = self
            .client
            .get_config()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let config_str =
            serde_json::to_string(&config).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(config_str)
    }

    #[napi]
    pub async fn stale_dataset(&self, dataset_name: String) -> Result<String, napi::Error> {
        let result = self
            .client
            .stale_dataset(dataset_name)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result_str = serde_json::to_string(&result).map_err(|e| {
            napi::Error::from_reason(format!("Failed to serialize stale dataset result: {e}"))
        })?;
        Ok(result_str)
    }
}
