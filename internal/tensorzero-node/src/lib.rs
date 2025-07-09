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

#[napi]
pub async fn get_config(config_path: String) -> Result<String, napi::Error> {
    let config = tensorzero::get_config(Path::new(&config_path).to_path_buf())
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to get config: {e}")))?;
    let config_str =
        serde_json::to_string(&config).map_err(|e| napi::Error::from_reason(e.to_string()))?;
    Ok(config_str)
}
