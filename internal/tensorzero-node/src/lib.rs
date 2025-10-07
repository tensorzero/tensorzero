#![deny(clippy::all)]
use std::{path::Path, time::Duration};
use url::Url;

use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, InferenceOutput,
    OptimizationJobHandle, QUANTILES,
};

#[macro_use]
mod napi_bridge;
mod database;

#[macro_use]
extern crate napi_derive;

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
            .stale_dataset(dataset_name)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result_str = serde_json::to_string(&result).map_err(|e| {
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
