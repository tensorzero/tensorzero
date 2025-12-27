#![recursion_limit = "256"]
#![deny(clippy::all)]
use tensorzero_core::endpoints::datasets::StaleDatasetResponse;
use url::Url;

use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientExt, ClientInferenceParams, InferenceOutput,
    OptimizationJobHandle, QUANTILES,
};

#[macro_use]
mod napi_bridge;
mod postgres;

#[macro_use]
extern crate napi_derive;

#[napi(js_name = "TensorZeroClient")]
pub struct TensorZeroClient {
    client: Client,
}

#[napi]
impl TensorZeroClient {
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
            return Err(napi::Error::from_reason(
                "Streaming inference is not supported. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
            ));
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
pub fn get_quantiles() -> Vec<f64> {
    QUANTILES.to_vec()
}
