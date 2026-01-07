#![recursion_limit = "256"]
#![deny(clippy::all)]
use url::Url;

use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientExt, OptimizationJobHandle, QUANTILES,
};

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
}

#[napi]
pub fn get_quantiles() -> Vec<f64> {
    QUANTILES.to_vec()
}
