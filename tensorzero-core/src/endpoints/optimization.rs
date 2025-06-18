use serde::Deserialize;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::Error,
    optimization::{Optimizer, OptimizerJobHandle, OptimizerStatus, UninitializedOptimizerInfo},
    stored_inference::RenderedStoredInference,
};

#[derive(Debug, Deserialize)]
pub struct Params {
    pub train_examples: Vec<RenderedStoredInference>,
    pub val_examples: Option<Vec<RenderedStoredInference>>,
    pub optimizer_config: UninitializedOptimizerInfo,
    // TODO: add a way to do {"type": "tensorzero", "name": "foo"} to grab an optimizer configured in
    // tensorzero.toml
}

// For the TODO above: will need to pass config in here
pub async fn launch_optimization(
    http_client: &reqwest::Client,
    params: Params,
) -> Result<OptimizerJobHandle, Error> {
    let Params {
        train_examples,
        val_examples,
        optimizer_config,
    } = params;
    let optimizer = optimizer_config.load()?;
    optimizer
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
        )
        .await
}

pub async fn poll_optimization(
    http_client: &reqwest::Client,
    job_handle: &OptimizerJobHandle,
) -> Result<OptimizerStatus, Error> {
    let optimizer = UninitializedOptimizerInfo::load_from_default_optimizer(job_handle)?;
    optimizer
        .poll(http_client, job_handle, &InferenceCredentials::default())
        .await
}
