use std::sync::Arc;

use rand::seq::SliceRandom;
use serde::Deserialize;
use ts_rs::TS;

use crate::{
    clickhouse::{
        query_builder::{InferenceFilterTreeNode, InferenceOutputSource, ListInferencesParams},
        ClickHouseConnectionInfo, ClickhouseFormat,
    },
    config_parser::Config,
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    optimization::{Optimizer, OptimizerJobHandle, OptimizerStatus, UninitializedOptimizerInfo},
    stored_inference::RenderedStoredInference,
};

#[derive(Debug, Deserialize, TS)]
pub struct StartOptimizationParams {
    pub function_name: String,
    pub template_variant_name: String,
    pub query_variant_name: Option<String>,
    pub filters: Option<InferenceFilterTreeNode>,
    pub output_source: InferenceOutputSource,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    pub val_fraction: Option<f64>,
    #[serde(default)]
    pub format: ClickhouseFormat,
    pub optimizer_config: UninitializedOptimizerInfo,
}

pub async fn start_optimization(
    http_client: &reqwest::Client,
    config: Arc<Config<'static>>,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    params: StartOptimizationParams,
) -> Result<OptimizerJobHandle, Error> {
    let StartOptimizationParams {
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
    let rendered_inferences = stored_inferences

    todo!()
}

#[derive(Debug, Deserialize, TS)]
#[ts(export)]
pub struct LaunchOptimizationParams {
    pub train_examples: Vec<RenderedStoredInference>,
    pub val_examples: Option<Vec<RenderedStoredInference>>,
    pub optimizer_config: UninitializedOptimizerInfo,
    // TODO: add a way to do {"type": "tensorzero", "name": "foo"} to grab an optimizer configured in
    // tensorzero.toml
}

// For the TODO above: will need to pass config in here
pub async fn launch_optimization(
    http_client: &reqwest::Client,
    params: LaunchOptimizationParams,
) -> Result<OptimizerJobHandle, Error> {
    let LaunchOptimizationParams {
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
