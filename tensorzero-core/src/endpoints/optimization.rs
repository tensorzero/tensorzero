use std::{collections::HashMap, sync::Arc};

use rand::seq::SliceRandom;
use serde::Deserialize;

use crate::{
    clickhouse::{
        query_builder::{InferenceFilterTreeNode, InferenceOutputSource, ListInferencesParams},
        ClickHouseConnectionInfo, ClickhouseFormat,
    },
    config_parser::Config,
    endpoints::{inference::InferenceCredentials, stored_inference::render_inferences},
    error::{Error, ErrorDetails},
    optimization::{Optimizer, OptimizerJobHandle, OptimizerStatus, UninitializedOptimizerInfo},
    serde_util::deserialize_option_u64,
    stored_inference::RenderedStoredInference,
};

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationWorkflowParams {
    pub function_name: String,
    pub template_variant_name: String,
    pub query_variant_name: Option<String>,
    pub filters: Option<InferenceFilterTreeNode>,
    pub output_source: InferenceOutputSource,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub limit: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub offset: Option<u64>,
    pub val_fraction: Option<f64>,
    #[serde(default)]
    pub format: ClickhouseFormat,
    pub optimizer_config: UninitializedOptimizerInfo,
}

/// Starts an optimization job.
/// This function will query inferences from the database,
/// render them by fetching any network resources needed and
/// templating them with the template variant,
/// and launch the optimization job specified.
pub async fn launch_optimization_workflow(
    http_client: &reqwest::Client,
    config: Arc<Config<'static>>,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    params: LaunchOptimizationWorkflowParams,
) -> Result<OptimizerJobHandle, Error> {
    let LaunchOptimizationWorkflowParams {
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
    // Query the database for the stored inferences
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
    let variants = HashMap::from([(function_name.clone(), template_variant_name.clone())]);
    // Template the inferences and fetch any network resources needed
    let rendered_inferences = render_inferences(config, stored_inferences, variants).await?;

    // Split the inferences into train and val sets
    let (train_examples, val_examples) = split_examples(rendered_inferences, val_fraction)?;

    // Launch the optimization job
    optimizer_config
        .load()?
        .launch(
            http_client,
            train_examples,
            val_examples,
            &InferenceCredentials::default(),
        )
        .await
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationParams {
    pub train_examples: Vec<RenderedStoredInference>,
    pub val_examples: Option<Vec<RenderedStoredInference>>,
    pub optimizer_config: UninitializedOptimizerInfo,
    // TODO: add a way to do {"type": "tensorzero", "name": "foo"} to grab an optimizer configured in
    // tensorzero.toml
}

/// Launch an optimization job.
/// This function already takes the data as an argument so it gives the caller more control
/// about preparing the data prior to launching the optimization job than the workflow method above.
pub async fn launch_optimization(
    http_client: &reqwest::Client,
    params: LaunchOptimizationParams,
    // For the TODO above: will need to pass config in here
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

/// Poll an existing optimization job.
/// This should return the status of the job.
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
