use serde::{Deserialize, Serialize};

use crate::{
    db::{
        clickhouse::query_builder::{InferenceFilter, OrderBy},
        inferences::InferenceOutputSource,
    },
    optimization::UninitializedOptimizerInfo,
    serde_util::deserialize_option_u64,
    stored_inference::RenderedSample,
};

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationWorkflowParams {
    pub function_name: String,
    pub template_variant_name: String,
    pub query_variant_name: Option<String>,
    pub filters: Option<InferenceFilter>,
    pub output_source: InferenceOutputSource,
    pub order_by: Option<Vec<OrderBy>>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub limit: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    pub offset: Option<u64>,
    pub val_fraction: Option<f64>,
    pub optimizer_config: UninitializedOptimizerInfo,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationParams {
    pub train_samples: Vec<RenderedSample>,
    pub val_samples: Option<Vec<RenderedSample>>,
    pub optimization_config: UninitializedOptimizerInfo,
    // TODO: add a way to do {"type": "tensorzero", "name": "foo"} to grab an optimizer configured in
    // tensorzero.toml
}
