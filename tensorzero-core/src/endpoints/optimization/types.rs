use serde::{Deserialize, Serialize};

use crate::db::clickhouse::query_builder::ListInferencesParams;
use crate::db::clickhouse::ClickhouseFormat;
use crate::endpoints::datasets::v1::types::ListDatapointsRequest;
use crate::optimization::UninitializedOptimizerInfo;

/// Data source for launch_optimization_workflow: experimental_list_inferences
type ExperimentalListInferencesData = ListInferencesParams;

/// Data source for launch_optimization_workflow: list_datapoints
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, ts(export))]
pub struct ListDatapointsData {
    pub dataset_name: String,
    #[serde(flatten)]
    #[cfg_attr(test, ts(flatten))]
    pub request: ListDatapointsRequest,
}

/// Data source for launch_optimization_workflow
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, ts(export, tag = "type", rename_all = "snake_case"))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizationData {
    #[serde(rename = "experimental_list_inferences")]
    ExperimentalListInferences(ExperimentalListInferencesData),
    #[serde(rename = "list_datapoints")]
    ListDatapoints(ListDatapointsData),
}

/// Parameters for launch_optimization_workflow
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize, Clone)]
#[cfg_attr(test, ts(export))]
pub struct LaunchOptimizationWorkflowParams {
    pub render_variant_name: String,
    pub data: OptimizationData,
    #[serde(default)]
    pub val_fraction: Option<f64>,
    #[serde(default)]
    pub format: ClickhouseFormat,
    pub optimizer_config: UninitializedOptimizerInfo,
}
