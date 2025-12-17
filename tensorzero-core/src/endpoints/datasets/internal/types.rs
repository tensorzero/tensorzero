use serde::{Deserialize, Serialize};

use crate::{
    db::datasets::{DatasetOutputSource, MetricFilter},
    endpoints::datasets::DatapointKind,
};

/// Request body for counting and inserting matching inferences into a dataset
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct FilterInferencesForDatasetBuilderRequest {
    /// The type of inference to filter for (chat or json)
    pub inference_type: DatapointKind,
    /// Optional function name to filter by
    pub function_name: Option<String>,
    /// Optional variant name to filter by (requires function_name)
    pub variant_name: Option<String>,
    /// How to handle the output field when matching
    pub output_source: DatasetOutputSource,
    /// Optional metric filter to apply
    pub metric_filter: Option<MetricFilter>,
}
