//! Feedback endpoint for querying feedback statistics by variant.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::db::feedback::{FeedbackByVariant, FeedbackQueries};
use crate::error::Error;

/// Combined request for getting feedback statistics by variant.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GetFeedbackByVariantToolParams {
    /// The name of the metric to query.
    pub metric_name: String,
    /// The name of the function to query.
    pub function_name: String,
    /// Optional filter for specific variants. If not provided, all variants are included.
    #[serde(default)]
    pub variant_names: Option<Vec<String>>,
}

/// Response containing feedback statistics by variant.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetFeedbackByVariantResponse {
    pub variants: Vec<FeedbackByVariant>,
}

/// Core business logic for getting feedback statistics by variant.
pub async fn get_feedback_by_variant(
    database: &(dyn FeedbackQueries + Sync),
    metric_name: &str,
    function_name: &str,
    variant_names: Option<&Vec<String>>,
) -> Result<GetFeedbackByVariantResponse, Error> {
    let variants = database
        .get_feedback_by_variant(metric_name, function_name, variant_names, None, None)
        .await?;

    Ok(GetFeedbackByVariantResponse { variants })
}
