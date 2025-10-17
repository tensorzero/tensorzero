use crate::error::Error;
use async_trait::async_trait;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::TableBounds;
use crate::serde_util::deserialize_u64;

#[async_trait]
pub trait FeedbackQueries {
    /// Retrieves cumulative feedback statistics for a given metric and function, optionally filtered by variant names.
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error>;

    /// Queries all feedback (boolean metrics, float metrics, comments, demonstrations) for a given target ID
    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error>;

    /// Queries feedback bounds for a given target ID
    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error>;

    /// Counts total feedback items for a given target ID
    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error>;

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error>;
}

#[derive(Debug, Deserialize)]
pub struct FeedbackByVariant {
    pub variant_name: String,
    pub mean: f32,
    pub variance: f32,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

// Feedback by target ID types
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct BooleanMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: bool,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FloatMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: f64,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CommentFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub target_type: CommentTargetType,
    pub value: String,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum CommentTargetType {
    Inference,
    Episode,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct DemonstrationFeedbackRow {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub value: String,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum FeedbackRow {
    Boolean(BooleanMetricFeedbackRow),
    Float(FloatMetricFeedbackRow),
    Comment(CommentFeedbackRow),
    Demonstration(DemonstrationFeedbackRow),
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FeedbackBounds {
    #[ts(optional)]
    pub first_id: Option<Uuid>,
    #[ts(optional)]
    pub last_id: Option<Uuid>,
    pub by_type: FeedbackBoundsByType,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FeedbackBoundsByType {
    pub boolean: TableBounds,
    pub float: TableBounds,
    pub comment: TableBounds,
    pub demonstration: TableBounds,
}
