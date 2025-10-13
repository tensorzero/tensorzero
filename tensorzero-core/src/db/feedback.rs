use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::TableBounds;
use crate::serde_util::deserialize_u64;

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
