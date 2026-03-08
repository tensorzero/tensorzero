use async_trait::async_trait;
use serde::Deserialize;

use super::ClickHouseConnectionInfo;
use crate::db::{HowdyFeedbackCounts, HowdyInferenceCounts, HowdyQueries, HowdyTokenUsage};
use crate::error::{Error, ErrorDetails};
use crate::serde_util::{deserialize_option_u64, deserialize_u64};

#[derive(Debug, Deserialize)]
struct ClickHouseInferenceCounts {
    #[serde(deserialize_with = "deserialize_u64")]
    chat_inference_count: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    json_inference_count: u64,
}

#[derive(Debug, Deserialize)]
struct ClickHouseFeedbackCounts {
    #[serde(deserialize_with = "deserialize_u64")]
    boolean_metric_feedback_count: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    float_metric_feedback_count: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    demonstration_feedback_count: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    comment_feedback_count: u64,
}

#[derive(Debug, Deserialize)]
struct ClickHouseTokenUsage {
    #[serde(deserialize_with = "deserialize_option_u64")]
    input_tokens: Option<u64>,
    #[serde(deserialize_with = "deserialize_option_u64")]
    output_tokens: Option<u64>,
}

#[async_trait]
impl HowdyQueries for ClickHouseConnectionInfo {
    async fn count_inferences_for_howdy(&self) -> Result<HowdyInferenceCounts, Error> {
        let response = self
            .run_query_synchronous_no_params(
                r"
                SELECT
                    (SELECT COUNT() FROM ChatInference) as chat_inference_count,
                    (SELECT COUNT() FROM JsonInference) as json_inference_count
                Format JSONEachRow
                "
                .to_string(),
            )
            .await?;
        let counts: ClickHouseInferenceCounts =
            serde_json::from_str(&response.response).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize howdy inference counts: {e}"),
                })
            })?;
        Ok(HowdyInferenceCounts {
            chat_inference_count: counts.chat_inference_count,
            json_inference_count: counts.json_inference_count,
        })
    }

    async fn count_feedbacks_for_howdy(&self) -> Result<HowdyFeedbackCounts, Error> {
        let response = self
            .run_query_synchronous_no_params(
                r"
                SELECT
                    (SELECT COUNT() FROM BooleanMetricFeedback) as boolean_metric_feedback_count,
                    (SELECT COUNT() FROM FloatMetricFeedback) as float_metric_feedback_count,
                    (SELECT COUNT() FROM DemonstrationFeedback) as demonstration_feedback_count,
                    (SELECT COUNT() FROM CommentFeedback) as comment_feedback_count
                Format JSONEachRow
                "
                .to_string(),
            )
            .await?;
        let counts: ClickHouseFeedbackCounts =
            serde_json::from_str(&response.response).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize howdy feedback counts: {e}"),
                })
            })?;
        Ok(HowdyFeedbackCounts {
            boolean_metric_feedback_count: counts.boolean_metric_feedback_count,
            float_metric_feedback_count: counts.float_metric_feedback_count,
            comment_feedback_count: counts.comment_feedback_count,
            demonstration_feedback_count: counts.demonstration_feedback_count,
        })
    }

    async fn get_token_totals_for_howdy(&self) -> Result<HowdyTokenUsage, Error> {
        let response = self
            .run_query_synchronous_no_params(
                r"
                SELECT
                    (SELECT count FROM CumulativeUsage FINAL WHERE type = 'input_tokens') as input_tokens,
                    (SELECT count FROM CumulativeUsage FINAL WHERE type = 'output_tokens') as output_tokens
                Format JSONEachRow
                "
                .to_string(),
            )
            .await?;
        let usage: ClickHouseTokenUsage =
            serde_json::from_str(&response.response).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize howdy token usage: {e}"),
                })
            })?;
        Ok(HowdyTokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        })
    }
}
