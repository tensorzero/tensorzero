//! This module is responsible for sending usage data to the TensorZero Howdy service.
//! It is configured via the `TENSORZERO_HOWDY_URL` environment variable.
//! If the `TENSORZERO_DISABLE_PSEUDONYMOUS_USAGE_ANALYTICS` environment variable is set to `1`,
//! the usage data will not be sent.
//!
//! The usage data is sent every 6 hours.
//!
//! The usage data is sent to the TensorZero Howdy service in the following format:
//!
//! ```json
//! {
//!     "deployment_id": "8c40087ea3a0928229c57c378d1144035cc4334e99ab9b5d1fd880849c16feff",
//!     "inference_count": 100,
//!     "feedback_count": 50,
//!     "gateway_version": "2025.7.0",
//!     "dryrun": false
//! }
//! ```
//!
//! We only send an opaque and unidentifiable deployment ID (64 char hex hash)
//! and the number of inferences and feedbacks to the TensorZero Howdy service.

use lazy_static::lazy_static;
use reqwest::Client;
use serde::{Deserialize, Deserializer, Serialize};
use std::env;
use tokio::{
    time::{self, Duration},
    try_join,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::{config::Config, utils::spawn_ignoring_shutdown};

lazy_static! {
    /// The URL to send usage data to.
    /// Configurable via the `TENSORZERO_HOWDY_URL` environment variable for testing.
    pub static ref HOWDY_URL: String =
        env::var("TENSORZERO_HOWDY_URL").unwrap_or_else(|_| "https://howdy.tensorzero.com".to_string());
}

/// Setup the howdy loop.
/// This is called from the main function in the gateway or embedded client.
pub fn setup_howdy(
    config: &Config,
    clickhouse: ClickHouseConnectionInfo,
    token: CancellationToken,
) {
    if config.gateway.disable_pseudonymous_usage_analytics
        || env::var("TENSORZERO_DISABLE_PSEUDONYMOUS_USAGE_ANALYTICS").unwrap_or_default() == "1"
    {
        info!("Pseudonymous usage analytics is disabled");
        return;
    }
    // TODO(shuyangli): Don't like this...
    if clickhouse.client_type() == ClickHouseClientType::Disabled {
        return;
    }
    spawn_ignoring_shutdown(howdy_loop(clickhouse, token));
}

/// Loops and sends usage data to the Howdy service every 6 hours.
pub async fn howdy_loop(clickhouse: ClickHouseConnectionInfo, token: CancellationToken) {
    let client = Client::new();
    let deployment_id = match get_deployment_id(&clickhouse).await {
        Ok(deployment_id) => deployment_id,
        Err(()) => {
            return;
        }
    };
    let mut interval = time::interval(Duration::from_secs(6 * 60 * 60));
    loop {
        let copied_clickhouse = clickhouse.clone();
        let copied_client = client.clone();
        let copied_deployment_id = deployment_id.clone();
        tokio::select! {
            () = token.cancelled() => {
                break;
            }
            _ = interval.tick() => {}
        }
        spawn_ignoring_shutdown(async move {
            if let Err(e) =
                send_howdy(&copied_clickhouse, &copied_client, &copied_deployment_id).await
            {
                debug!("{e}");
            }
        });
    }
}

/// Sends usage data to the Howdy service.
async fn send_howdy(
    clickhouse: &ClickHouseConnectionInfo,
    client: &Client,
    deployment_id: &str,
) -> Result<(), String> {
    let howdy_url = HOWDY_URL.clone();
    let howdy_report = get_howdy_report(clickhouse, deployment_id).await?;
    if let Err(e) = client.post(&howdy_url).json(&howdy_report).send().await {
        return Err(format!("Failed to send howdy: {e}"));
    }
    Ok(())
}

/// Gets the deployment ID from the ClickHouse DB.
/// This is a 64 char hex hash that is used to identify the deployment.
/// It is stored in the `DeploymentID` table.
pub async fn get_deployment_id(clickhouse: &ClickHouseConnectionInfo) -> Result<String, ()> {
    let response = clickhouse
        .run_query_synchronous_no_params(
            "SELECT deployment_id FROM DeploymentID LIMIT 1".to_string(),
        )
        .await
        .map_err(|_| ())?;
    if response.response.is_empty() {
        debug!("Failed to get deployment ID (response was empty)");
        return Err(());
    }
    Ok(response.response.trim().to_string())
}

/// Gets the howdy report.
/// This is the data that is sent to the Howdy service.
/// It contains the deployment ID, the number of inferences, and the number of feedbacks.
/// It is also configurable to run in dryrun mode for testing.
pub async fn get_howdy_report<'a>(
    clickhouse: &ClickHouseConnectionInfo,
    deployment_id: &'a str,
) -> Result<HowdyReportBody<'a>, String> {
    let dryrun = cfg!(any(test, feature = "e2e_tests"));
    let (inference_counts, feedback_counts, token_totals) = try_join!(
        count_inferences(clickhouse),
        count_feedbacks(clickhouse),
        get_token_totals(clickhouse)
    )?;
    Ok(HowdyReportBody {
        deployment_id,
        inference_count: inference_counts.inference_count,
        feedback_count: feedback_counts.feedback_count,
        gateway_version: crate::endpoints::status::TENSORZERO_VERSION,
        commit_hash: crate::built_info::GIT_COMMIT_HASH_SHORT,
        input_token_total: token_totals.input_tokens.map(|x| x.to_string()),
        output_token_total: token_totals.output_tokens.map(|x| x.to_string()),
        chat_inference_count: Some(inference_counts.chat_inference_count),
        json_inference_count: Some(inference_counts.json_inference_count),
        float_metric_feedback_count: Some(feedback_counts.float_metric_feedback_count),
        boolean_metric_feedback_count: Some(feedback_counts.boolean_metric_feedback_count),
        comment_feedback_count: Some(feedback_counts.comment_feedback_count),
        demonstration_feedback_count: Some(feedback_counts.demonstration_feedback_count),
        dryrun,
    })
}

#[derive(Debug)]
struct InferenceCounts {
    inference_count: String,
    chat_inference_count: String,
    json_inference_count: String,
}

#[derive(Debug, Deserialize)]
struct ClickHouseInferenceCounts {
    #[serde(deserialize_with = "deserialize_u64")]
    chat_inference_count: u64,
    #[serde(deserialize_with = "deserialize_u64")]
    json_inference_count: u64,
}

// Since ClickHouse returns strings for UInt64, we need to deserialize them as u64
fn deserialize_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Count all inferences in the ClickHouse DB.
/// This returns a string since u64 cannot be represented in JSON and we simply pass it through to howdy
async fn count_inferences(
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<InferenceCounts, String> {
    let Ok(response) = clickhouse
        .run_query_synchronous_no_params(
            r"
            SELECT
                (SELECT COUNT() FROM ChatInference) as chat_inference_count,
                (SELECT COUNT() FROM JsonInference) as json_inference_count
            Format JSONEachRow
            "
            .to_string(),
        )
        .await
    else {
        return Err("Failed to query ClickHouse for inference count".to_string());
    };
    let response_counts: ClickHouseInferenceCounts = serde_json::from_str(&response.response)
        .map_err(|e| format!("Failed to deserialize ClickHouseInferenceCounts: {e}"))?;

    let inference_count =
        response_counts.chat_inference_count + response_counts.json_inference_count;

    Ok(InferenceCounts {
        inference_count: inference_count.to_string(),
        chat_inference_count: response_counts.chat_inference_count.to_string(),
        json_inference_count: response_counts.json_inference_count.to_string(),
    })
}

#[derive(Debug, Deserialize)]
struct CumulativeUsage {
    #[serde(deserialize_with = "deserialize_optional_u64")]
    input_tokens: Option<u64>,
    #[serde(deserialize_with = "deserialize_optional_u64")]
    output_tokens: Option<u64>,
}

// Since ClickHouse returns strings for UInt64, we need to deserialize them as u64
// If the value is null we return None
fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNull {
        String(String),
        Null,
    }

    match StringOrNull::deserialize(deserializer)? {
        StringOrNull::String(s) => Some(s.parse().map_err(serde::de::Error::custom)).transpose(),
        StringOrNull::Null => Ok(None),
    }
}

async fn get_token_totals(
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<CumulativeUsage, String> {
    let Ok(response) = clickhouse
        .run_query_synchronous_no_params(
            r"
            SELECT
                (SELECT count FROM CumulativeUsage FINAL WHERE type = 'input_tokens') as input_tokens,
                (SELECT count FROM CumulativeUsage FINAL WHERE type = 'output_tokens') as output_tokens
            Format JSONEachRow
            "
            .to_string(),
        )
        .await
    else {
        return Err("Failed to query ClickHouse for token total count".to_string());
    };
    serde_json::from_str(&response.response)
        .map_err(|e| format!("Failed to deserialize ClickHouseInferenceCounts: {e}"))
}

#[derive(Debug)]
struct FeedbackCounts {
    feedback_count: String,
    float_metric_feedback_count: String,
    boolean_metric_feedback_count: String,
    comment_feedback_count: String,
    demonstration_feedback_count: String,
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

/// Count all feedbacks in the ClickHouse DB.
/// This returns a string since u64 cannot be represented in JSON and we simply pass it through to howdy
async fn count_feedbacks(clickhouse: &ClickHouseConnectionInfo) -> Result<FeedbackCounts, String> {
    let Ok(response) = clickhouse
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
        .await
    else {
        return Err("Failed to query ClickHouse for feedback count".to_string());
    };
    let response_counts: ClickHouseFeedbackCounts = serde_json::from_str(&response.response)
        .map_err(|e| format!("Failed to deserialize ClickHouseFeedbackCounts: {e}"))?;

    let feedback_count = response_counts.boolean_metric_feedback_count
        + response_counts.float_metric_feedback_count
        + response_counts.demonstration_feedback_count
        + response_counts.comment_feedback_count;

    Ok(FeedbackCounts {
        feedback_count: feedback_count.to_string(),
        float_metric_feedback_count: response_counts.float_metric_feedback_count.to_string(),
        boolean_metric_feedback_count: response_counts.boolean_metric_feedback_count.to_string(),
        comment_feedback_count: response_counts.comment_feedback_count.to_string(),
        demonstration_feedback_count: response_counts.demonstration_feedback_count.to_string(),
    })
}

#[derive(Debug, Serialize)]
pub struct HowdyReportBody<'a> {
    pub deployment_id: &'a str,
    pub inference_count: String,
    pub feedback_count: String,
    pub gateway_version: &'static str,
    pub commit_hash: Option<&'static str>,
    pub input_token_total: Option<String>,
    pub output_token_total: Option<String>,
    pub chat_inference_count: Option<String>,
    pub json_inference_count: Option<String>,
    pub float_metric_feedback_count: Option<String>,
    pub boolean_metric_feedback_count: Option<String>,
    pub comment_feedback_count: Option<String>,
    pub demonstration_feedback_count: Option<String>,
    #[serde(default)]
    pub dryrun: bool,
}
