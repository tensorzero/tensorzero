//! This module is responsible for sending usage data to the Howdy service.
//! It is configured via the `TENSORZERO_HOWDY_URL` environment variable.
//! If the `TENSORZERO_DISABLE_USAGE_DATA` environment variable is set to `1`,
//! the usage data will not be sent.
//!
//! The usage data is sent every 6 hours.
//!
//! The usage data is sent to the Howdy service in the following format:
//!
//! ```json
//! {
//!     "deployment_id": "123",
//!     "inferences": 100,
//!     "feedbacks": 50,
//!     "dryrun": false
//! }
//! ```
//!
//! We only send an opaque and unidentifiable deployment ID (64 char hex hash)
//! and the number of inferences and feedbacks to the Howdy service.

use lazy_static::lazy_static;
use reqwest::Client;
use serde::Serialize;
use std::env;
use tokio::{
    time::{self, Duration},
    try_join,
};
use tracing::{debug, info};

use crate::clickhouse::ClickHouseConnectionInfo;

lazy_static! {
    /// The URL to send usage data to.
    /// Configurable via the `TENSORZERO_HOWDY_URL` environment variable for testing.
    pub static ref HOWDY_URL: String =
        env::var("TENSORZERO_HOWDY_URL").unwrap_or("https://howdy.tensorzero.com".to_string());
}

/// Setup the howdy loop.
/// This is called from the main function in the gateway or embedded client.
pub fn setup_howdy(clickhouse: ClickHouseConnectionInfo) {
    if env::var("TENSORZERO_DISABLE_USAGE_DATA").unwrap_or_default() == "1" {
        info!("Usage data is disabled");
        return;
    }
    tokio::spawn(howdy_loop(clickhouse));
}

/// Loops and sends usage data to the Howdy service every 6 hours.
pub async fn howdy_loop(clickhouse: ClickHouseConnectionInfo) {
    let client = Client::new();
    let deployment_id = match get_deployment_id(&clickhouse).await {
        Ok(deployment_id) => deployment_id,
        Err(_) => {
            return;
        }
    };
    let mut interval = time::interval(Duration::from_secs(6 * 60 * 60));
    loop {
        interval.tick().await;
        // TODO: do we need to spawn this? it can't fail in theory
        if let Err(e) = send_howdy(&clickhouse, &client, &deployment_id).await {
            debug!("{e}");
        }
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
    let (inferences, feedbacks) =
        try_join!(count_inferences(clickhouse), count_feedbacks(clickhouse))?;
    Ok(HowdyReportBody {
        deployment_id,
        inferences,
        feedbacks,
        dryrun,
    })
}

/// Count all inferences in the ClickHouse DB.
/// This returns a string since u64 cannot be represented in JSON and we simply pass it through to howdy
async fn count_inferences(clickhouse: &ClickHouseConnectionInfo) -> Result<String, String> {
    let Ok(response) = clickhouse
        .run_query_synchronous_no_params(
            r#"SELECT SUM(count) AS inference_count
                  FROM
                  (
                    SELECT COUNT() AS count FROM ChatInference
                    UNION ALL
                    SELECT COUNT() AS count FROM JsonInference
                  )
                  "#
            .to_string(),
        )
        .await
    else {
        return Err("Failed to query ClickHouse for inference count".to_string());
    };
    let response_str = response.response.trim().to_string();
    // make sure this parses as a u64
    if response_str.parse::<u64>().is_err() {
        return Err(format!(
            "Failed to parse inference count as u64: {}",
            response.response
        ));
    }
    Ok(response_str)
}

/// Count all feedbacks in the ClickHouse DB.
/// This returns a string since u64 cannot be represented in JSON and we simply pass it through to howdy
async fn count_feedbacks(clickhouse: &ClickHouseConnectionInfo) -> Result<String, String> {
    let Ok(response) = clickhouse
        .run_query_synchronous_no_params(
            r#"SELECT SUM(count) AS feedback_count
                  FROM
                  (
                    SELECT COUNT() AS count FROM BooleanMetricFeedback
                    UNION ALL
                    SELECT COUNT() AS count FROM FloatMetricFeedback
                    UNION ALL
                    SELECT COUNT() AS count FROM DemonstrationFeedback
                    UNION ALL
                    SELECT COUNT() AS count FROM CommentFeedback
                  )
                  "#
            .to_string(),
        )
        .await
    else {
        return Err("Failed to query ClickHouse for feedback count".to_string());
    };
    let response_str = response.response.trim().to_string();
    if response_str.parse::<u64>().is_err() {
        return Err(format!(
            "Failed to parse feedback count as u64: {}",
            response.response
        ));
    }
    Ok(response_str)
}

#[derive(Debug, Serialize)]
pub struct HowdyReportBody<'a> {
    pub deployment_id: &'a str,
    pub inferences: String,
    pub feedbacks: String,
    #[serde(default)]
    pub dryrun: bool,
}
