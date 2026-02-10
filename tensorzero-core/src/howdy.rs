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
use serde::Serialize;
use std::env;
use tokio::{
    time::{self, Duration},
    try_join,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::{DeploymentIdQueries, HowdyQueries};
use crate::{config::Config, utils::spawn_ignoring_shutdown};
use crate::{db::clickhouse::ClickHouseConnectionInfo, feature_flags};

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
    postgres: PostgresConnectionInfo,
    token: CancellationToken,
) {
    if config.gateway.disable_pseudonymous_usage_analytics
        || env::var("TENSORZERO_DISABLE_PSEUDONYMOUS_USAGE_ANALYTICS").unwrap_or_default() == "1"
    {
        info!("Pseudonymous usage analytics is disabled");
        return;
    }

    // TODO(#5691): Support reading deployment ID when ClickHouse is disabled.
    let clickhouse_disabled = clickhouse.client_type() == ClickHouseClientType::Disabled;
    if clickhouse_disabled {
        return;
    }
    spawn_ignoring_shutdown(howdy_loop(clickhouse, postgres, token));
}

/// Loops and sends usage data to the Howdy service every 6 hours.
pub async fn howdy_loop(
    clickhouse: ClickHouseConnectionInfo,
    postgres: PostgresConnectionInfo,
    token: CancellationToken,
) {
    let db = DelegatingDatabaseConnection::new(clickhouse.clone(), postgres.clone());
    let client = Client::new();
    let deployment_id = match get_deployment_id(&clickhouse, &postgres).await {
        Ok(deployment_id) => deployment_id,
        Err(()) => {
            return;
        }
    };
    let mut interval = time::interval(Duration::from_secs(6 * 60 * 60));
    loop {
        let copied_db = db.clone();
        let copied_client = client.clone();
        let copied_deployment_id = deployment_id.clone();
        tokio::select! {
            () = token.cancelled() => {
                break;
            }
            _ = interval.tick() => {}
        }
        spawn_ignoring_shutdown(async move {
            if let Err(e) = send_howdy(&copied_db, &copied_client, &copied_deployment_id).await {
                debug!("{e}");
            }
        });
    }
}

/// Sends usage data to the Howdy service.
async fn send_howdy(
    db: &DelegatingDatabaseConnection,
    client: &Client,
    deployment_id: &str,
) -> Result<(), String> {
    let howdy_url = HOWDY_URL.clone();
    let howdy_report = get_howdy_report(db, deployment_id).await?;
    if let Err(e) = client.post(&howdy_url).json(&howdy_report).send().await {
        return Err(format!("Failed to send howdy: {e}"));
    }
    Ok(())
}

/// Synchronizes the deployment ID from ClickHouse to Postgres if Postgres is enabled.
/// For existing ClickHouse deployments, we make sure Postgres contains the same deployment ID.
/// This only executes the actual synchronization once.
async fn synchronize_deployment_id(
    clickhouse: &ClickHouseConnectionInfo,
    postgres: &PostgresConnectionInfo,
) -> Result<(), ()> {
    // Even though this writes deployment ID to Postgres, we gate it behind ENABLE_POSTGRES_READ
    // because it's serving a read query.
    if !feature_flags::ENABLE_POSTGRES_READ.get() {
        return Ok(());
    }
    if clickhouse.client_type() != ClickHouseClientType::Production {
        return Ok(());
    }
    if !matches!(postgres, &PostgresConnectionInfo::Enabled { .. }) {
        return Ok(());
    }
    let Ok(id) = clickhouse.get_deployment_id().await else {
        tracing::debug!("Failed to get deployment ID from ClickHouse");
        return Err(());
    };
    if let Err(e) = &postgres.insert_deployment_id(&id).await {
        tracing::debug!("Failed to sync deployment ID to Postgres: {e}");
        return Err(());
    }

    Ok(())
}

/// Gets the deployment ID.
/// This is a 64 char hex hash that is used to identify the deployment.
pub async fn get_deployment_id(
    clickhouse: &ClickHouseConnectionInfo,
    postgres: &PostgresConnectionInfo,
) -> Result<String, ()> {
    // Make sure deployment ID is consistent between ClickHouse and Postgres
    synchronize_deployment_id(clickhouse, postgres).await?;

    DelegatingDatabaseConnection::new(clickhouse.clone(), postgres.clone())
        .get_deployment_id()
        .await
        .map_err(|e| {
            tracing::debug!("Failed to get deployment ID: {e}");
        })
}

/// Gets the howdy report.
/// This is the data that is sent to the Howdy service.
/// It contains the deployment ID, the number of inferences, and the number of feedbacks.
/// It is also configurable to run in dryrun mode for testing.
pub async fn get_howdy_report<'a>(
    db: &(dyn HowdyQueries + Sync),
    deployment_id: &'a str,
) -> Result<HowdyReportBody<'a>, String> {
    let dryrun = cfg!(any(test, feature = "e2e_tests"));
    let (inference_counts, feedback_counts, token_totals) = try_join!(
        db.count_inferences_for_howdy(),
        db.count_feedbacks_for_howdy(),
        db.get_token_totals_for_howdy()
    )
    .map_err(|e| e.to_string())?;

    let inference_count =
        inference_counts.chat_inference_count + inference_counts.json_inference_count;
    let feedback_count = feedback_counts.boolean_metric_feedback_count
        + feedback_counts.float_metric_feedback_count
        + feedback_counts.comment_feedback_count
        + feedback_counts.demonstration_feedback_count;

    Ok(HowdyReportBody {
        deployment_id,
        inference_count: inference_count.to_string(),
        feedback_count: feedback_count.to_string(),
        gateway_version: crate::endpoints::status::TENSORZERO_VERSION,
        commit_hash: crate::built_info::GIT_COMMIT_HASH_SHORT,
        input_token_total: token_totals.input_tokens.map(|x| x.to_string()),
        output_token_total: token_totals.output_tokens.map(|x| x.to_string()),
        chat_inference_count: Some(inference_counts.chat_inference_count.to_string()),
        json_inference_count: Some(inference_counts.json_inference_count.to_string()),
        float_metric_feedback_count: Some(feedback_counts.float_metric_feedback_count.to_string()),
        boolean_metric_feedback_count: Some(
            feedback_counts.boolean_metric_feedback_count.to_string(),
        ),
        comment_feedback_count: Some(feedback_counts.comment_feedback_count.to_string()),
        demonstration_feedback_count: Some(
            feedback_counts.demonstration_feedback_count.to_string(),
        ),
        dryrun,
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
