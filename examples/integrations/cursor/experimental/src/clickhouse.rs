use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use tensorzero_internal::serde_util::deserialize_json_string;
use tensorzero_internal::{
    clickhouse::ClickHouseConnectionInfo,
    inference::types::{ContentBlockChatOutput, ResolvedInput},
};
use uuid::Uuid;

use crate::git::CommitInterval;

#[derive(Debug, Deserialize)]
pub struct InferenceInfo {
    pub id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
}

pub async fn get_inferences_in_time_range(
    clickhouse: &ClickHouseConnectionInfo,
    commit_interval: CommitInterval,
) -> Result<Vec<InferenceInfo>> {
    // If start_time is None, set to 1 day ago
    let start_time_str = commit_interval
        .parent_timestamp
        .unwrap_or(Utc::now() - chrono::Duration::days(1))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let end_time_str = commit_interval
        .commit_timestamp
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    let query = r#"
    SELECT
        ci.id as id,
        ci.input as input,
        ci.output as output
    FROM InferenceById AS inferences
    JOIN ChatInference AS ci ON inferences.id_uint = toUInt128(ci.id)
         AND ci.function_name = inferences.function_name
         AND ci.variant_name = inferences.variant_name
         AND ci.episode_id = inferences.episode_id
         AND UUIDv7ToDateTime(uint_to_uuid(inferences.id_uint)) < {end_time:DateTime}
         AND UUIDv7ToDateTime(uint_to_uuid(inferences.id_uint)) >= {start_time:DateTime}
    FORMAT JSONEachRow
    "#;

    // Create the parameter map with string slices (&str)
    let parameters = HashMap::from([
        ("start_time", start_time_str.as_str()),
        ("end_time", end_time_str.as_str()),
    ]);

    // Pass the owned query string and the map of string slices
    let results = clickhouse
        .run_query_synchronous(query.to_string(), &parameters)
        .await?;

    let inferences: Vec<InferenceInfo> = results
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<InferenceInfo>)
        .collect::<Result<_, _>>()?;

    Ok(inferences)
}
