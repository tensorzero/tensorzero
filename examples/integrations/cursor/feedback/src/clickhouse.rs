use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use tensorzero_core::serde_util::deserialize_json_string;
use tensorzero_core::{
    db::clickhouse::ClickHouseConnectionInfo,
    inference::types::{ContentBlockChatOutput, StoredInput},
};
use uuid::Uuid;

use crate::git::CommitInterval;
use crate::util::{get_max_uuidv7, get_min_uuidv7};

#[derive(Debug, Deserialize)]
pub struct InferenceInfo {
    pub id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
}

/// Gets the inferences which are relevant to a particular commit's interval.
/// For an inference to be relevant, it must satisfy 4 conditions:
/// 1. The `function_name` must be 'cursorzero'
/// 2. The inference must have happened in between the previous commit and the current commit's timestamps
/// 3. The inference must not have been given a float metric feedback
/// 4. The inference must be from the user specified in the user argument (if provided)
pub async fn get_inferences_in_time_range(
    clickhouse: &ClickHouseConnectionInfo,
    commit_interval: CommitInterval,
    user: Option<String>,
) -> Result<Vec<Arc<InferenceInfo>>> {
    // If start_time is None, set to 1 day ago
    let lower_bound = get_min_uuidv7(
        commit_interval
            .parent_timestamp
            .unwrap_or_else(|| Utc::now() - chrono::Duration::days(1)),
    )
    .to_string();
    let upper_bound = get_max_uuidv7(commit_interval.commit_timestamp).to_string();

    // Create the parameter map with string slices (&str)
    let mut parameters = HashMap::from([
        ("lower_bound", lower_bound.as_str()),
        ("upper_bound", upper_bound.as_str()),
    ]);

    let user_where_clause = if let Some(ref user) = user {
        parameters.insert("user", user);
        "AND ci.tags['user'] = {user:String}".to_string()
    } else {
        String::new()
    };

    let mut query = r"WITH inference_ids AS (
        SELECT uint_to_uuid(id_uint) as id
        FROM InferenceById
        WHERE id_uint >= toUInt128({lower_bound:UUID})
        AND id_uint <= toUInt128({upper_bound:UUID})
    )
    SELECT
        ci.id as id,
        ci.input as input,
        ci.output as output
    FROM ChatInference AS ci
    LEFT ANTI JOIN FloatMetricFeedbackByTargetId AS fmf ON fmf.target_id = ci.id
    WHERE
        ci.function_name = 'cursorzero'
        AND ci.id IN (SELECT id FROM inference_ids)"
        .to_string();
    query.push_str(&user_where_clause);
    query.push_str(" FORMAT JSONEachRow");

    // Pass the owned query string and the map of string slices
    let results = clickhouse
        .run_query_synchronous(query.to_string(), &parameters)
        .await?;

    let inferences: Vec<Arc<InferenceInfo>> = results
        .response
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|x| serde_json::from_str::<InferenceInfo>(x).map(Arc::new))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(inferences)
}
