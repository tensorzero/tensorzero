use std::collections::HashMap;

use anyhow::Result;
use tensorzero_core::endpoints::datasets::{
    StoredChatInferenceDatapoint, StoredDatapoint, StoredJsonInferenceDatapoint,
};
use tensorzero_core::{db::clickhouse::ClickHouseConnectionInfo, function::FunctionConfig};
use tracing::{debug, info, instrument};

#[instrument(skip_all, fields(dataset_name = %dataset_name, function_name = %function_name, limit = ?limit))]
pub async fn query_dataset(
    clickhouse_client: &ClickHouseConnectionInfo,
    dataset_name: &str,
    function_name: &str,
    function_config: &FunctionConfig,
    limit: Option<usize>,
) -> Result<Vec<StoredDatapoint>> {
    let table_name = match function_config {
        FunctionConfig::Chat(_) => "ChatInferenceDatapoint",
        FunctionConfig::Json(_) => "JsonInferenceDatapoint",
    };
    debug!(table_name = %table_name, "Determined table name for function type");

    // Construct the query to fetch datapoints from the appropriate table
    let limit_clause = if let Some(limit_value) = limit {
        debug!(limit = limit_value, "Applying LIMIT to dataset query");
        "LIMIT {limit: UInt64}"
    } else {
        ""
    };

    let query = format!(
        r"SELECT * FROM {{table_name: Identifier}} FINAL
         WHERE dataset_name = {{dataset_name: String}}
         AND function_name = {{function_name: String}}
         AND staled_at IS NULL
         {limit_clause}
         FORMAT JSON"
    );

    // Need to bind limit_str to extend its lifetime beyond the HashMap creation
    let limit_str;
    let params = if let Some(limit_value) = limit {
        limit_str = limit_value.to_string();
        HashMap::from([
            ("table_name", table_name),
            ("dataset_name", dataset_name),
            ("function_name", function_name),
            ("limit", limit_str.as_str()),
        ])
    } else {
        HashMap::from([
            ("table_name", table_name),
            ("dataset_name", dataset_name),
            ("function_name", function_name),
        ])
    };

    debug!(query = %query, "Executing ClickHouse query");
    let result = clickhouse_client
        .run_query_synchronous(query.to_string(), &params)
        .await?;
    debug!(
        result_length = result.response.len(),
        "Query executed successfully"
    );
    debug!("Parsing datapoints from query result");
    let datapoints: Vec<StoredDatapoint> = match function_config {
        FunctionConfig::Chat(_) => {
            debug!("Parsing as chat datapoints");
            let chat_datapoints: serde_json::Value = serde_json::from_str(&result.response)?;
            let chat_datapoints: Vec<StoredChatInferenceDatapoint> =
                serde_json::from_value(chat_datapoints["data"].clone())?;
            let datapoints: Vec<StoredDatapoint> = chat_datapoints
                .into_iter()
                .map(StoredDatapoint::Chat)
                .collect();
            debug!(count = datapoints.len(), "Chat datapoints parsed");
            datapoints
        }
        FunctionConfig::Json(_) => {
            debug!("Parsing as JSON datapoints");
            let json_value: serde_json::Value = serde_json::from_str(&result.response)?;
            let json_datapoints: Vec<StoredJsonInferenceDatapoint> =
                serde_json::from_value(json_value["data"].clone())?;
            let datapoints: Vec<StoredDatapoint> = json_datapoints
                .into_iter()
                .map(StoredDatapoint::Json)
                .collect();
            debug!(count = datapoints.len(), "JSON datapoints parsed");
            datapoints
        }
    };

    info!(
        total_datapoints = datapoints.len(),
        "Dataset query completed successfully"
    );
    Ok(datapoints)
}
