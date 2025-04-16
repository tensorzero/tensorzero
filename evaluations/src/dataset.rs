use std::collections::HashMap;

use anyhow::Result;
use tensorzero_internal::endpoints::datasets::{
    ClickHouseChatInferenceDatapoint, ClickHouseDatapoint, ClickHouseJsonInferenceDatapoint,
    Datapoint,
};
use tensorzero_internal::{clickhouse::ClickHouseConnectionInfo, function::FunctionConfig};

pub async fn query_dataset(
    clickhouse_client: &ClickHouseConnectionInfo,
    dataset_name: &str,
    function_name: &str,
    function_config: &FunctionConfig,
) -> Result<Vec<Datapoint>> {
    let table_name = match function_config {
        FunctionConfig::Chat(_) => "ChatInferenceDatapoint",
        FunctionConfig::Json(_) => "JsonInferenceDatapoint",
    };

    // Construct the query to fetch datapoints from the appropriate table
    let query = r#"SELECT * FROM {table_name: Identifier} FINAL
         WHERE dataset_name = {dataset_name: String}
         AND function_name = {function_name: String}
         AND staled_at IS NULL
         FORMAT JSON"#;

    let params = HashMap::from([
        ("table_name", table_name),
        ("dataset_name", dataset_name),
        ("function_name", function_name),
    ]);

    let result = clickhouse_client
        .run_query_synchronous(query.to_string(), Some(&params))
        .await?;
    let datapoints: Vec<Datapoint> = match function_config {
        FunctionConfig::Chat(_) => {
            let chat_datapoints: serde_json::Value = serde_json::from_str(&result)?;
            let chat_datapoints: Vec<ClickHouseChatInferenceDatapoint> =
                serde_json::from_value(chat_datapoints["data"].clone())?;
            chat_datapoints
                .into_iter()
                .map(ClickHouseDatapoint::Chat)
                .map(Datapoint::from)
                .collect()
        }
        FunctionConfig::Json(_) => {
            let json_value: serde_json::Value = serde_json::from_str(&result)?;
            let json_datapoints: Vec<ClickHouseJsonInferenceDatapoint> =
                serde_json::from_value(json_value["data"].clone())?;
            json_datapoints
                .into_iter()
                .map(ClickHouseDatapoint::Json)
                .map(Datapoint::from)
                .collect()
        }
    };
    Ok(datapoints)
}
