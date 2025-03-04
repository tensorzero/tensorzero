use std::collections::HashMap;

use anyhow::Result;
use tensorzero_internal::endpoints::datasets::{
    ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint,
};
use tensorzero_internal::{clickhouse::ClickHouseConnectionInfo, function::FunctionConfig};

#[allow(dead_code)]
pub async fn query_dataset(
    clickhouse_client: &ClickHouseConnectionInfo,
    dataset_name: &str,
    function_name: &str,
    function_config: &FunctionConfig,
) -> Result<Vec<Datapoint>> {
    let dataset_table = match function_config {
        FunctionConfig::Chat(_) => "ChatInferenceDataset",
        FunctionConfig::Json(_) => "JsonInferenceDataset",
    };
    let query = "SELECT * FROM {table_name: Identifier} WHERE dataset_name = '{dataset_name: String}' and function_name = '{function_name: String}' FORMAT JSONEachRow".to_string();
    let params = HashMap::from([
        ("table_name", dataset_table),
        ("dataset_name", dataset_name),
        ("function_name", function_name),
    ]);

    let result = clickhouse_client.run_query(query, Some(&params)).await?;
    let datapoints: Vec<Datapoint> = match function_config {
        FunctionConfig::Chat(_) => {
            let chat_datapoints: Vec<ChatInferenceDatapoint> = serde_json::from_str(&result)?;
            chat_datapoints
                .into_iter()
                .map(Datapoint::ChatInference)
                .collect()
        }
        FunctionConfig::Json(_) => {
            let json_datapoints: Vec<JsonInferenceDatapoint> = serde_json::from_str(&result)?;
            json_datapoints
                .into_iter()
                .map(Datapoint::JsonInference)
                .collect()
        }
    };
    Ok(datapoints)
}
