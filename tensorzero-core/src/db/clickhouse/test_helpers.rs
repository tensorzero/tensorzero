#![expect(
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::print_stdout,
    clippy::unwrap_used
)]
use crate::config::BatchWritesConfig;
use crate::endpoints::datasets::{JsonInferenceDatapoint, StoredChatInferenceDatapoint};
use crate::endpoints::workflow_evaluation_run::{
    WorkflowEvaluationRunEpisodeRow, WorkflowEvaluationRunRow,
};

#[cfg(feature = "e2e_tests")]
use super::escape_string_for_clickhouse_literal;
use super::ClickHouseConnectionInfo;
#[cfg(feature = "e2e_tests")]
use crate::endpoints::feedback::human_feedback::StaticEvaluationHumanFeedback;
use serde_json::Value;
#[cfg(feature = "e2e_tests")]
use std::collections::HashMap;
use uuid::Uuid;

lazy_static::lazy_static! {
    pub static ref CLICKHOUSE_URL: String = std::env::var("TENSORZERO_CLICKHOUSE_URL").expect("Environment variable TENSORZERO_CLICKHOUSE_URL must be set");
    pub static ref CLICKHOUSE_REPLICA_URL: Option<String> = std::env::var("TENSORZERO_CLICKHOUSE_REPLICA_URL").ok();
}

pub async fn get_clickhouse() -> ClickHouseConnectionInfo {
    let clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    let start = std::time::Instant::now();
    println!("Connecting to ClickHouse");
    let res = ClickHouseConnectionInfo::new(clickhouse_url.as_ref(), BatchWritesConfig::default())
        .await
        .expect("Failed to connect to ClickHouse");
    println!("Connected to ClickHouse in {:?}", start.elapsed());
    res
}

pub async fn get_clickhouse_replica() -> Option<ClickHouseConnectionInfo> {
    let clickhouse_url = CLICKHOUSE_REPLICA_URL.as_ref()?;
    let clickhouse_url = url::Url::parse(clickhouse_url).unwrap();
    let start = std::time::Instant::now();
    println!("Connecting to ClickHouse");
    let res = ClickHouseConnectionInfo::new(clickhouse_url.as_ref(), BatchWritesConfig::default())
        .await
        .expect("Failed to connect to ClickHouse");
    println!("Connected to ClickHouse in {:?}", start.elapsed());
    Some(res)
}

#[cfg(feature = "e2e_tests")]
pub async fn clickhouse_flush_async_insert(clickhouse: &ClickHouseConnectionInfo) {
    if let Err(e) = clickhouse
        .run_query_synchronous_no_params("SYSTEM FLUSH ASYNC INSERT QUEUE".to_string())
        .await
    {
        tracing::warn!("Failed to run `SYSTEM FLUSH ASYNC INSERT QUEUE`: {}", e);
    }
}

pub async fn select_chat_datapoint_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT
            dataset_name,
            function_name,
            id,
            name,
            episode_id,
            input,
            output,
            tool_params,
            dynamic_tools,
            dynamic_provider_tools,
            tool_choice,
            parallel_tool_calls,
            allowed_tools,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id,
            staled_at,
            updated_at
        FROM ChatInferenceDatapoint FINAL
        WHERE id = '{inference_id}'
        LIMIT 1
        FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_json_datapoint_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM JsonInferenceDatapoint FINAL WHERE id = '{inference_id}' LIMIT 1 FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_chat_dataset_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    dataset_name: &str,
) -> Option<Vec<StoredChatInferenceDatapoint>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT
            dataset_name,
            function_name,
            id,
            name,
            episode_id,
            input,
            output,
            tool_params,
            dynamic_tools,
            dynamic_provider_tools,
            tool_choice,
            parallel_tool_calls,
            allowed_tools,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id,
            staled_at,
            formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
        FROM ChatInferenceDatapoint FINAL
        WHERE dataset_name = '{dataset_name}' AND staled_at IS NULL
        FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let lines = text.response.lines();
    let mut chat_rows: Vec<StoredChatInferenceDatapoint> = Vec::new();
    for line in lines {
        let chat_row: StoredChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        chat_rows.push(chat_row);
    }
    Some(chat_rows)
}

pub async fn select_json_dataset_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    dataset_name: &str,
) -> Option<Vec<JsonInferenceDatapoint>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT
            dataset_name,
            function_name,
            id,
            name,
            episode_id,
            input,
            output,
            output_schema,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id,
            staled_at,
            formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
        FROM JsonInferenceDatapoint FINAL
        WHERE dataset_name = '{dataset_name}' AND staled_at IS NULL
        FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let lines = text.response.lines();
    let mut json_rows: Vec<JsonInferenceDatapoint> = Vec::new();
    for line in lines {
        let json_row: JsonInferenceDatapoint = serde_json::from_str(line).unwrap();
        json_rows.push(json_row);
    }

    Some(json_rows)
}

pub async fn select_chat_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    episode_id: Uuid,
) -> Option<Vec<Value>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query =
        format!("SELECT * FROM ChatInference WHERE episode_id = '{episode_id}' FORMAT JSONEachRow");

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json_rows: Vec<Value> = text
        .response
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if json_rows.is_empty() {
        None
    } else {
        Some(json_rows)
    }
}

pub async fn select_chat_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ChatInference WHERE id = '{inference_id}' LIMIT 1 FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_json_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    // We limit to 1 in case there are duplicate entries (can be caused by a race condition in polling batch inferences)
    let query = format!(
        "SELECT * FROM JsonInference WHERE id = '{inference_id}' LIMIT 1 FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_model_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    // We limit to 1 in case there are duplicate entries (can be caused by a race condition in polling batch inferences)
    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{inference_id}' LIMIT 1 FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_all_model_inferences_by_chat_episode_id_clickhouse(
    episode_id: Uuid,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
) -> Option<Vec<Value>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id IN (SELECT id FROM ChatInference WHERE episode_id = '{episode_id}') FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json_rows: Vec<Value> = text
        .response
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if json_rows.is_empty() {
        None
    } else {
        Some(json_rows)
    }
}

pub async fn select_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Vec<Value>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{inference_id}' FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json_rows: Vec<Value> = text
        .response
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if json_rows.is_empty() {
        None
    } else {
        Some(json_rows)
    }
}

pub async fn select_inference_tags_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    function_name: &str,
    tag_key: &str,
    tag_value: &str,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM InferenceTag WHERE function_name = '{function_name}' AND key = '{tag_key}' AND value = '{tag_value}' AND inference_id = '{inference_id}' FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_batch_model_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    let query = format!(
        r"
        SELECT bmi.*
        FROM BatchModelInference bmi
        INNER JOIN BatchIdByInferenceId bid ON bmi.inference_id = bid.inference_id
        WHERE bid.inference_id = '{inference_id}'
        FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    Some(serde_json::from_str(&text.response).unwrap())
}

pub async fn select_batch_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_id: Uuid,
) -> Option<Vec<Value>> {
    let query = format!(
        r"
        SELECT bmi.*
        FROM BatchModelInference bmi
        WHERE bmi.batch_id = '{batch_id}'
        FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json_rows: Vec<Value> = text
        .response
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    Some(json_rows)
}

pub async fn select_latest_batch_request_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_id: Uuid,
) -> Option<Value> {
    let query = format!(
        "SELECT * FROM BatchRequest WHERE batch_id = '{batch_id}' ORDER BY timestamp DESC LIMIT 1 FORMAT JSONEachRow"
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
    feedback_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!("SELECT * FROM {table_name} WHERE id = '{feedback_id}' FORMAT JSONEachRow");

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_by_target_id_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
    target_id: Uuid,
    metric_name: Option<&str>,
) -> Option<Value> {
    let query = match metric_name {
        Some(metric_name) => {
            format!(
                "SELECT * FROM {table_name} WHERE target_id = '{target_id}' AND metric_name = '{metric_name}' FORMAT JSONEachRow"
            )
        }
        None => {
            format!("SELECT * FROM {table_name} WHERE target_id = '{target_id}' FORMAT JSONEachRow")
        }
    };

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

pub async fn select_workflow_evaluation_run_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    run_id: Uuid,
) -> Option<WorkflowEvaluationRunRow> {
    let query = format!(
        "SELECT
            uint_to_uuid(run_id_uint) as run_id,
            variant_pins,
            tags,
            project_name,
            run_display_name
        FROM DynamicEvaluationRun
        WHERE run_id_uint = toUInt128(toUUID('{run_id}'))
        FORMAT JSONEachRow",
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();

    Some(serde_json::from_str(&text.response).unwrap())
}

pub async fn select_workflow_evaluation_run_episode_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    run_id: Uuid,
    episode_id: Uuid,
) -> Option<WorkflowEvaluationRunEpisodeRow> {
    let query = format!(
        "SELECT run_id, uint_to_uuid(episode_id_uint) as episode_id, variant_pins, datapoint_name AS task_name, tags FROM DynamicEvaluationRunEpisode WHERE run_id = '{run_id}' AND episode_id_uint = toUInt128(toUUID('{episode_id}')) FORMAT JSONEachRow",
    );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    Some(serde_json::from_str(&text.response).unwrap())
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_tags_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    metric_name: &str,
    tag_key: &str,
    tag_value: &str,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
            "SELECT * FROM FeedbackTag WHERE metric_name = '{metric_name}' AND key = '{tag_key}' AND value = '{tag_value}' FORMAT JSONEachRow"
        );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .expect("Failed to execute query in select_feedback_tags_clickhouse");
    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_tags_clickhouse_with_feedback_id(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    feedback_id: &str,
    metric_name: &str,
    tag_key: &str,
    tag_value: &str,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
            "SELECT * FROM FeedbackTag WHERE feedback_id = '{feedback_id}' AND metric_name = '{metric_name}' AND key = '{tag_key}' AND value = '{tag_value}' FORMAT JSONEachRow"
        );

    let text = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await
        .expect("Failed to execute query in select_feedback_tags_clickhouse_with_feedback_id");

    let json: Value = serde_json::from_str(&text.response).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_inference_evaluation_human_feedback_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    metric_name: &str,
    datapoint_id: Uuid,
    output: &str,
) -> Option<StaticEvaluationHumanFeedback> {
    let datapoint_id_str = datapoint_id.to_string();
    let escaped_output = escape_string_for_clickhouse_literal(output);
    let params = HashMap::from([
        ("metric_name", metric_name),
        ("datapoint_id", &datapoint_id_str),
        ("output", &escaped_output),
    ]);
    let query = r"
        SELECT * FROM StaticEvaluationHumanFeedback
        WHERE
            metric_name = {metric_name:String}
            AND datapoint_id = {datapoint_id:UUID}
            AND output = {output:String}
        FORMAT JSONEachRow"
        .to_string();
    let text = clickhouse_connection_info
        .run_query_synchronous(query, &params)
        .await
        .unwrap();
    if text.response.is_empty() {
        // Return None if the query returns no rows
        None
    } else {
        // Panic if the query fails to parse or multiple rows are returned
        let json: StaticEvaluationHumanFeedback = serde_json::from_str(&text.response).unwrap();
        Some(json)
    }
}
