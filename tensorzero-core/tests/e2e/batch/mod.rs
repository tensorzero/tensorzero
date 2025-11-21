use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tensorzero_core::config::{Config, ConfigLoadInfo};
use tensorzero_core::db::clickhouse::{ClickHouseConnectionInfo, TableName};
/// End-to-end tests for particular internal functionality in the batch inference endpoint
/// These are not tests of the public API (those should go in tests/e2e/providers/batch.rs)
use tensorzero_core::endpoints::batch_inference::{
    get_batch_inferences, get_batch_request, get_completed_batch_inference_response,
    write_batch_request_row, write_completed_batch_inference, write_poll_batch_inference,
    PollInferenceResponse, PollPathParams,
};
use tensorzero_core::endpoints::inference::{InferenceParams, InferenceResponse};
use tensorzero_core::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
use tensorzero_core::inference::types::batch::{
    BatchModelInferenceRow, BatchRequestRow, BatchStatus, PollBatchInferenceResponse,
    ProviderBatchInferenceOutput, ProviderBatchInferenceResponse, UnparsedBatchRequestRow,
};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, FinishReason, JsonInferenceOutput, StoredInput, Usage,
};
use tensorzero_core::jsonschema_util::StaticJSONSchema;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
    select_model_inferences_clickhouse,
};

/// Tests the `get_batch_request` function by first writing a batch inference to ClickHouse and reading with
#[tokio::test]
async fn test_get_batch_request() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"foo": "bar"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request";
    let raw_response = "raw response";
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        raw_request,
        raw_response,
        model_provider_name,
        status: BatchStatus::Pending,
        errors: vec![],
    });
    write_batch_request_row(&clickhouse, &batch_request)
        .await
        .unwrap();
    // Sleep a bit to ensure the write has propagated
    sleep(Duration::from_millis(200)).await;

    // First, let's query by batch ID
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let batch_request = get_batch_request(&clickhouse, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id);
    assert_eq!(batch_request.batch_params.into_owned(), batch_params);
    assert_eq!(batch_request.function_name, function_name);
    assert_eq!(batch_request.variant_name, variant_name);
    assert_eq!(&*batch_request.model_name, model_name);
    assert_eq!(&*batch_request.model_provider_name, model_provider_name);
    assert_eq!(batch_request.status, BatchStatus::Pending);
    assert_eq!(batch_request.raw_request, raw_request);
    assert_eq!(batch_request.raw_response, raw_response);

    // Next, we'll insert a BatchModelInferenceRow
    let inference_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let input = StoredInput {
        system: None,
        messages: vec![],
    };

    let row = BatchModelInferenceRow {
        batch_id,
        inference_id,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input,
        input_messages: vec![],
        system: None,
        tool_params: None,
        inference_params: Cow::Owned(InferenceParams::default()),
        output_schema: None,
        raw_request: Cow::Borrowed(""),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
    };
    let rows = vec![row];
    clickhouse
        .write_batched(&rows, TableName::BatchModelInference)
        .await
        .unwrap();
    // Sleep a bit to ensure the write has propagated
    sleep(Duration::from_millis(200)).await;

    // Now, let's query by inference ID
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id),
    };
    let batch_request = get_batch_request(&clickhouse, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id);
    assert_eq!(batch_request.function_name, function_name);
    assert_eq!(batch_request.variant_name, variant_name);
    assert_eq!(&*batch_request.model_name, model_name);
    assert_eq!(&*batch_request.model_provider_name, model_provider_name);
}

#[tokio::test]
async fn test_write_poll_batch_inference() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function2";
    let variant_name = "test_variant2";
    let model_name = "test_model2";
    let model_provider_name = "test_model_provider2";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let status = BatchStatus::Pending;
    let errors = vec![];
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors,
    });
    let ConfigLoadInfo { config, .. } = Config::new_empty().await.unwrap();

    // Write a pending batch
    let poll_inference_response = write_poll_batch_inference(
        &clickhouse,
        &batch_request,
        PollBatchInferenceResponse::Pending {
            raw_request: raw_request.clone(),
            raw_response: raw_response.clone(),
        },
        &config,
    )
    .await
    .unwrap();
    assert_eq!(poll_inference_response, PollInferenceResponse::Pending);
    sleep(Duration::from_millis(1200)).await;
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let batch_request = get_batch_request(&clickhouse, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id);
    assert_eq!(batch_request.status, BatchStatus::Pending);

    // Write a failed batch
    let status = BatchStatus::Failed;
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors: vec![],
    });
    let poll_inference_response = write_poll_batch_inference(
        &clickhouse,
        &batch_request,
        PollBatchInferenceResponse::Failed {
            raw_request: raw_request.clone(),
            raw_response: raw_response.clone(),
        },
        &config,
    )
    .await
    .unwrap();
    assert_eq!(poll_inference_response, PollInferenceResponse::Failed);

    sleep(Duration::from_millis(200)).await;
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    // This should return the failed batch as it is more recent
    let batch_request = get_batch_request(&clickhouse, &query).await.unwrap();
    assert_eq!(batch_request.batch_id, batch_id);
    assert_eq!(batch_request.status, BatchStatus::Failed);
}

/// Helper function to write 2 rows to the BatchModelInference table
async fn write_2_batch_model_inference_rows(
    clickhouse: &ClickHouseConnectionInfo,
    batch_id: Uuid,
) -> Vec<BatchModelInferenceRow<'static>> {
    let inference_id = Uuid::now_v7();
    let function_name = "test_function";
    let variant_name = "test_variant";
    let episode_id = Uuid::now_v7();
    let input = StoredInput {
        system: None,
        messages: vec![],
    };
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let row1 = BatchModelInferenceRow {
        batch_id,
        inference_id,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input: input.clone(),
        input_messages: vec![],
        system: None,
        tool_params: None,
        inference_params: Cow::Owned(InferenceParams::default()),
        output_schema: None,
        raw_request: Cow::Borrowed(""),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
    };
    let inference_id2 = Uuid::now_v7();
    let row2 = BatchModelInferenceRow {
        batch_id,
        inference_id: inference_id2,
        function_name: function_name.into(),
        variant_name: variant_name.into(),
        episode_id,
        input,
        input_messages: vec![],
        system: None,
        tool_params: None,
        inference_params: Cow::Owned(InferenceParams::default()),
        output_schema: None,
        raw_request: Cow::Borrowed(""),
        model_name: Cow::Borrowed(model_name),
        model_provider_name: Cow::Borrowed(model_provider_name),
        tags: HashMap::new(),
    };
    let rows = vec![row1, row2];
    clickhouse
        .write_batched(&rows, TableName::BatchModelInference)
        .await
        .unwrap();
    rows
}

#[tokio::test]
async fn test_get_batch_inferences() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let mut expected_batch_rows = write_2_batch_model_inference_rows(&clickhouse, batch_id).await;
    let other_batch_id = Uuid::now_v7();
    let other_batch_rows = write_2_batch_model_inference_rows(&clickhouse, other_batch_id).await;
    let mut batch_inferences = get_batch_inferences(
        &clickhouse,
        batch_id,
        &[
            expected_batch_rows[0].inference_id,
            expected_batch_rows[1].inference_id,
            other_batch_rows[0].inference_id,
            other_batch_rows[1].inference_id,
        ],
    )
    .await
    .unwrap();

    assert_eq!(batch_inferences.len(), 2);

    // Sort both arrays by inference_id to ensure matching order
    batch_inferences.sort_by_key(|row| row.inference_id);
    expected_batch_rows.sort_by_key(|row| row.inference_id);

    assert_eq!(batch_inferences[0], expected_batch_rows[0]);
    assert_eq!(batch_inferences[1], expected_batch_rows[1]);
}

/// Tests writing and reading a completed batch inference for a chat function
/// Exercises the write path in `write_completed_batch_inference` and the read path in `get_completed_batch_inference_response`
#[tokio::test]
async fn test_write_read_completed_batch_inference_chat() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let status = BatchStatus::Pending;
    let errors = vec![];
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        model_name,
        model_provider_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        status,
        errors,
    });
    let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
        variants: HashMap::new(),
        ..Default::default()
    }));
    let ConfigLoadInfo { mut config, .. } = Config::new_empty().await.unwrap();
    config.functions = HashMap::from([(function_name.to_string(), function_config)]);
    let batch_model_inference_rows =
        write_2_batch_model_inference_rows(&clickhouse, batch_id).await;
    let inference_id1 = batch_model_inference_rows[0].inference_id;
    let output_1 = ProviderBatchInferenceOutput {
        id: inference_id1,
        output: vec!["hello world".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
        },
        finish_reason: Some(FinishReason::Stop),
    };
    let inference_id2 = batch_model_inference_rows[1].inference_id;
    let output_2 = ProviderBatchInferenceOutput {
        id: inference_id2,
        output: vec!["goodbye world".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(20),
            output_tokens: Some(30),
        },
        finish_reason: Some(FinishReason::ToolCall),
    };
    let response = ProviderBatchInferenceResponse {
        elements: HashMap::from([(inference_id1, output_1), (inference_id2, output_2)]),
        raw_request: raw_request.clone(),
        raw_response: raw_response.clone(),
    };
    let mut inference_responses =
        write_completed_batch_inference(&clickhouse, &batch_request, response, &config)
            .await
            .unwrap();

    // Sort inferences by inference_id to ensure consistent ordering
    inference_responses.sort_by_key(tensorzero::InferenceResponse::inference_id);

    assert_eq!(inference_responses.len(), 2);
    let inference_response_1 = &inference_responses[0];
    let inference_response_2 = &inference_responses[1];

    match inference_response_1 {
        InferenceResponse::Chat(chat_inference_response) => {
            assert_eq!(chat_inference_response.inference_id, inference_id1);
            assert_eq!(
                chat_inference_response.finish_reason,
                Some(FinishReason::Stop)
            );
            match &chat_inference_response.content[0] {
                ContentBlockChatOutput::Text(text_block) => {
                    assert_eq!(text_block.text, "hello world");
                }
                _ => panic!("Unexpected content block type"),
            }
            assert_eq!(chat_inference_response.usage.input_tokens, Some(10));
            assert_eq!(chat_inference_response.usage.output_tokens, Some(20));
        }
        InferenceResponse::Json(_) => panic!("Unexpected inference response type"),
    }

    match inference_response_2 {
        InferenceResponse::Chat(chat_inference_response) => {
            assert_eq!(chat_inference_response.inference_id, inference_id2);
            match &chat_inference_response.content[0] {
                ContentBlockChatOutput::Text(text_block) => {
                    assert_eq!(text_block.text, "goodbye world");
                }
                _ => panic!("Unexpected content block type"),
            }
            assert_eq!(chat_inference_response.usage.input_tokens, Some(20));
            assert_eq!(chat_inference_response.usage.output_tokens, Some(30));
        }
        InferenceResponse::Json(_) => panic!("Unexpected inference response type"),
    }

    sleep(Duration::from_millis(200)).await;
    let chat_inference_1 = select_chat_inference_clickhouse(&clickhouse, inference_id1)
        .await
        .unwrap();
    let retrieved_inference_id1 = chat_inference_1["id"].as_str().unwrap();
    assert_eq!(retrieved_inference_id1, inference_id1.to_string());
    let retrieved_function_name = chat_inference_1["function_name"].as_str().unwrap();
    assert_eq!(retrieved_function_name, function_name);
    let retrieved_variant_name = chat_inference_1["variant_name"].as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);
    let chat_inference_2 = select_chat_inference_clickhouse(&clickhouse, inference_id2)
        .await
        .unwrap();
    let retrieved_inference_id2 = chat_inference_2["id"].as_str().unwrap();
    assert_eq!(retrieved_inference_id2, inference_id2.to_string());
    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id1)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let model_inference = &model_inferences[0];
    assert_eq!(model_inference["inference_id"], inference_id1.to_string());
    assert_eq!(model_inference["model_name"], model_name);
    assert_eq!(model_inference["model_provider_name"], model_provider_name);

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id2)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let model_inference = &model_inferences[0];
    assert_eq!(model_inference["inference_id"], inference_id2.to_string());
    assert_eq!(model_inference["model_name"], model_name);
    assert_eq!(model_inference["model_provider_name"], model_provider_name);

    // Now, let's read this using `get_completed_batch_inference_response`
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &clickhouse,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(completed_inference_response.batch_id, batch_id);
    assert_eq!(completed_inference_response.inferences.len(), 2);

    // Create HashMaps keyed by inference_id for both sets of inferences
    let completed_inferences: HashMap<_, _> = completed_inference_response
        .inferences
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();
    let original_inferences: HashMap<_, _> = [inference_response_1, inference_response_2]
        .into_iter()
        .map(|r| (r.inference_id(), r))
        .collect();

    // Compare the HashMaps
    assert_eq!(completed_inferences, original_inferences);

    // Now let's read using `get_completed_batch_inference_response` with a `PollInferenceQuery::Inference`
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id1),
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &clickhouse,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        &completed_inference_response.inferences[0],
        original_inferences[&inference_id1]
    );
}

#[tokio::test]
/// Tests writing and reading a completed batch inference for a JSON function
/// Exercises the write path in `write_completed_batch_inference` and the read path in `get_completed_batch_inference_response`
async fn test_write_read_completed_batch_inference_json() {
    let clickhouse = get_clickhouse().await;
    let batch_id = Uuid::now_v7();
    let batch_params = json!({"baz": "bat"});
    let function_name = "test_function";
    let variant_name = "test_variant";
    let model_name = "test_model";
    let model_provider_name = "test_model_provider";
    let raw_request = "raw request";
    let raw_response = "raw response";
    let status = BatchStatus::Pending;
    let batch_request = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id,
        batch_params: &batch_params,
        function_name,
        variant_name,
        raw_request,
        raw_response,
        model_name,
        model_provider_name,
        status,
        errors: vec![],
    });
    let output_schema = StaticJSONSchema::from_value(json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string"
            }
        },
        "required": ["answer"]
    }))
    .unwrap();
    let function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
        variants: HashMap::new(),
        output_schema,
        ..Default::default()
    }));
    let ConfigLoadInfo { mut config, .. } = Config::new_empty().await.unwrap();
    config.functions = HashMap::from([(function_name.to_string(), function_config)]);
    let batch_model_inference_rows =
        write_2_batch_model_inference_rows(&clickhouse, batch_id).await;
    let inference_id1 = batch_model_inference_rows[0].inference_id;
    let output_1 = ProviderBatchInferenceOutput {
        id: inference_id1,
        output: vec!["{\"answer\": \"hello world\"}".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
        },
        finish_reason: Some(FinishReason::Stop),
    };
    let inference_id2 = batch_model_inference_rows[1].inference_id;
    let output_2 = ProviderBatchInferenceOutput {
        id: inference_id2,
        output: vec!["{\"response\": \"goodbye world\"}".to_string().into()],
        raw_response: String::new(),
        usage: Usage {
            input_tokens: Some(20),
            output_tokens: Some(30),
        },
        finish_reason: Some(FinishReason::ToolCall),
    };
    let raw_request = "raw request".to_string();
    let raw_response = "raw response".to_string();
    let response = ProviderBatchInferenceResponse {
        elements: HashMap::from([(inference_id1, output_1), (inference_id2, output_2)]),
        raw_request,
        raw_response,
    };
    let inference_responses =
        write_completed_batch_inference(&clickhouse, &batch_request, response, &config)
            .await
            .unwrap();

    assert_eq!(inference_responses.len(), 2);

    // Create a map of inferences by inference_id for easier lookup
    let inferences_by_id: HashMap<_, _> = inference_responses
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();

    // Now we can safely access each response by its ID
    let response_1 = inferences_by_id
        .get(&inference_id1)
        .expect("Missing response for inference_id1");
    let response_2 = inferences_by_id
        .get(&inference_id2)
        .expect("Missing response for inference_id2");

    match response_1 {
        InferenceResponse::Json(json_inference_response) => {
            assert_eq!(json_inference_response.inference_id, inference_id1);
            assert_eq!(
                json_inference_response.output.raw,
                Some("{\"answer\": \"hello world\"}".to_string())
            );
            assert_eq!(
                json_inference_response.output.parsed.as_ref().unwrap()["answer"],
                "hello world"
            );
            assert_eq!(json_inference_response.usage.input_tokens, Some(10));
            assert_eq!(
                json_inference_response.finish_reason,
                Some(FinishReason::Stop)
            );
        }
        InferenceResponse::Chat(_) => panic!("Unexpected inference response type"),
    }

    match response_2 {
        InferenceResponse::Json(json_inference_response) => {
            assert_eq!(json_inference_response.inference_id, inference_id2);
            assert_eq!(
                json_inference_response.output.raw,
                Some("{\"response\": \"goodbye world\"}".to_string())
            );
            assert!(json_inference_response.output.parsed.is_none());
            assert_eq!(
                json_inference_response.finish_reason,
                Some(FinishReason::ToolCall)
            );
        }
        InferenceResponse::Chat(_) => panic!("Unexpected inference response type"),
    }

    sleep(Duration::from_millis(200)).await;
    let json_inference_1 = select_json_inference_clickhouse(&clickhouse, inference_id1)
        .await
        .unwrap();
    let retrieved_inference_id1 = json_inference_1["id"].as_str().unwrap();
    assert_eq!(retrieved_inference_id1, inference_id1.to_string());
    let retrieved_function_name = json_inference_1["function_name"].as_str().unwrap();
    assert_eq!(retrieved_function_name, function_name);
    let retrieved_variant_name = json_inference_1["variant_name"].as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);
    let retrieved_output_1 = json_inference_1["output"].as_str().unwrap();
    let retrieved_output_1_json: JsonInferenceOutput =
        serde_json::from_str(retrieved_output_1).unwrap();
    assert_eq!(
        retrieved_output_1_json.parsed.unwrap()["answer"],
        "hello world"
    );
    assert_eq!(
        retrieved_output_1_json.raw,
        Some("{\"answer\": \"hello world\"}".to_string())
    );
    let json_inference_2 = select_json_inference_clickhouse(&clickhouse, inference_id2)
        .await
        .unwrap();
    let retrieved_inference_id2 = json_inference_2["id"].as_str().unwrap();
    assert_eq!(retrieved_inference_id2, inference_id2.to_string());
    let retrieved_output_2 = json_inference_2["output"].as_str().unwrap();
    let retrieved_output_2_json: JsonInferenceOutput =
        serde_json::from_str(retrieved_output_2).unwrap();
    assert!(retrieved_output_2_json.parsed.is_none());
    assert_eq!(
        retrieved_output_2_json.raw,
        Some("{\"response\": \"goodbye world\"}".to_string())
    );
    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id1)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let model_inference = &model_inferences[0];
    assert_eq!(model_inference["inference_id"], inference_id1.to_string());
    assert_eq!(model_inference["model_name"], model_name);
    assert_eq!(model_inference["model_provider_name"], model_provider_name);
    assert_eq!(
        model_inference["output"],
        "[{\"type\":\"text\",\"text\":\"{\\\"answer\\\": \\\"hello world\\\"}\"}]"
    );

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id2)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let model_inference = &model_inferences[0];
    assert_eq!(model_inference["inference_id"], inference_id2.to_string());
    assert_eq!(model_inference["model_name"], model_name);
    assert_eq!(model_inference["model_provider_name"], model_provider_name);
    assert_eq!(
        model_inference["output"],
        "[{\"type\":\"text\",\"text\":\"{\\\"response\\\": \\\"goodbye world\\\"}\"}]"
    );

    // Now, let's read this using `get_completed_batch_inference_response`
    let query = PollPathParams {
        batch_id,
        inference_id: None,
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &clickhouse,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(completed_inference_response.batch_id, batch_id);
    assert_eq!(completed_inference_response.inferences.len(), 2);

    // Create HashMaps keyed by inference_id for both sets of inferences
    let completed_inferences: HashMap<_, _> = completed_inference_response
        .inferences
        .iter()
        .map(|r| (r.inference_id(), r))
        .collect();
    let original_inferences: HashMap<_, _> = [response_1, response_2]
        .into_iter()
        .map(|r| (r.inference_id(), *r))
        .collect();

    // Compare the HashMaps
    assert_eq!(completed_inferences, original_inferences);

    // Now let's read using `get_completed_batch_inference_response` with a `PollInferenceQuery::Inference`
    let query = PollPathParams {
        batch_id,
        inference_id: Some(inference_id1),
    };
    let completed_inference_response = get_completed_batch_inference_response(
        &clickhouse,
        &batch_request,
        &query,
        &config.functions[function_name],
    )
    .await
    .unwrap();
    assert_eq!(
        &completed_inference_response.inferences[0],
        original_inferences[&inference_id1]
    );
}
