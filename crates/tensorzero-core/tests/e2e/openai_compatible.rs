#![expect(clippy::print_stdout)]

use std::collections::HashSet;

use googletest::prelude::*;
use tensorzero::ClientExt;

use axum::extract::State;
use http_body_util::BodyExt;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::openai_compatible::OpenAIStructuredJson;
use tensorzero_core::endpoints::openai_compatible::chat_completions::chat_completions_handler;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_route_new_format() {
    Box::pin(test_openai_compatible_route_with_function_name_as_model(
        "tensorzero::function_name::basic_test_no_system_schema",
    ))
    .await;
}

async fn test_openai_compatible_route_with_function_name_as_model(model: &str) {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = chat_completions_handler(
        State(state),
        None,
        OpenAIStructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "TensorBot"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Japan?"
                    }
                ],
                "stream": false,
                "tensorzero::tags": {
                    "foo": "bar"
                },
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    println!("response: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(
        content,
        "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(
        response_model,
        "tensorzero::function_name::basic_test_no_system_schema::variant_name::test"
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check Inference table
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    assert_eq!(chat.function_name, "basic_test_no_system_schema");
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let tags = &chat.tags;
    assert_eq!(tags.get("foo").unwrap(), "bar");
    assert_eq!(tags.len(), 1);
    let output = chat.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    let output_value = serde_json::to_value(&output[0]).unwrap();
    assert_eq!(output_value.get("type").unwrap().as_str().unwrap(), "text");
    assert_eq!(output_value.get("text").unwrap().as_str().unwrap(), content);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "test");
    // Check the processing time
    assert!(chat.processing_time_ms.is_some());

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    println!("ModelInference result: {mi:?}");
    assert_eq!(mi.inference_id, inference_id);
    assert_eq!(mi.model_name, "test");
    assert_eq!(mi.model_provider_name, "good");
    assert_eq!(mi.raw_request.as_deref().unwrap(), "raw request");
    assert!(mi.input_tokens.unwrap() > 5);
    assert!(mi.output_tokens.unwrap() > 0);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    let raw_response = mi.raw_response.as_deref().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
    assert_eq!(
        mi.finish_reason,
        Some(tensorzero_core::inference::types::FinishReason::Stop)
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_matches_response_fields() {
    let client = Client::new();

    let tensorzero_payload = json!({
        "model": "tensorzero::model_name::openai::gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
    });

    let openai_payload = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
    });

    let tensorzero_response_fut = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&tensorzero_payload)
        .send();

    let openai_response_fut = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(std::env::var("OPENAI_API_KEY").unwrap())
        .json(&openai_payload)
        .send();

    let (tensorzero_response, openai_response) =
        tokio::try_join!(tensorzero_response_fut, openai_response_fut).unwrap();

    assert_that!(
        tensorzero_response.status(),
        eq(StatusCode::OK),
        "TensorZero request failed"
    );
    assert_that!(
        openai_response.status(),
        eq(StatusCode::OK),
        "OpenAI request failed"
    );

    let openai_json: serde_json::Value = openai_response.json().await.unwrap();
    let tensorzero_json: serde_json::Value = tensorzero_response.json().await.unwrap();

    let openai_keys: HashSet<_> = openai_json.as_object().unwrap().keys().collect();
    let tensorzero_keys: HashSet<_> = tensorzero_json.as_object().unwrap().keys().collect();

    let missing_keys: Vec<_> = openai_keys.difference(&tensorzero_keys).collect();
    expect_that!(
        missing_keys.is_empty(),
        eq(true),
        "Missing keys in TensorZero response: {missing_keys:?}"
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_dryrun() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::json",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::dryrun": true
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert_that!(choices.len(), eq(1));
    let choice = choices.first().unwrap();
    expect_that!(choice.get("index").unwrap().as_u64().unwrap(), eq(0));
    let message = choice.get("message").unwrap();
    expect_that!(
        message.get("role").unwrap().as_str().unwrap(),
        eq("assistant")
    );
    let content = message.get("content").unwrap().as_str().unwrap();
    expect_that!(content, eq("{\"answer\":\"Hello\"}"));
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    expect_that!(finish_reason, eq("stop"));
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    expect_that!(response_model, eq("tensorzero::model_name::json"));

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // No inference should be written to the database when dryrun is true
    let config = get_e2e_config().await;
    let chat_inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    expect_that!(
        chat_inferences.is_empty(),
        eq(true),
        "No inference should be written when dryrun is true"
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_route_model_name_shorthand() {
    test_openai_compatible_route_with_default_function("tensorzero::model_name::dummy::good", "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.").await;
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_route_model_name_toml() {
    test_openai_compatible_route_with_default_function(
        "tensorzero::model_name::json",
        "{\"answer\":\"Hello\"}",
    )
    .await;
}

async fn test_openai_compatible_route_with_default_function(
    prefixed_model_name: &str,
    expected_content: &str,
) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": prefixed_model_name,
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "tensorzero::episode_id": episode_id.to_string(),
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, expected_content);
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(response_model, prefixed_model_name);

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check Inference table
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    assert_eq!(chat.function_name, "tensorzero::default");
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let output = chat.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    let output_value = serde_json::to_value(&output[0]).unwrap();
    assert_eq!(output_value.get("type").unwrap().as_str().unwrap(), "text");
    assert_eq!(output_value.get("text").unwrap().as_str().unwrap(), content);
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the processing time
    assert!(chat.processing_time_ms.is_some());

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    let mi = &model_inferences[0];
    assert_eq!(mi.inference_id, inference_id);
    assert_eq!(
        mi.model_name,
        prefixed_model_name
            .strip_prefix("tensorzero::model_name::")
            .unwrap()
    );
    assert_eq!(mi.raw_request.as_deref().unwrap(), "raw request");
    assert!(mi.input_tokens.unwrap() > 5);
    assert!(mi.output_tokens.unwrap() > 0);
    assert!(mi.response_time_ms.unwrap() > 0);
    assert!(mi.ttft_ms.is_none());
    let raw_response = mi.raw_response.as_deref().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
    assert_eq!(
        mi.finish_reason,
        Some(tensorzero_core::inference::types::FinishReason::Stop)
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_route_bad_model_name() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::my_missing_model",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": {
            "message": "Invalid inference target: Invalid model name: Model name 'my_missing_model' not found in model table",
            "error_json": {
                "InvalidInferenceTarget": {
                    "message": "Invalid model name: Model name 'my_missing_model' not found in model table"
                }
            },
            "tensorzero_error_json": {
                "InvalidInferenceTarget": {
                    "message": "Invalid model name: Model name 'my_missing_model' not found in model table"
                }
            }
        }
    });
    expect_that!(response_json, eq(&expected));
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_route_with_json_mode_on() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::basic_test_no_system_schema",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "response_format":{"type":"json_object"},
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert_that!(choices.len(), eq(1));
    let choice = choices.first().unwrap();
    expect_that!(choice.get("index").unwrap().as_u64().unwrap(), eq(0));
    let message = choice.get("message").unwrap();
    expect_that!(
        message.get("role").unwrap().as_str().unwrap(),
        eq("assistant")
    );
    let content = message.get("content").unwrap().as_str().unwrap();
    expect_that!(
        content,
        eq(
            "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        )
    );
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    expect_that!(
        response_model,
        eq("tensorzero::function_name::basic_test_no_system_schema::variant_name::test")
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check Inference table
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(chat.inference_id, eq(inference_id));
    expect_that!(&chat.function_name, eq("basic_test_no_system_schema"));
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    expect_that!(input, eq(&correct_input));
    let output = chat.output.as_ref().unwrap();
    assert_that!(output.len(), eq(1));
    let output_value = serde_json::to_value(&output[0]).unwrap();
    expect_that!(
        output_value.get("type").unwrap().as_str().unwrap(),
        eq("text")
    );
    expect_that!(
        output_value.get("text").unwrap().as_str().unwrap(),
        eq(content)
    );
    expect_that!(chat.episode_id, eq(episode_id));
    expect_that!(&chat.variant_name, eq("test"));
    expect_that!(&chat.processing_time_ms, some(anything()));
    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    expect_that!(json_mode, eq("on"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(1));
    let mi = &model_inferences[0];
    expect_that!(mi.inference_id, eq(inference_id));
    expect_that!(&mi.model_name, eq("test"));
    expect_that!(&mi.model_provider_name, eq("good"));
    expect_that!(mi.raw_request.as_deref().unwrap(), eq("raw request"));
    expect_that!(mi.input_tokens.unwrap(), gt(5));
    expect_that!(mi.output_tokens.unwrap(), gt(0));
    expect_that!(mi.response_time_ms.unwrap(), gt(0));
    expect_that!(&mi.ttft_ms, none());
    let raw_response = mi.raw_response.as_deref().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_route_with_json_schema() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::basic_test_no_system_schema",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "response_format":{"type":"json_schema", "json_schema":{"name":"test", "strict":true, "schema":{}}}
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert_that!(choices.len(), eq(1));
    let choice = choices.first().unwrap();
    expect_that!(choice.get("index").unwrap().as_u64().unwrap(), eq(0));
    let message = choice.get("message").unwrap();
    expect_that!(
        message.get("role").unwrap().as_str().unwrap(),
        eq("assistant")
    );
    let content = message.get("content").unwrap().as_str().unwrap();
    expect_that!(
        content,
        eq(
            "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        )
    );
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    expect_that!(finish_reason, eq("stop"));
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    expect_that!(
        response_model,
        eq("tensorzero::function_name::basic_test_no_system_schema::variant_name::test")
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check Inference table
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(chat.inference_id, eq(inference_id));
    expect_that!(&chat.function_name, eq("basic_test_no_system_schema"));
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    expect_that!(input, eq(&correct_input));
    let output = chat.output.as_ref().unwrap();
    assert_that!(output.len(), eq(1));
    let output_value = serde_json::to_value(&output[0]).unwrap();
    expect_that!(
        output_value.get("type").unwrap().as_str().unwrap(),
        eq("text")
    );
    expect_that!(
        output_value.get("text").unwrap().as_str().unwrap(),
        eq(content)
    );
    expect_that!(chat.episode_id, eq(episode_id));
    expect_that!(&chat.variant_name, eq("test"));
    expect_that!(&chat.processing_time_ms, some(anything()));
    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    expect_that!(json_mode, eq("strict"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(1));
    let mi = &model_inferences[0];
    expect_that!(mi.inference_id, eq(inference_id));
    expect_that!(&mi.model_name, eq("test"));
    expect_that!(&mi.model_provider_name, eq("good"));
    expect_that!(mi.raw_request.as_deref().unwrap(), eq("raw request"));
    expect_that!(mi.input_tokens.unwrap(), gt(5));
    expect_that!(mi.output_tokens.unwrap(), gt(0));
    expect_that!(mi.response_time_ms.unwrap(), gt(0));
    expect_that!(&mi.ttft_ms, none());
    let raw_response = mi.raw_response.as_deref().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_streaming_tool_call() {
    use futures::StreamExt;
    use reqwest_sse_stream::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let body = json!({
        "stream": true,
        "stream_options": {
            "include_usage": true
        },
        "model": "tensorzero::model_name::openai::gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Boston today?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = response.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    expect_that!(found_done_chunk, eq(true));
    let first_chunk = chunks.first().unwrap();
    let parsed_chunk: Value = serde_json::from_str(first_chunk).unwrap();
    expect_that!(parsed_chunk["choices"][0]["index"].as_i64().unwrap(), eq(0));
    expect_that!(
        parsed_chunk["choices"][0]["delta"]["role"]
            .as_str()
            .unwrap(),
        eq("assistant")
    );
    expect_that!(parsed_chunk["choices"][0]["delta"].get("content"), none());
    println!("parsed_chunk: {parsed_chunk:?}");
    let tool_calls = parsed_chunk["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_that!(tool_calls.len(), eq(1));
    let tool_call = tool_calls[0].as_object().unwrap();
    expect_that!(tool_call["index"].as_i64().unwrap(), eq(0));
    expect_that!(
        tool_call["function"]["name"].as_str().unwrap(),
        eq("get_current_weather")
    );
    expect_that!(tool_call["function"]["arguments"].as_str().unwrap(), eq(""));
    for (i, chunk) in chunks.iter().enumerate() {
        let parsed_chunk: Value = serde_json::from_str(chunk).unwrap();
        if let Some(tool_calls) = parsed_chunk["choices"][0]["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let index = tool_call["index"].as_i64().unwrap();
                expect_that!(index, eq(0));
            }
        }
        if let Some(finish_reason) = parsed_chunk["choices"][0]["finish_reason"].as_str() {
            expect_that!(finish_reason, eq("tool_calls"));
            expect_that!(
                i,
                eq(chunks.len() - 1),
                "finish_reason should be on final chunk"
            );
        }
        if i == chunks.len() - 1 {
            let usage = parsed_chunk["usage"].as_object().unwrap();
            expect_that!(usage["prompt_tokens"].as_i64().unwrap(), gt(0));
            expect_that!(usage["completion_tokens"].as_i64().unwrap(), gt(0));
        }
        let response_model = parsed_chunk.get("model").unwrap().as_str().unwrap();
        expect_that!(response_model, eq("tensorzero::model_name::openai::gpt-4o"));
    }
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_warn_unknown_fields() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().clone();
    chat_completions_handler(
        State(state),
        None,
        OpenAIStructuredJson(
            serde_json::from_value(serde_json::json!({
                "messages": [],
                "model": "tensorzero::model_name::dummy::good",
                "my_fake_param": "fake_value"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    expect_that!(
        logs_contain("Ignoring unknown fields in OpenAI-compatible request: [\"my_fake_param\"]"),
        eq(true)
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_deny_unknown_fields() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().clone();
    let err = chat_completions_handler(
        State(state),
        None,
        OpenAIStructuredJson(
            serde_json::from_value(serde_json::json!({
                "messages": [],
                "model": "tensorzero::model_name::dummy::good",
                "tensorzero::deny_unknown_fields": true,
                "my_fake_param": "fake_value",
                "my_other_fake_param": "fake_value_2"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap_err();
    expect_that!(
        err.to_string(),
        eq(
            "Invalid request to OpenAI-compatible endpoint: `tensorzero::deny_unknown_fields` is set to true, but found unknown fields in the request: [my_fake_param, my_other_fake_param]"
        )
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_streaming() {
    use futures::StreamExt;
    use reqwest_sse_stream::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let body = json!({
        "stream": true,
        "model": "tensorzero::model_name::openai::gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "What's the reason for why we use AC not DC?"
            }
        ],
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = response.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    expect_that!(found_done_chunk, eq(true));
    let first_chunk = chunks.first().unwrap();
    let parsed_chunk: Value = serde_json::from_str(first_chunk).unwrap();
    expect_that!(parsed_chunk["choices"][0]["index"].as_i64().unwrap(), eq(0));
    expect_that!(
        parsed_chunk["choices"][0]["delta"]["role"]
            .as_str()
            .unwrap(),
        eq("assistant")
    );
    let _content = parsed_chunk["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap();
    expect_that!(
        parsed_chunk["choices"][0]["delta"].get("tool_calls"),
        none()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let parsed_chunk: Value = serde_json::from_str(chunk).unwrap();
        expect_that!(
            parsed_chunk["choices"][0]["delta"].get("tool_calls"),
            none()
        );
        if i < chunks.len() - 2 {
            let _content = parsed_chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap();
        }
        expect_that!(parsed_chunk["service_tier"].is_null(), eq(true));
        expect_that!(parsed_chunk["choices"][0]["logprobs"].is_null(), eq(true));
        if let Some(finish_reason) = parsed_chunk["choices"][0]["delta"]["finish_reason"].as_str() {
            expect_that!(finish_reason, eq("stop"));
            expect_that!(i, eq(chunks.len() - 2));
        }

        let response_model = parsed_chunk.get("model").unwrap().as_str().unwrap();
        expect_that!(response_model, eq("tensorzero::model_name::openai::gpt-4o"));
    }
}

// Test using 'stop' parameter in the openai-compatible endpoint
#[gtest]
#[tokio::test]
async fn test_openai_compatible_stop_sequence() {
    let client = Client::new();

    let payload = json!({
        "model": "tensorzero::model_name::anthropic::claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": "Output 'Hello world' followed by either '0' or '1'. Do not output anything else"
            }
        ],
        "stop": ["0", "1"],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json}");
    let finish_reason = response_json["choices"][0]["finish_reason"]
        .as_str()
        .unwrap();
    expect_that!(finish_reason, eq("stop"));
    let output = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    expect_that!(
        output.contains("Hello"),
        eq(true),
        "Unexpected output: {output}"
    );
    expect_that!(
        !output.contains("zero") && !output.contains("one"),
        eq(true),
        "Unexpected output: {output}"
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_file_with_custom_filename() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = chat_completions_handler(
        State(state),
        None,
        OpenAIStructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this file?"
                            },
                            {
                                "type": "file",
                                "file": {
                                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                                    "filename": "myfile.pdf"
                                }
                            }
                        ]
                    }
                ],
                "stream": false,
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    // Check Response is OK
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    // Verify the input was stored correctly with the custom filename
    let input = serde_json::to_value(&chat.input).unwrap();

    // Check that the file content block has the custom filename
    let messages = input.get("messages").unwrap().as_array().unwrap();
    assert_that!(messages.len(), eq(1));
    let content = messages[0].get("content").unwrap().as_array().unwrap();
    assert_that!(content.len(), eq(2));

    // Second content block should be the file
    let file_block = &content[1];
    expect_that!(
        file_block.get("type").unwrap().as_str().unwrap(),
        eq("file")
    );

    // Verify filename is present in the stored file (fields are at top level, not nested)
    expect_that!(
        file_block.get("filename").unwrap().as_str().unwrap(),
        eq("myfile.pdf")
    );
}

#[gtest]
#[tokio::test]
async fn test_openai_compatible_parallel_tool_calls_multi_turn() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // First request: Get parallel tool calls
    let body = json!({
        "stream": false,
        "model": "tensorzero::function_name::weather_helper_parallel",
        "messages": [
            { "role": "system", "content": [{"type": "tensorzero::template", "name": "system", "arguments": {"assistant_name": "Dr.Mehta"}}]},
            {
                "role": "user",
                "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            }
        ],
        "parallel_tool_calls": true,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::variant_name": "gpt-5-mini",
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();

    println!("First request response: {response_json:#?}");

    // Extract inference_id from response
    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Wait for data to be written to the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Validate Inference table
    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    println!("ChatInference: {chat:#?}");

    // Validate ModelInference table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(1));
    let mi = &model_inferences[0];
    println!("ModelInference: {mi:#?}");

    // Extract tool calls from response
    let first_message = response_json["choices"][0]["message"].clone();
    let tool_calls = first_message["tool_calls"].as_array().unwrap();
    assert_that!(tool_calls.len(), eq(2));

    // Build messages with tool results (one tool message per tool call)
    let mut messages = vec![
        json!({ "role": "system", "content": [{"type": "tensorzero::template", "name": "system", "arguments": {"assistant_name": "Dr.Mehta"}}]}),
        json!({
            "role": "user",
            "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
        }),
        first_message.clone(),
    ];

    // Add one tool message for each tool call
    for tool_call in tool_calls {
        let tool_id = tool_call["id"].as_str().unwrap();
        let tool_name = tool_call["function"]["name"].as_str().unwrap();

        let result_content = match tool_name {
            "get_temperature" => "22°C",
            "get_humidity" => "65%",
            _ => panic!("Unexpected tool: {tool_name}"),
        };

        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result_content
        }));
    }

    // Second request: Submit tool results and get final response
    let second_body = json!({
        "stream": false,
        "model": "tensorzero::function_name::weather_helper_parallel",
        "messages": messages,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::variant_name": "openai",
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&second_body)
        .send()
        .await
        .unwrap();

    assert_that!(response.status(), eq(StatusCode::OK));
    let final_response_json = response.json::<Value>().await.unwrap();

    println!("Final response: {final_response_json:#?}");

    // Validate final response
    let final_choice = &final_response_json["choices"][0];
    let finish_reason = final_choice["finish_reason"].as_str().unwrap();
    // Should be "stop" (normal completion) not "tool_calls" since we provided results
    expect_that!(finish_reason, eq("stop"));

    // Should have text content in the response
    let content = final_choice["message"]["content"].as_str();
    assert_that!(&content, some(anything()));
    expect_that!(content.unwrap().is_empty(), eq(false));

    // Should not have tool_calls in final response
    expect_that!(
        final_choice["message"].get("tool_calls"),
        none(),
        "Expected no tool_calls field in final response"
    );

    println!(
        "Multi-turn test passed! Got final response: {}",
        content.unwrap()
    );
}
