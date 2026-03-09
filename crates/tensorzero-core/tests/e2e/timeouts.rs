use std::time::{Duration, Instant};

use crate::common::get_gateway_endpoint;
use googletest::prelude::*;
use http::StatusCode;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero::{ClientInferenceParams, Input, InputMessage, InputMessageContent, Role};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::Text;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;
use tokio_stream::StreamExt;
use uuid::Uuid;

// Variant timeout tests

#[gtest]
#[tokio::test]
async fn test_variant_timeout_non_streaming() {
    let payload = json!({
        "function_name": "basic_test_variant_timeout",
        "variant_name": "slow_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "Variant `slow_timeout` timed out due to configured `non_streaming.total_ms` timeout (400ms)",
        "error_json": {
            "VariantTimeout": {
                "variant_name": "slow_timeout",
                "timeout": {
                    "secs": 0,
                    "nanos": 400000000
                },
                "kind": "non_streaming_total"
            }
        }
    });
    expect_that!(response_json, eq(&expected));
}

#[gtest]
#[tokio::test]
async fn test_variant_timeout_streaming() {
    let payload = json!({
        "function_name": "basic_test_variant_timeout",
        "variant_name": "slow_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "Variant `slow_timeout` timed out due to configured `streaming.ttft_ms` timeout (500ms)",
        "error_json": {
            "VariantTimeout": {
                "variant_name": "slow_timeout",
                "timeout": {
                    "secs": 0,
                    "nanos": 500000000
                },
                "kind": "streaming_ttft"
            }
        }
    });
    expect_that!(response_json, eq(&expected));
    expect_that!(status, eq(StatusCode::REQUEST_TIMEOUT));
}

#[gtest]
#[tokio::test]
async fn test_variant_timeout_slow_second_chunk_streaming() {
    slow_second_chunk_streaming(json!({
        "function_name": "basic_test_variant_timeout",
        "variant_name": "slow_second_chunk",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    }))
    .await;
}

#[gtest]
#[tokio::test]
async fn test_variant_total_timeout_streaming() {
    streaming_total_timeout(json!({
        "function_name": "basic_test_variant_timeout",
        "variant_name": "slow_second_chunk_total_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    }))
    .await;
}

#[gtest]
#[tokio::test]
async fn test_chat_inference_ttft_ms() {
    test_inference_ttft_ms("inference_ttft_chat").await;
}

#[gtest]
#[tokio::test]
async fn test_json_inference_ttft_ms() {
    test_inference_ttft_ms("inference_ttft_json").await;
}

async fn test_inference_ttft_ms(function_name: &str) {
    let payload = json!({
        "function_name": function_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut inference_id = None;

    while let Some(event) = response.next().await {
        let chunk = event.unwrap();
        println!("chunk: {chunk:?}");
        if let Event::Message(event) = chunk {
            if event.data == "[DONE]" {
                break;
            }
            let event = serde_json::from_str::<Value>(&event.data).unwrap();
            inference_id = Some(event["inference_id"].as_str().unwrap().parse().unwrap());
        }
    }

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inference_id = inference_id.unwrap();
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
    assert_eq!(inferences.len(), 1, "Expected exactly one inference");

    // The inference level-TTFT should be high, due to the timeout for the first provider
    let ttft_ms = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c.ttft_ms,
        StoredInferenceDatabase::Json(j) => j.ttft_ms,
    };
    assert!(ttft_ms.unwrap() > 500, "Unexpected ttft_ms: {ttft_ms:?}");

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    // The first provider will time out, so we should only have one inference
    assert_eq!(model_inferences.len(), 1);
    println!("model_inferences: {model_inferences:?}");
    assert_eq!(model_inferences[0].model_name, "first_provider_timeout");
    assert_eq!(model_inferences[0].model_provider_name, "second_good");
    // The model inference should have a small TTFT, since the successful model provider
    // is fast
    assert!(
        model_inferences[0].ttft_ms.unwrap() < 100,
        "Unexpected ttft ms in model provider"
    );
}

// We don't currently support setting an actual variant as the judge,
// so we can't apply a variant timeout to the judge itself.

// Test that if a candidate times out, the evaluator can still see the other candidates
#[gtest]
#[tokio::test]
async fn test_variant_timeout_best_of_n_other_candidate() {
    best_of_n_other_candidate(json!({
        "function_name": "basic_test_variant_timeout",
        "variant_name": "best_of_n",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    }))
    .await;
}

// Model provider timeout tests

#[gtest]
#[tokio::test]
async fn test_model_provider_timeout_non_streaming() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `non_streaming.total_ms` timeout (400ms)",
        "error_json": {
            "AllModelProvidersFailed": {
                "provider_errors": {
                    "slow": {
                        "ModelProviderTimeout": {
                            "provider_name": "slow",
                            "timeout": {
                                "secs": 0,
                                "nanos": 400000000
                            },
                            "kind": "non_streaming_total"
                        }
                    }
                }
            }
        }
    });
    expect_that!(response_json, eq(&expected));
}

#[gtest]
#[tokio::test]
async fn test_model_provider_timeout_streaming() {
    let payload = json!({
        "function_name": "basic_test_timeout",
        "variant_name": "timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `streaming.ttft_ms` timeout (500ms)",
        "error_json": {
            "AllModelProvidersFailed": {
                "provider_errors": {
                    "slow": {
                        "ModelProviderTimeout": {
                            "provider_name": "slow",
                            "timeout": {
                                "secs": 0,
                                "nanos": 500000000
                            },
                            "kind": "streaming_ttft"
                        }
                    }
                }
            }
        }
    });
    expect_that!(response_json, eq(&expected));
    expect_that!(status, eq(StatusCode::INTERNAL_SERVER_ERROR));
}

#[gtest]
#[tokio::test]
async fn test_model_provider_timeout_slow_second_chunk_streaming() {
    slow_second_chunk_streaming(json!({
        "function_name": "basic_test_timeout",
        "variant_name": "slow_second_chunk",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    }))
    .await;
}

#[gtest]
#[tokio::test]
async fn test_model_provider_total_timeout_streaming() {
    streaming_total_timeout(json!({
        "function_name": "basic_test_timeout",
        "variant_name": "slow_second_chunk_total_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    }))
    .await;
}

// Test that if a candidate times out, the evaluator can still see the other candidates
#[gtest]
#[tokio::test]
async fn test_model_provider_timeout_best_of_n_other_candidate() {
    best_of_n_other_candidate(json!({
        "function_name": "basic_test_timeout",
        "variant_name": "best_of_n",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    }))
    .await;
}

async fn best_of_n_other_candidate(payload: Value) {
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();

    let response_json = response.json::<Value>().await.unwrap();
    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(
        content.as_slice(),
        &[serde_json::json!({
            "type": "text",
            "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
        })]
    );
    assert_eq!(status, StatusCode::OK);

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    // One of the candidates timed out, leaving us with 2 candidates and the judge
    assert_eq!(model_inferences.len(), 3);
    assert_eq!(model_inferences[0].inference_id, inference_id);

    let mut model_names: Vec<&str> = model_inferences
        .iter()
        .map(|mi| mi.model_name.as_str())
        .collect();
    model_names.sort_unstable();
    assert_eq!(
        model_names,
        ["dummy::best_of_n_0", "dummy::good", "dummy::reasoner"]
    );

    // We don't check the content beyond the model names, as we already have lots
    // of tests for best_of_n
}

#[gtest]
#[tokio::test]
async fn test_model_provider_timeout_best_of_n_judge_timeout() {
    best_of_n_judge_timeout(json!({
        "function_name": "basic_test_timeout",
        "variant_name": "best_of_n_judge_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    }))
    .await;
}

async fn best_of_n_judge_timeout(payload: Value) {
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();

    // We don't check the content, as it will be a random candidate
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(status, StatusCode::OK);

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();

    // Both of the candidates should succeed, but the judge should time out,
    // leaving us with 2 successful inferences
    assert_eq!(model_inferences.len(), 2);
    assert_eq!(model_inferences[0].inference_id, inference_id);

    let mut model_names: Vec<&str> = model_inferences
        .iter()
        .map(|mi| mi.model_name.as_str())
        .collect();
    model_names.sort_unstable();
    assert_eq!(model_names, ["dummy::good", "dummy::reasoner"]);
}

async fn slow_second_chunk_streaming(payload: Value) {
    let start = Instant::now();
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut inference_id = None;

    while let Some(event) = response.next().await {
        let chunk = event.unwrap();
        println!("chunk: {chunk:?}");
        if let Event::Message(event) = chunk {
            if event.data == "[DONE]" {
                break;
            }
            let event = serde_json::from_str::<Value>(&event.data).unwrap();
            inference_id = Some(event["inference_id"].as_str().unwrap().parse().unwrap());
        }
    }

    // The overall stream duration should be at least 2 seconds, because we used the 'slow_second_chunk' model
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(2000),
        "slow_second_chunk should take at least 2 seconds, but took {elapsed:?}"
    );

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id.unwrap())
        .await
        .unwrap();
    assert_eq!(
        model_inferences.len(),
        1,
        "Expected exactly one model inference"
    );

    // The TTFT should be under 400ms (it should really be instant, but we give some buffer in case CI is overloaded)
    // As a result, the 'streaming.ttft_ms' timeout was not hit.
    let ttft_ms = model_inferences[0].ttft_ms.unwrap();
    assert!(
        ttft_ms <= 400,
        "ttft_ms should be less than 400ms, but was {ttft_ms}"
    );
}

/// Tests that `streaming.total_ms` fires mid-stream.
/// The `slow_second_chunk` model produces the first chunk immediately but delays 2s on the second.
/// With `total_ms = 500`, the stream should be cut off before the second chunk arrives.
async fn streaming_total_timeout(payload: Value) {
    let start = Instant::now();
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut saw_data_chunk = false;
    let mut saw_error = false;

    while let Some(event) = response.next().await {
        let chunk = event.unwrap();
        println!("chunk: {chunk:?}");
        if let Event::Message(event) = chunk {
            if event.data == "[DONE]" {
                break;
            }
            let event_json = serde_json::from_str::<Value>(&event.data).unwrap();
            if event_json.get("error").is_some() {
                saw_error = true;
                let error_msg = event_json["error"].as_str().unwrap();
                assert!(
                    error_msg.contains("streaming.total_ms"),
                    "Expected error message to contain `streaming.total_ms`, got: {error_msg}"
                );
            } else {
                saw_data_chunk = true;
            }
        }
    }

    assert!(
        saw_data_chunk,
        "Expected at least one data chunk before the timeout"
    );
    assert!(
        saw_error,
        "Expected a mid-stream error event from the total timeout"
    );

    // The stream should have been cut off well before 2 seconds (the slow_second_chunk delay).
    // With a 500ms total_ms, we expect roughly 500ms-1s total.
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(1800),
        "streaming_total_timeout should complete in under 1.8s, but took {elapsed:?}"
    );
}

// Model timeout tests

// Test timeouts that occur at both the model and model-provider level
#[gtest]
#[tokio::test]
async fn test_double_model_timeout() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let config = r#"
[functions.double_timeout]
type = "chat"

[functions.double_timeout.variants.slow_variant]
type = "chat_completion"
model = "double_timeout"

[models.double_timeout]
# Each of these providers has a non-streaming timeout of 500ms
# When we send an inference, the first two provides should time out,
# and then the model-level timeout should trigger while the third provider
# is still running
routing = ["first_timeout", "second_timeout", "third_timeout"]
timeouts = { non_streaming = { total_ms = 1200 }, streaming = { ttft_ms = 500 } }

[models.double_timeout.providers.first_timeout]
type = "dummy"
model_name = "slow"
timeouts = { non_streaming = { total_ms = 500 }, streaming = { ttft_ms = 500 } }

[models.double_timeout.providers.second_timeout]
type = "dummy"
model_name = "slow"
timeouts = { non_streaming = { total_ms = 500 }, streaming = { ttft_ms = 500 } }

[models.double_timeout.providers.third_timeout]
type = "dummy"
model_name = "slow"
timeouts = { non_streaming = { total_ms = 500 }, streaming = { ttft_ms = 500 } }
    "#;

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;
    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("double_timeout".to_string()),
            variant_name: Some("slow_variant".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello, world!".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap_err();

    expect_that!(
        response.to_string(),
        contains_substring(
            "Model double_timeout timed out due to configured `non_streaming.total_ms` timeout (1.2s)"
        )
    );
    // The first two providers should time out, but the third one shouldn't
    // (since the top-level model timeout will trigger first)
    expect_that!(
        logs_contain("Model provider first_timeout timed out"),
        eq(true)
    );
    expect_that!(
        logs_contain("Model provider first_timeout timed out"),
        eq(true)
    );
    expect_that!(logs_contain("third_timeout"), eq(false));
}

#[gtest]
#[tokio::test]
async fn test_model_timeout_non_streaming() {
    let payload = json!({
        "function_name": "basic_test_model_timeout",
        "variant_name": "slow_variant",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "Model slow_model_timeout timed out due to configured `non_streaming.total_ms` timeout (400ms)",
        "error_json": {
            "ModelTimeout": {
                "model_name": "slow_model_timeout",
                "timeout": {
                    "secs": 0,
                    "nanos": 400000000
                },
                "kind": "non_streaming_total"
            }
        }
    });
    expect_that!(response_json, eq(&expected));
}

#[gtest]
#[tokio::test]
async fn test_model_timeout_streaming() {
    let payload = json!({
        "function_name": "basic_test_model_timeout",
        "variant_name": "slow_variant",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let expected = json!({
        "error": "Model slow_model_timeout timed out due to configured `streaming.ttft_ms` timeout (500ms)",
        "error_json": {
            "ModelTimeout": {
                "model_name": "slow_model_timeout",
                "timeout": {
                    "secs": 0,
                    "nanos": 500000000
                },
                "kind": "streaming_ttft"
            }
        }
    });
    expect_that!(response_json, eq(&expected));
    expect_that!(status, eq(StatusCode::REQUEST_TIMEOUT));
}

#[gtest]
#[tokio::test]
async fn test_model_timeout_slow_second_chunk_streaming() {
    slow_second_chunk_streaming(json!({
        "function_name": "basic_test_model_timeout",
        "variant_name": "slow_second_chunk",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!",
                }
            ]
        },
        "stream": true,
    }))
    .await;
}

#[gtest]
#[tokio::test]
async fn test_model_total_timeout_streaming() {
    streaming_total_timeout(json!({
        "function_name": "basic_test_model_timeout",
        "variant_name": "slow_second_chunk_total_timeout",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    }))
    .await;
}
