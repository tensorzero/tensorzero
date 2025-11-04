use std::time::{Duration, Instant};

use http::StatusCode;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent, Role,
};
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
        select_model_inference_clickhouse, select_model_inferences_clickhouse,
    },
    inference::types::TextKind,
};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

// Variant timeout tests

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
    assert_eq!(
        response_json,
        json!({
            "error": "Variant `slow_timeout` timed out due to configured `non_streaming.total_ms` timeout (400ms)",
            "error_json": {
                "VariantTimeout": {
                    "variant_name": "slow_timeout",
                    "timeout": {
                        "secs": 0,
                        "nanos": 400000000
                    },
                    "streaming": false
                }
            }
        })
    );
}

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
    assert_eq!(
        response_json,
        json!({
            "error": "Variant `slow_timeout` timed out due to configured `streaming.ttft_ms` timeout (500ms)",
            "error_json": {
                "VariantTimeout": {
                    "variant_name": "slow_timeout",
                    "timeout": {
                        "secs": 0,
                        "nanos": 500000000
                    },
                    "streaming": true
                }
            }
        })
    );
    assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
}

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

#[tokio::test]
async fn test_chat_inference_ttft_ms() {
    test_inference_ttft_ms(
        json!({
            "function_name": "inference_ttft_chat",
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
        }),
        false,
    )
    .await;
}
#[tokio::test]
async fn test_json_inference_ttft_ms() {
    test_inference_ttft_ms(
        json!({
            "function_name": "inference_ttft_json",
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
        }),
        true,
    )
    .await;
}

async fn test_inference_ttft_ms(payload: Value, json: bool) {
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
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

    // Sleep for 200ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let clickhouse = get_clickhouse().await;

    let inference = if json {
        select_json_inference_clickhouse(&clickhouse, inference_id.unwrap())
            .await
            .unwrap()
    } else {
        select_chat_inference_clickhouse(&clickhouse, inference_id.unwrap())
            .await
            .unwrap()
    };

    // The inference level-TTFT should be high, due to the timeout for the first provider
    assert!(
        inference["ttft_ms"].as_u64().unwrap() > 500,
        "Unexpected ttft_ms: {inference:?}"
    );

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id.unwrap())
        .await
        .unwrap();

    // The first provider will time out, so we should only have one inference
    assert_eq!(model_inferences.len(), 1);
    println!("model_inferences: {model_inferences:?}");
    assert_eq!(model_inferences[0]["model_name"], "first_provider_timeout");
    assert_eq!(model_inferences[0]["model_provider_name"], "second_good");
    // The model inference should have a small TTFT, since the successful model provider
    // is fast
    assert!(
        model_inferences[0]["ttft_ms"].as_u64().unwrap() < 100,
        "Unexpected ttft ms in model provider"
    );
}

// We don't currently support setting an actual variant as the judge,
// so we can't apply a variant timeout to the judge itself.

// Test that if a candidate times out, the evaluator can still see the other candidates
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
    assert_eq!(
        response_json,
        json!({
            "error": "All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `non_streaming.total_ms` timeout (400ms)",
            "error_json": {
                "ModelProvidersExhausted": {
                    "provider_errors": {
                        "slow": {
                            "ModelProviderTimeout": {
                                "provider_name": "slow",
                                "timeout": {
                                    "secs": 0,
                                    "nanos": 400000000
                                },
                                "streaming": false
                            }
                        }
                    }
                }
            }
        })
    );
}

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
    assert_eq!(
        response_json,
        json!({
            "error": "All model providers failed to infer with errors: slow: Model provider slow timed out due to configured `streaming.ttft_ms` timeout (500ms)",
            "error_json": {
                "ModelProvidersExhausted": {
                    "provider_errors": {
                        "slow": {
                            "ModelProviderTimeout": {
                                "provider_name": "slow",
                                "timeout": {
                                    "secs": 0,
                                    "nanos": 500000000
                                },
                                "streaming": true
                            }
                        }
                    }
                }
            }
        })
    );
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}

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

// Test that if a candidate times out, the evaluator can still see the other candidates
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

    // Sleep for 200ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // One of the candidates timed out, leaving us with 2 candidates and the judge
    assert_eq!(model_inferences.len(), 3);
    assert_eq!(
        model_inferences[0]
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        inference_id.to_string()
    );

    let mut model_names = Vec::new();

    for model_inference in &model_inferences {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        model_names.push(model_name);
    }
    model_names.sort();
    assert_eq!(
        model_names,
        ["dummy::best_of_n_0", "dummy::good", "dummy::reasoner"]
    );

    // We don't check the content beyond the model names, as we already have lots
    // of tests for best_of_n
}

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

    // Sleep for 200ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let model_inferences = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // Both of the candidates should succeed, but the judge should time out,
    // leaving us with 2 successful inferences
    assert_eq!(model_inferences.len(), 2);
    assert_eq!(
        model_inferences[0]
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        inference_id.to_string()
    );

    let mut model_names = Vec::new();

    for model_inference in &model_inferences {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        model_names.push(model_name);
    }
    model_names.sort();
    assert_eq!(model_names, ["dummy::good", "dummy::reasoner"]);
}

async fn slow_second_chunk_streaming(payload: Value) {
    let start = Instant::now();
    let mut response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
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

    // Wait 200ms to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let clickhouse = get_clickhouse().await;

    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id.unwrap())
        .await
        .unwrap();

    // The TTFT should be under 400ms (it should really be instant, but we give some buffer in case CI is overloaded)
    // As a result, the 'streaming.ttft_ms' timeout was not hit.
    let ttft_ms = model_inference["ttft_ms"].as_u64().unwrap();
    assert!(
        ttft_ms <= 400,
        "ttft_ms should be less than 400ms, but was {ttft_ms}"
    );
}

// Model timeout tests

// Test timeouts that occur at both the model and model-provider level
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
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello, world!".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap_err();

    assert!(
        response.to_string().contains("Model double_timeout timed out due to configured `non_streaming.total_ms` timeout (1.2s)"),
        "Unexpected error message: {response}"
    );
    // The first two providers should time out, but the third one shouldn't
    // (since the top-level model timeout will trigger first)
    assert!(logs_contain("Model provider first_timeout timed out"));
    assert!(logs_contain("Model provider first_timeout timed out"));
    assert!(!logs_contain("third_timeout"));
}

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
    assert_eq!(
        response_json,
        json!({
            "error": "Model slow_model_timeout timed out due to configured `non_streaming.total_ms` timeout (400ms)",
            "error_json": {
                "ModelTimeout": {
                    "model_name": "slow_model_timeout",
                    "timeout": {
                        "secs": 0,
                        "nanos": 400000000
                    },
                    "streaming": false
                }
            }
        })
    );
}

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
    assert_eq!(
        response_json,
        json!({
            "error": "Model slow_model_timeout timed out due to configured `streaming.ttft_ms` timeout (500ms)",
            "error_json": {
                "ModelTimeout": {
                    "model_name": "slow_model_timeout",
                    "timeout": {
                        "secs": 0,
                        "nanos": 500000000
                    },
                    "streaming": true
                }
            }
        })
    );
    assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
}

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
