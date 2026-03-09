#![expect(clippy::print_stdout)]
use crate::common::get_gateway_endpoint;
use crate::providers::common::E2ETestProvider;
use crate::providers::helpers::get_modal_extra_headers;
use futures::StreamExt;
use googletest::prelude::*;
use reqwest::Client;
use reqwest::StatusCode;
use reqwest_sse_stream::Event;
use reqwest_sse_stream::RequestBuilderExt;
use serde_json::Value;
use serde_json::json;
use tensorzero::Role;
use tensorzero_core::db::{
    delegating_connection::DelegatingDatabaseConnection,
    inferences::{InferenceQueries, ListInferencesParams},
    model_inferences::ModelInferenceQueries,
    test_helpers::TestDatabaseHelpers,
};
use tensorzero_core::inference::types::ContentBlockOutput;
use tensorzero_core::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, StoredContentBlock, StoredModelInference, StoredRequestMessage, Text,
};
use tensorzero_core::stored_inference::{
    StoredChatInferenceDatabase, StoredInferenceDatabase, StoredJsonInference,
};
use tensorzero_core::test_helpers::get_e2e_config;
use uuid::Uuid;

pub async fn test_reasoning_inference_request_simple_nonstreaming_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Calculator"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is 34 * 57 + 21 / 3? Answer with just the number."
                }
            ]},
        "extra_headers": extra_headers,
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();

    let mut found_text = false;
    let mut found_thought = false;
    let mut text_content = String::new();

    for block in content {
        let block_type = block.get("type").unwrap().as_str().unwrap();
        match block_type {
            "text" => {
                found_text = true;
                text_content = block.get("text").unwrap().as_str().unwrap().to_string();
            }
            "thought" => {
                found_thought = true;
            }
            _ => {
                // Skip unknown content block types (e.g., raw reasoning data from OpenAI Responses API)
            }
        }
    }

    assert!(found_text, "Expected to find a text block");
    assert!(found_thought, "Expected to find a thought block");
    // We only check that the response contains digits rather than a specific answer,
    // since models can make arithmetic mistakes.
    assert!(
        text_content.chars().any(|c| c.is_ascii_digit()),
        "Expected numeric digits in text content: {text_content}"
    );

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Check the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check Inference table
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
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    println!("ChatInference: {chat_inf:#?}");

    expect_that!(
        chat_inf,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            function_name: eq("basic_test"),
            variant_name: eq(provider.variant_name.as_str()),
            processing_time_ms: some(gt(&0)),
            ..
        })
    );

    let input_value = serde_json::to_value(&chat_inf.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Calculator"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 34 * 57 + 21 / 3? Answer with just the number."}]
            }
        ]
    });
    assert_eq!(input_value, correct_input);

    let output = chat_inf.output.as_ref().expect("output should be present");
    let mut found_text = false;
    let mut found_thought = false;
    let mut db_content = String::new();
    for block in output {
        match block {
            ContentBlockChatOutput::Text(t) => {
                found_text = true;
                db_content = t.text.clone();
            }
            ContentBlockChatOutput::Thought(_) => {
                found_thought = true;
            }
            _ => {}
        }
    }
    assert!(found_text, "Expected to find a text block in output");
    assert!(found_thought, "Expected to find a thought block in output");
    assert_eq!(db_content, text_content);

    expect_that!(chat_inf.tags.get("foo"), some(eq(&"bar".to_string())));

    assert!(chat_inf.tool_params.is_none(), "Expected no tool_params");

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    println!("ModelInference: {mi:#?}");

    let raw_request_str = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(
        raw_request_str.contains("34"),
        "Expected raw_request to contain math problem"
    );
    let _: Value = serde_json::from_str(raw_request_str).expect("raw_request should be valid JSON");

    let raw_response_str = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    // We only check that the response contains digits rather than a specific answer,
    // since models can make arithmetic mistakes.
    assert!(
        raw_response_str.chars().any(|c| c.is_ascii_digit()),
        "Expected numeric digits in raw_response"
    );
    let _: Value =
        serde_json::from_str(raw_response_str).expect("raw_response should be valid JSON");

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(provider.model_name.as_str()),
            model_provider_name: eq(provider.model_provider_name.as_str()),
            input_tokens: some(gt(&0)),
            output_tokens: some(gt(&0)),
            response_time_ms: some(gt(&0)),
            ttft_ms: none(),
            ..
        })
    );

    let system = mi.system.as_ref().expect("system should be present");
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Calculator"
    );

    let input_messages = mi
        .input_messages
        .as_ref()
        .expect("input_messages should be present");
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is 34 * 57 + 21 / 3? Answer with just the number.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    let output = mi.output.as_ref().expect("output should be present");
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Text(_))),
        "Missing text block in output: {output:#?}"
    );
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Thought(_))),
        "Missing thought block in output: {output:#?}"
    );
}

pub async fn test_reasoning_inference_request_simple_streaming_with_provider(
    provider: E2ETestProvider,
) {
    use reqwest_sse_stream::{Event, RequestBuilderExt};
    use serde_json::Value;

    use crate::common::get_gateway_endpoint;

    // TODO (#6680): re-enable once streaming reasoning is fixed for GCP Vertex Gemini
    if provider.model_provider_name == "gcp_vertex_gemini" {
        return;
    }

    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Calculator"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is 34 * 57 + 21 / 3? Answer with just the number."
                }
            ]},
        "stream": true,
        "extra_headers": extra_headers,
        "tags": {"key": tag_value},
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
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
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut full_thought = None;
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            if content_block.get("type").unwrap().as_str().unwrap() == "text" {
                let content = content_block.get("text").unwrap().as_str().unwrap();
                full_content.push_str(content);
            } else if content_block.get("type").unwrap().as_str().unwrap() == "thought" {
                // Some providers give signature-only thought blocks
                if let Some(thought_text) = content_block.get("text").and_then(|v| v.as_str()) {
                    full_thought
                        .get_or_insert_with(String::new)
                        .push_str(thought_text);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    // We only check that the response contains digits rather than a specific answer,
    // since models can make arithmetic mistakes.
    assert!(
        full_content.chars().any(|c| c.is_ascii_digit()),
        "Expected numeric digits in content: {full_content}"
    );
    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Check the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check Inference table
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
    assert_that!(inferences, len(eq(1)));
    let chat_inf = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    println!("ChatInference: {chat_inf:#?}");

    expect_that!(
        chat_inf,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            function_name: eq("basic_test"),
            variant_name: eq(provider.variant_name.as_str()),
            processing_time_ms: some(gt(&0)),
            ..
        })
    );

    let input_value = serde_json::to_value(&chat_inf.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Calculator"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 34 * 57 + 21 / 3? Answer with just the number."}]
            }
        ]
    });
    assert_eq!(input_value, correct_input);

    let output = chat_inf.output.as_ref().expect("output should be present");
    let mut found_text = false;
    let mut found_thought = false;
    let mut db_content = String::new();
    let mut db_thought = None;
    for block in output {
        match block {
            ContentBlockChatOutput::Text(t) => {
                found_text = true;
                db_content.push_str(&t.text);
            }
            ContentBlockChatOutput::Thought(t) => {
                found_thought = true;
                if let Some(thought_text) = &t.text {
                    db_thought
                        .get_or_insert_with(String::new)
                        .push_str(thought_text);
                }
            }
            _ => {}
        }
    }

    assert!(found_text, "Expected to find a text block");
    assert!(found_thought, "Expected to find a thought block");
    assert_eq!(db_content, full_content);
    assert_eq!(db_thought, full_thought);

    assert!(chat_inf.tool_params.is_none(), "Expected no tool_params");

    expect_that!(chat_inf.tags.get("key"), some(eq(&tag_value)));

    // Check ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    println!("ModelInference: {mi:#?}");

    let raw_request_str = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(
        raw_request_str.contains("34"),
        "Expected raw_request to contain math problem"
    );
    let _: Value = serde_json::from_str(raw_request_str).expect("raw_request should be valid JSON");

    let raw_response_str = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    // Check if raw_response is valid JSONL
    for line in raw_response_str.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        expect_that!(mi.input_tokens, some(eq(0u32)));
        expect_that!(mi.output_tokens, some(eq(0u32)));
    } else {
        expect_that!(mi.input_tokens, some(gt(0u32)));
        expect_that!(mi.output_tokens, some(gt(0u32)));
    }

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(provider.model_name.as_str()),
            model_provider_name: eq(provider.model_provider_name.as_str()),
            response_time_ms: some(gt(&0)),
            ttft_ms: some(gt(&0)),
            ..
        })
    );

    let ttft_ms = mi.ttft_ms.expect("ttft_ms should be present");
    let response_time_ms = mi
        .response_time_ms
        .expect("response_time_ms should be present");
    assert!(ttft_ms <= response_time_ms);

    let system = mi.system.as_ref().expect("system should be present");
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Calculator"
    );

    let input_messages = mi
        .input_messages
        .as_ref()
        .expect("input_messages should be present");
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is 34 * 57 + 21 / 3? Answer with just the number.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    let output = mi.output.as_ref().expect("output should be present");
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Text(_))),
        "Missing text block in output: {output:#?}"
    );
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Thought(_))),
        "Missing thought block in output: {output:#?}"
    );
}

pub async fn test_reasoning_inference_request_json_mode_nonstreaming_with_provider(
    provider: E2ETestProvider,
) {
    // Direct Anthropic uses output_format for json_mode=strict
    // AWS Bedrock and GCP Vertex Anthropic use json_mode=off (prompt-based JSON) to avoid prefill conflicts

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "json_math",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Calculator"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is 34 * 57 + 21 / 3? Answer with just the number."
                }
            ]},
        "stream": false,
        "extra_headers": extra_headers,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let output = response_json.get("output").unwrap().as_object().unwrap();
    assert!(output.keys().len() == 2);
    let parsed_output = output.get("parsed").unwrap().as_object().unwrap();
    // We only check that the answer contains digits rather than a specific number,
    // since models can make arithmetic mistakes.
    assert!(
        parsed_output
            .get("answer")
            .unwrap()
            .as_str()
            .unwrap()
            .chars()
            .any(|c| c.is_ascii_digit()),
        "Expected numeric digits in answer"
    );
    let raw_output = output.get("raw").unwrap().as_str().unwrap();
    let raw_output: Value = serde_json::from_str(raw_output).unwrap();
    assert_eq!(&raw_output, output.get("parsed").unwrap());

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Check the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check JsonInference table
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
    assert_that!(inferences, len(eq(1)));
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected JSON inference"),
    };

    println!("JsonInference: {json_inf:#?}");

    expect_that!(
        json_inf,
        matches_pattern!(StoredJsonInference {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            function_name: eq("json_math"),
            variant_name: eq(provider.variant_name.as_str()),
            processing_time_ms: some(gt(&0)),
            ..
        })
    );

    let input_value = serde_json::to_value(&json_inf.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Calculator"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 34 * 57 + 21 / 3? Answer with just the number."}]
            }
        ]
    });
    assert_eq!(input_value, correct_input);

    let json_output = json_inf.output.as_ref().expect("output should be present");
    let json_output_value = serde_json::to_value(json_output).unwrap();
    let json_output_obj = json_output_value.as_object().unwrap();
    assert_eq!(json_output_obj, output);

    let output_schema = json_inf
        .output_schema
        .as_ref()
        .expect("output_schema should be present");
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, &expected_output_schema);

    // Check auxiliary content via ModelInference output: `reasoning_effort = "low"` may not
    // produce thought blocks, so we only verify that any thought blocks present are well-formed.
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    println!("ModelInference: {mi:#?}");

    let mi_output = mi.output.as_ref().expect("output should be present");
    for block in mi_output {
        if matches!(block, ContentBlockOutput::Thought(_)) {
            // Thought block is well-formed (deserialized successfully)
        }
    }

    let raw_request_str = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(
        raw_request_str.contains("34"),
        "Expected raw_request to contain math problem"
    );
    let _: Value = serde_json::from_str(raw_request_str).expect("raw_request should be valid JSON");

    let raw_response_str = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    // We only check that the response contains digits rather than a specific answer,
    // since models can make arithmetic mistakes.
    assert!(
        raw_response_str.chars().any(|c| c.is_ascii_digit()),
        "Expected numeric digits in raw_response"
    );
    let _: Value =
        serde_json::from_str(raw_response_str).expect("raw_response should be valid JSON");

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(provider.model_name.as_str()),
            model_provider_name: eq(provider.model_provider_name.as_str()),
            input_tokens: some(gt(&0)),
            output_tokens: some(gt(&0)),
            response_time_ms: some(gt(&0)),
            ttft_ms: none(),
            ..
        })
    );

    let system = mi.system.as_ref().expect("system should be present");
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Calculator.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );

    let input_messages = mi
        .input_messages
        .as_ref()
        .expect("input_messages should be present");
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is 34 * 57 + 21 / 3? Answer with just the number.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    let output = mi.output.as_ref().expect("output should be present");
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Text(_))),
        "Unexpected output: {output:#?}"
    );
}

pub async fn test_reasoning_inference_request_json_mode_streaming_with_provider(
    provider: E2ETestProvider,
) {
    // OpenAI O1 doesn't support streaming responses
    if provider.model_provider_name.contains("openai") && provider.model_name.starts_with("o1") {
        return;
    }

    // Direct Anthropic uses output_format for json_mode=strict
    // AWS Bedrock and GCP Vertex Anthropic use json_mode=off (prompt-based JSON) to avoid prefill conflicts

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "json_math",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Calculator"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is 34 * 57 + 21 / 3? Answer with just the number."
                }
            ]},
        "stream": true,
        "extra_headers": extra_headers,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
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
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let raw = chunk_json
            .get("raw")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !raw.is_empty() {
            full_content.push_str(raw);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    // We only check that the response contains digits rather than a specific answer,
    // since models can make arithmetic mistakes.
    assert!(
        full_content.chars().any(|c| c.is_ascii_digit()),
        "Expected numeric digits in content: {full_content}"
    );

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Check the database
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let config = get_e2e_config().await;

    // Check JsonInference table
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
    assert_that!(inferences, len(eq(1)));
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected JSON inference"),
    };

    println!("JsonInference: {json_inf:#?}");

    expect_that!(
        json_inf,
        matches_pattern!(StoredJsonInference {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            function_name: eq("json_math"),
            variant_name: eq(provider.variant_name.as_str()),
            processing_time_ms: some(gt(&0)),
            ..
        })
    );

    let input_value = serde_json::to_value(&json_inf.input).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Calculator"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 34 * 57 + 21 / 3? Answer with just the number."}]
            }
        ]
    });
    assert_eq!(input_value, correct_input);

    let json_output = json_inf.output.as_ref().expect("output should be present");
    let json_output_value = serde_json::to_value(json_output).unwrap();
    let json_output_obj = json_output_value.as_object().unwrap();
    assert_eq!(json_output_obj.keys().len(), 2);
    let db_parsed = json_output_obj.get("parsed").unwrap().as_object().unwrap();
    let db_raw = json_output_obj.get("raw").unwrap().as_str().unwrap();
    let db_raw: Value = serde_json::from_str(db_raw).unwrap();
    let db_raw = db_raw.as_object().unwrap();
    assert_eq!(db_parsed, db_raw);
    let full_content_parsed: Value = serde_json::from_str(&full_content).unwrap();
    let full_content_parsed = full_content_parsed.as_object().unwrap();
    assert_eq!(db_parsed, full_content_parsed);

    let output_schema = json_inf
        .output_schema
        .as_ref()
        .expect("output_schema should be present");
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(output_schema, &expected_output_schema);

    // Check auxiliary content via ModelInference output: `reasoning_effort = "low"` may not
    // produce thought blocks, so we only verify that any thought blocks present are well-formed.
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(1)));
    let mi = &model_inferences[0];

    println!("ModelInference: {mi:#?}");

    let mi_output = mi.output.as_ref().expect("output should be present");
    for block in mi_output {
        if matches!(block, ContentBlockOutput::Thought(_)) {
            // Thought block is well-formed (deserialized successfully)
        }
    }

    let raw_request_str = mi
        .raw_request
        .as_ref()
        .expect("raw_request should be present");
    assert!(
        raw_request_str.contains("34"),
        "Expected raw_request to contain math problem"
    );
    assert!(
        raw_request_str.to_lowercase().contains("calculator"),
        "Expected raw_request to contain assistant name"
    );
    let _: Value = serde_json::from_str(raw_request_str).expect("raw_request should be valid JSON");

    let raw_response_str = mi
        .raw_response
        .as_ref()
        .expect("raw_response should be present");
    // Check if raw_response is valid JSONL
    for line in raw_response_str.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        expect_that!(mi.input_tokens, some(eq(0u32)));
        expect_that!(mi.output_tokens, some(eq(0u32)));
    } else {
        expect_that!(mi.input_tokens, some(gt(0u32)));
        expect_that!(mi.output_tokens, some(gt(0u32)));
    }

    expect_that!(
        mi,
        matches_pattern!(StoredModelInference {
            inference_id: eq(&inference_id),
            model_name: eq(provider.model_name.as_str()),
            model_provider_name: eq(provider.model_provider_name.as_str()),
            response_time_ms: some(gt(&0)),
            ttft_ms: some(gt(&0)),
            ..
        })
    );

    let ttft_ms = mi.ttft_ms.expect("ttft_ms should be present");
    let response_time_ms = mi
        .response_time_ms
        .expect("response_time_ms should be present");
    assert!(ttft_ms <= response_time_ms);

    let system = mi.system.as_ref().expect("system should be present");
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Calculator.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );

    let input_messages = mi
        .input_messages
        .as_ref()
        .expect("input_messages should be present");
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "What is 34 * 57 + 21 / 3? Answer with just the number.".to_string(),
        })],
    }];
    assert_eq!(input_messages, &expected_input_messages);

    let output = mi.output.as_ref().expect("output should be present");
    assert!(
        output
            .iter()
            .any(|c| matches!(c, ContentBlockOutput::Text(_))),
        "Missing text block in output: {output:#?}"
    );
}
