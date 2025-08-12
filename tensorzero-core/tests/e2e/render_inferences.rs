use chrono::Utc;
use object_store::path::Path;
use serde_json::json;
use std::collections::HashMap;
use tensorzero::{
    ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint, Role, StorageKind, StoragePath,
    StoredChatInference, StoredInference, StoredJsonInference, Tool,
};
use tensorzero_core::{
    inference::types::{
        resolved_input::FileWithPath, Base64File, ContentBlock, ContentBlockChatOutput,
        JsonInferenceOutput, RequestMessage, ResolvedInput, ResolvedInputMessage,
        ResolvedInputMessageContent, Text,
    },
    tool::{ToolCallConfigDatabaseInsert, ToolCallOutput, ToolChoice},
};
use tracing_test::traced_test;
use uuid::Uuid;

use crate::providers::common::make_embedded_gateway;

/// Test that the render_samples function works when given an empty array of stored inferences.
#[tokio::test(flavor = "multi_thread")]
pub async fn test_render_samples_empty() {
    let client = make_embedded_gateway().await;

    // Test with an empty stored inferences array.
    let stored_inferences: Vec<StoredInference> = vec![];
    let rendered_inferences = client
        .experimental_render_samples(stored_inferences, HashMap::new())
        .await
        .unwrap();
    assert!(rendered_inferences.is_empty());
}

/// Test that the render_samples function drops the stored inference when the variants map is empty.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_samples_no_function() {
    let client = make_embedded_gateway().await;

    let stored_inferences = vec![StoredInference::Chat(StoredChatInference {
        function_name: "basic_test".to_string(),
        variant_name: "dummy".to_string(),
        input: ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: vec![],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: ToolCallConfigDatabaseInsert::default(),
        timestamp: Utc::now(),
        dispreferred_outputs: vec![],
        tags: HashMap::from([("test_key".to_string(), "test_value".to_string())]),
    })];

    let rendered_inferences = client
        .experimental_render_samples(stored_inferences, HashMap::new())
        .await
        .unwrap();
    assert!(rendered_inferences.is_empty());
    assert!(logs_contain("Missing function in variants: basic_test"));
}

/// Test that the render_samples function errors when the variants map contains a function with a nonexistent variant.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_samples_no_variant() {
    let client = make_embedded_gateway().await;

    let stored_inferences = vec![StoredInference::Chat(StoredChatInference {
        function_name: "basic_test".to_string(),
        variant_name: "dummy".to_string(),
        input: ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: vec![],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: ToolCallConfigDatabaseInsert::default(),
        timestamp: Utc::now(),
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    })];

    let error = client
        .experimental_render_samples(
            stored_inferences,
            HashMap::from([("basic_test".to_string(), "notavariant".to_string())]),
        )
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("Variant notavariant for function basic_test not found."));
    assert!(logs_contain(
        "Variant notavariant for function basic_test not found."
    ));
}

/// Test that the render_samples function drops the inference example when the
/// input is missing a required variable that the schema uses.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_samples_missing_variable() {
    let client = make_embedded_gateway().await;

    let stored_inferences = vec![StoredInference::Chat(StoredChatInference {
        function_name: "basic_test".to_string(),
        variant_name: "dummy".to_string(),
        input: ResolvedInput {
            system: Some(json!({"foo": "bar"})),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: vec![],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: ToolCallConfigDatabaseInsert::default(),
        timestamp: Utc::now(),
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    })];

    let rendered_inferences = client
        .experimental_render_samples(
            stored_inferences,
            HashMap::from([("basic_test".to_string(), "test".to_string())]),
        )
        .await
        .unwrap();
    assert!(rendered_inferences.is_empty());
    assert!(logs_contain("Could not render template: undefined value"));
}

/// Test that the render_samples function can render a normal chat example, a tool call example, a json example, and an example using images.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_samples_normal() {
    let client = make_embedded_gateway().await;

    let stored_inferences = vec![
        StoredInference::Chat(StoredChatInference {
            function_name: "basic_test".to_string(),
            variant_name: "dummy".to_string(),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            output: vec![],
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: ToolCallConfigDatabaseInsert::default(),
            timestamp: Utc::now(),
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        }),
        StoredInference::Json(StoredJsonInference {
            function_name: "json_success".to_string(),
            variant_name: "dummy".to_string(),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!({"country": "Japan"}),
                    }],
                }],
            },
            output: JsonInferenceOutput {
                parsed: Some(json!({})),
                raw: Some("{}".to_string()), // This should not be validated
            },
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            output_schema: json!({}), // This should be taken as-is
            timestamp: Utc::now(),
            dispreferred_outputs: vec![JsonInferenceOutput {
                parsed: Some(json!({})),
                raw: Some("{}".to_string()), // This should not be validated
            }],
            tags: HashMap::new(),
        }),
        StoredInference::Chat(StoredChatInference {
            function_name: "weather_helper".to_string(),
            variant_name: "dummy".to_string(),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            output: vec![ContentBlockChatOutput::ToolCall(ToolCallOutput {
                name: Some("get_temperature".to_string()),
                arguments: Some(json!({"location": "Tokyo"})),
                id: Uuid::now_v7().to_string(),
                raw_name: "get_temperature".to_string(),
                raw_arguments: "{\"location\":\"Tokyo\"}".to_string(),
            })],
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: ToolCallConfigDatabaseInsert {
                tools_available: vec![Tool {
                    name: "get_temperature".to_string(),
                    description: "Get the temperature of a location".to_string(),
                    parameters: json!({}), // Don't need to validate the arguments so we can leave blank
                    strict: false,
                }],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
            },
            timestamp: Utc::now(),
            dispreferred_outputs: vec![vec![ContentBlockChatOutput::Text(Text {
                text: "Hello, world!".to_string(),
            })]],
            tags: HashMap::new(),
        }),
        StoredInference::Chat(StoredChatInference {
            function_name: "basic_test".to_string(),
            variant_name: "gpt-4o-mini-2024-07-18".to_string(),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: json!("What is this a picture of?"),
                        },
                        ResolvedInputMessageContent::File(Box::new(FileWithPath {
                            file: Base64File {
                                url: None,
                                mime_type: mime::IMAGE_PNG,
                                data: None,
                            },
                            storage_path: StoragePath {
                                kind: StorageKind::S3Compatible {
                                    bucket_name: Some("tensorzero-e2e-test-images".to_string()),
                                    region: Some("us-east-1".to_string()),
                                    prefix: String::new(),
                                    endpoint: None,
                                    allow_http: None,
                                },
                                path: Path::from("observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"),
                            },
                        })),
                    ],
                }],
            },
            output: vec![],
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: ToolCallConfigDatabaseInsert::default(),
            timestamp: Utc::now(),
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        }),
    ];

    let rendered_inferences = client
        .experimental_render_samples(
            stored_inferences,
            HashMap::from([
                ("json_success".to_string(), "test".to_string()),
                ("weather_helper".to_string(), "anthropic".to_string()),
                ("basic_test".to_string(), "test".to_string()),
            ]),
        )
        .await
        .unwrap();
    assert_eq!(rendered_inferences.len(), 4);

    // Check the first rendered inference
    let first_inference = &rendered_inferences[0];
    // Check the input
    assert_eq!(
        first_inference.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta".to_string())
    );
    assert_eq!(first_inference.input.messages.len(), 1);

    let first_message = &first_inference.input.messages[0];
    assert_eq!(first_message.role, Role::User);
    assert_eq!(first_message.content.len(), 1);

    let ContentBlock::Text(text) = &first_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "Hello, world!");

    // Check other fields
    assert!(first_inference.output.as_ref().unwrap().is_empty());
    assert!(first_inference.tool_params.is_some());
    assert!(first_inference.output_schema.is_none());

    // Check the second rendered inference
    let second_inference = &rendered_inferences[1];
    assert_eq!(second_inference.function_name, "json_success");

    // Check the input
    assert_eq!(
        second_inference.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}".to_string())
    );
    assert_eq!(second_inference.input.messages.len(), 1);

    let second_message = &second_inference.input.messages[0];
    assert_eq!(second_message.role, Role::User);
    assert_eq!(second_message.content.len(), 1);

    let ContentBlock::Text(text) = &second_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "What is the name of the capital city of Japan?");

    // Check the output
    assert_eq!(second_inference.output.as_ref().unwrap().len(), 1);
    let ContentBlockChatOutput::Text(output_text) = &second_inference.output.as_ref().unwrap()[0]
    else {
        panic!("Expected text output");
    };
    assert_eq!(output_text.text, "{}");

    // Check the dispreferred outputs
    assert_eq!(second_inference.dispreferred_outputs.len(), 1);
    let ContentBlockChatOutput::Text(output_text) = &second_inference.dispreferred_outputs[0][0]
    else {
        panic!("Expected text output");
    };
    assert_eq!(output_text.text, "{}");
    // Check other fields
    assert!(second_inference.tool_params.is_none());
    assert!(second_inference.output_schema.is_some());

    // Check the third rendered inference
    let third_inference = &rendered_inferences[2];
    assert_eq!(third_inference.function_name, "weather_helper");

    // Check the input
    assert_eq!(
        third_inference.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").".to_string())
    );
    assert_eq!(third_inference.input.messages.len(), 1);

    let third_message = &third_inference.input.messages[0];
    assert_eq!(third_message.role, Role::User);
    assert_eq!(third_message.content.len(), 1);

    let ContentBlock::Text(text) = &third_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "Hello, world!");

    // Check the output
    assert_eq!(third_inference.output.as_ref().unwrap().len(), 1);
    let ContentBlockChatOutput::ToolCall(tool_call) = &third_inference.output.as_ref().unwrap()[0]
    else {
        panic!("Expected tool call output");
    };
    assert_eq!(tool_call.raw_name, "get_temperature");
    assert_eq!(tool_call.raw_arguments, "{\"location\":\"Tokyo\"}");
    assert_eq!(tool_call.name, Some("get_temperature".to_string()));
    assert_eq!(tool_call.arguments, Some(json!({"location": "Tokyo"})));

    // Check the dispreferred outputs
    assert_eq!(third_inference.dispreferred_outputs.len(), 1);
    let ContentBlockChatOutput::Text(output_text) = &third_inference.dispreferred_outputs[0][0]
    else {
        panic!("Expected text output");
    };
    assert_eq!(output_text.text, "Hello, world!");
    // Check other fields
    assert!(third_inference.tool_params.is_some());
    assert!(third_inference.output_schema.is_none());

    // Check the fourth rendered inference
    let fourth_inference = &rendered_inferences[3];
    assert_eq!(fourth_inference.function_name, "basic_test");

    // Check the input
    assert_eq!(
        fourth_inference.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta".to_string())
    );
    assert_eq!(fourth_inference.input.messages.len(), 1);

    let fourth_message = &fourth_inference.input.messages[0];
    assert_eq!(fourth_message.role, Role::User);
    assert_eq!(fourth_message.content.len(), 2);

    let ContentBlock::Text(text) = &fourth_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "What is this a picture of?");

    let ContentBlock::File(file) = &fourth_message.content[1] else {
        panic!("Expected file content");
    };

    // Check that the base64 string is > 1000 chars
    if let Some(data) = &file.file.data {
        assert!(data.len() > 1000);
    } else {
        panic!("Expected base64 data");
    }

    // Check the output
    assert_eq!(fourth_inference.output.as_ref().unwrap().len(), 0);

    // Check other fields
    assert!(fourth_inference.tool_params.is_some());
    assert!(fourth_inference.output_schema.is_none());
}

/// Test that the render_samples function can render a normal chat example, a tool call example, a json example, and an example using images.
#[tokio::test(flavor = "multi_thread")]
pub async fn test_render_samples_template_no_schema() {
    let client = make_embedded_gateway().await;

    let stored_inferences = vec![StoredInference::Chat(StoredChatInference {
        function_name: "basic_test_template_no_schema".to_string(),
        variant_name: "test".to_string(),
        timestamp: Utc::now(),
        input: ResolvedInput {
            system: Some("My system message".into()),
            messages: vec![
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: "First user message".into(),
                        },
                        ResolvedInputMessageContent::Text {
                            value: "Second user message".into(),
                        },
                    ],
                },
                ResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: "First assistant message".into(),
                        },
                        ResolvedInputMessageContent::Text {
                            value: "Second assistant message".into(),
                        },
                    ],
                },
            ],
        },
        output: vec![],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: ToolCallConfigDatabaseInsert::default(),
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    })];

    let rendered_inferences = client
        .experimental_render_samples(
            stored_inferences,
            HashMap::from([(
                "basic_test_template_no_schema".to_string(),
                "test".to_string(),
            )]),
        )
        .await
        .unwrap();
    assert_eq!(rendered_inferences.len(), 1);

    // Check the first rendered inference
    let first_inference = &rendered_inferences[0];
    // Check the input
    assert_eq!(
        first_inference.input.system,
        Some("The system text was `My system message`".to_string())
    );
    assert_eq!(first_inference.input.messages.len(), 2);

    assert_eq!(
        first_inference.input.messages[0],
        RequestMessage {
            role: Role::User,
            content: vec![
                ContentBlock::Text(Text {
                    text: "User content: `First user message`".into(),
                }),
                ContentBlock::Text(Text {
                    text: "User content: `Second user message`".into(),
                })
            ],
        }
    );

    assert_eq!(
        first_inference.input.messages[1],
        RequestMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text(Text {
                    text: "Assistant content: `First assistant message`".into(),
                }),
                ContentBlock::Text(Text {
                    text: "Assistant content: `Second assistant message`".into(),
                })
            ],
        }
    );

    // Check other fields
    assert!(first_inference.output.as_ref().unwrap().is_empty());
    assert!(first_inference.tool_params.is_some());
    assert!(first_inference.output_schema.is_none());
}

// ===== DATAPOINT TESTS =====
// The following tests mirror the StoredInference tests above but use Datapoint structs instead

/// Test that the render_samples function works when given an empty array of datapoints.
#[tokio::test(flavor = "multi_thread")]
pub async fn test_render_datapoints_empty() {
    let client = make_embedded_gateway().await;

    // Test with an empty datapoints array.
    let datapoints: Vec<Datapoint> = vec![];
    let rendered_samples = client
        .experimental_render_samples(datapoints, HashMap::new())
        .await
        .unwrap();
    assert!(rendered_samples.is_empty());
}

/// Test that the render_samples function drops the datapoint when the variants map is empty.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_datapoints_no_function() {
    let client = make_embedded_gateway().await;

    let datapoints = vec![Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "basic_test".to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input: ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        tags: None,
        auxiliary: "{}".to_string(),
        is_deleted: false,
        source_inference_id: None,
        staled_at: None,
        is_custom: false,
    })];

    let rendered_samples = client
        .experimental_render_samples(datapoints, HashMap::new())
        .await
        .unwrap();
    assert!(rendered_samples.is_empty());
    assert!(logs_contain("Missing function in variants: basic_test"));
}

/// Test that the render_samples function errors when the variants map contains a function with a nonexistent variant.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_datapoints_no_variant() {
    let client = make_embedded_gateway().await;

    let datapoints = vec![Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "basic_test".to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input: ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        tags: None,
        auxiliary: "{}".to_string(),
        is_deleted: false,
        source_inference_id: None,
        staled_at: None,
        is_custom: false,
    })];

    let error = client
        .experimental_render_samples(
            datapoints,
            HashMap::from([("basic_test".to_string(), "notavariant".to_string())]),
        )
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("Variant notavariant for function basic_test not found."));
    assert!(logs_contain(
        "Variant notavariant for function basic_test not found."
    ));
}

/// Test that the render_samples function drops the datapoint when the
/// input is missing a required variable that the schema uses.
/// Also test that a warning is logged.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_datapoints_missing_variable() {
    let client = make_embedded_gateway().await;

    let datapoints = vec![Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "basic_test".to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input: ResolvedInput {
            system: Some(json!({"foo": "bar"})),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        },
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        tags: None,
        auxiliary: "{}".to_string(),
        is_deleted: false,
        source_inference_id: None,
        staled_at: None,
        is_custom: false,
    })];

    let rendered_samples = client
        .experimental_render_samples(
            datapoints,
            HashMap::from([("basic_test".to_string(), "test".to_string())]),
        )
        .await
        .unwrap();
    assert!(rendered_samples.is_empty());
    assert!(logs_contain("Could not render template: undefined value"));
}

/// Test that the render_samples function can render a normal chat datapoint, a tool call datapoint, a json datapoint, and a datapoint using images.
#[tokio::test(flavor = "multi_thread")]
#[traced_test]
pub async fn test_render_datapoints_normal() {
    let client = make_embedded_gateway().await;

    let datapoints = vec![
        Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "basic_test".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            output: Some(vec![]),
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            tags: None,
            auxiliary: "{}".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            is_custom: false,
        }),
        Datapoint::Json(JsonInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "json_success".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!({"country": "Japan"}),
                    }],
                }],
            },
            output: Some(JsonInferenceOutput {
                parsed: Some(json!({})),
                raw: Some("{}".to_string()), // This should not be validated
            }),
            output_schema: json!({}), // This should be taken as-is
            tags: None,
            auxiliary: "{}".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            is_custom: false,
        }),
        Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "weather_helper".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::ToolCall(ToolCallOutput {
                name: Some("get_temperature".to_string()),
                arguments: Some(json!({"location": "Tokyo"})),
                id: Uuid::now_v7().to_string(),
                raw_name: "get_temperature".to_string(),
                raw_arguments: "{\"location\":\"Tokyo\"}".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert {
                tools_available: vec![Tool {
                    name: "get_temperature".to_string(),
                    description: "Get the temperature of a location".to_string(),
                    parameters: json!({}), // Don't need to validate the arguments so we can leave blank
                    strict: false,
                }],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
            }),
            tags: None,
            auxiliary: "{}".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            is_custom: false,
        }),
        Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "basic_test".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: Some(json!({"assistant_name": "Dr. Mehta"})),
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: json!("What is this a picture of?"),
                        },
                        ResolvedInputMessageContent::File(Box::new(FileWithPath {
                            file: Base64File {
                                url: None,
                                mime_type: mime::IMAGE_PNG,
                                data: None,
                            },
                            storage_path: StoragePath {
                                kind: StorageKind::S3Compatible {
                                    bucket_name: Some("tensorzero-e2e-test-images".to_string()),
                                    region: Some("us-east-1".to_string()),
                                    prefix: String::new(),
                                    endpoint: None,
                                    allow_http: None,
                                },
                                path: Path::from("observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"),
                            },
                        })),
                    ],
                }],
            },
            output: Some(vec![]),
            tool_params: Some(ToolCallConfigDatabaseInsert::default()),
            tags: None,
            auxiliary: "{}".to_string(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            is_custom: false,
        }),
    ];

    let rendered_samples = client
        .experimental_render_samples(
            datapoints,
            HashMap::from([
                ("json_success".to_string(), "test".to_string()),
                ("weather_helper".to_string(), "anthropic".to_string()),
                ("basic_test".to_string(), "test".to_string()),
            ]),
        )
        .await
        .unwrap();
    assert_eq!(rendered_samples.len(), 4);

    // Check the first rendered sample (basic_test chat)
    let first_sample = &rendered_samples[0];
    // Check the input
    assert_eq!(
        first_sample.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta".to_string())
    );
    assert_eq!(first_sample.input.messages.len(), 1);

    let first_message = &first_sample.input.messages[0];
    assert_eq!(first_message.role, Role::User);
    assert_eq!(first_message.content.len(), 1);

    let ContentBlock::Text(text) = &first_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "Hello, world!");

    // Check other fields
    assert!(first_sample.output.as_ref().unwrap().is_empty());
    assert!(first_sample.tool_params.is_some());
    assert!(first_sample.output_schema.is_none());

    // Check the second rendered sample (json_success)
    let second_sample = &rendered_samples[1];
    assert_eq!(second_sample.function_name, "json_success");

    // Check the input
    assert_eq!(
        second_sample.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}".to_string())
    );
    assert_eq!(second_sample.input.messages.len(), 1);

    let second_message = &second_sample.input.messages[0];
    assert_eq!(second_message.role, Role::User);
    assert_eq!(second_message.content.len(), 1);

    let ContentBlock::Text(text) = &second_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "What is the name of the capital city of Japan?");

    // Check the output
    assert_eq!(second_sample.output.as_ref().unwrap().len(), 1);
    let ContentBlockChatOutput::Text(output_text) = &second_sample.output.as_ref().unwrap()[0]
    else {
        panic!("Expected text output");
    };
    assert_eq!(output_text.text, "{}");

    // Check other fields
    assert!(second_sample.tool_params.is_none());
    assert!(second_sample.output_schema.is_some());

    // Check the third rendered sample (weather_helper with tool call)
    let third_sample = &rendered_samples[2];
    assert_eq!(third_sample.function_name, "weather_helper");

    // Check the input
    assert_eq!(
        third_sample.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\").".to_string())
    );
    assert_eq!(third_sample.input.messages.len(), 1);

    let third_message = &third_sample.input.messages[0];
    assert_eq!(third_message.role, Role::User);
    assert_eq!(third_message.content.len(), 1);

    let ContentBlock::Text(text) = &third_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "Hello, world!");

    // Check the output
    assert_eq!(third_sample.output.as_ref().unwrap().len(), 1);
    let ContentBlockChatOutput::ToolCall(tool_call) = &third_sample.output.as_ref().unwrap()[0]
    else {
        panic!("Expected tool call output");
    };
    assert_eq!(tool_call.raw_name, "get_temperature");
    assert_eq!(tool_call.raw_arguments, "{\"location\":\"Tokyo\"}");
    assert_eq!(tool_call.name, Some("get_temperature".to_string()));
    assert_eq!(tool_call.arguments, Some(json!({"location": "Tokyo"})));

    // Check other fields
    assert!(third_sample.tool_params.is_some());
    assert!(third_sample.output_schema.is_none());

    // Check the fourth rendered sample (basic_test with image)
    let fourth_sample = &rendered_samples[3];
    assert_eq!(fourth_sample.function_name, "basic_test");

    // Check the input
    assert_eq!(
        fourth_sample.input.system,
        Some("You are a helpful and friendly assistant named Dr. Mehta".to_string())
    );
    assert_eq!(fourth_sample.input.messages.len(), 1);

    let fourth_message = &fourth_sample.input.messages[0];
    assert_eq!(fourth_message.role, Role::User);
    assert_eq!(fourth_message.content.len(), 2);

    let ContentBlock::Text(text) = &fourth_message.content[0] else {
        panic!("Expected text content");
    };
    assert_eq!(text.text, "What is this a picture of?");

    let ContentBlock::File(file) = &fourth_message.content[1] else {
        panic!("Expected file content");
    };

    // Check that the base64 string is > 1000 chars
    if let Some(data) = &file.file.data {
        assert!(data.len() > 1000);
    } else {
        panic!("Expected base64 data");
    }

    // Check the output
    assert_eq!(fourth_sample.output.as_ref().unwrap().len(), 0);

    // Check other fields
    assert!(fourth_sample.tool_params.is_some());
    assert!(fourth_sample.output_schema.is_none());
}

/// Test that the render_samples function can render a datapoint with template but no schema.
#[tokio::test(flavor = "multi_thread")]
pub async fn test_render_datapoints_template_no_schema() {
    let client = make_embedded_gateway().await;

    let datapoints = vec![Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "basic_test_template_no_schema".to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input: ResolvedInput {
            system: Some("My system message".into()),
            messages: vec![
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: "First user message".into(),
                        },
                        ResolvedInputMessageContent::Text {
                            value: "Second user message".into(),
                        },
                    ],
                },
                ResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: "First assistant message".into(),
                        },
                        ResolvedInputMessageContent::Text {
                            value: "Second assistant message".into(),
                        },
                    ],
                },
            ],
        },
        output: Some(vec![]),
        tool_params: Some(ToolCallConfigDatabaseInsert::default()),
        tags: None,
        auxiliary: "{}".to_string(),
        is_deleted: false,
        source_inference_id: None,
        staled_at: None,
        is_custom: false,
    })];

    let rendered_samples = client
        .experimental_render_samples(
            datapoints,
            HashMap::from([(
                "basic_test_template_no_schema".to_string(),
                "test".to_string(),
            )]),
        )
        .await
        .unwrap();
    assert_eq!(rendered_samples.len(), 1);

    // Check the first rendered sample
    let first_sample = &rendered_samples[0];
    // Check the input
    assert_eq!(
        first_sample.input.system,
        Some("The system text was `My system message`".to_string())
    );
    assert_eq!(first_sample.input.messages.len(), 2);

    assert_eq!(
        first_sample.input.messages[0],
        RequestMessage {
            role: Role::User,
            content: vec![
                ContentBlock::Text(Text {
                    text: "User content: `First user message`".into(),
                }),
                ContentBlock::Text(Text {
                    text: "User content: `Second user message`".into(),
                })
            ],
        }
    );

    assert_eq!(
        first_sample.input.messages[1],
        RequestMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text(Text {
                    text: "Assistant content: `First assistant message`".into(),
                }),
                ContentBlock::Text(Text {
                    text: "Assistant content: `Second assistant message`".into(),
                })
            ],
        }
    );

    // Check other fields
    assert!(first_sample.output.as_ref().unwrap().is_empty());
    assert!(first_sample.tool_params.is_some());
    assert!(first_sample.output_schema.is_none());
}
