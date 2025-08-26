use jsonschema::{Draft, JSONSchema};
use serde_json::{json, Value};
use serde_yaml::Value as YamlValue;
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Clone)]
struct Schemas {
    chat_completion_schema: Arc<JSONSchema>,
    chat_completion_stream_schema: Arc<JSONSchema>,
    error_schema: Arc<JSONSchema>,
}

// OnceCell ensures the schemas are compiled only once
static SCHEMAS: OnceCell<Schemas> = OnceCell::const_new();

// Initializes and compiles JSON schemas for validation.
async fn get_schemas() -> &'static Schemas {
    SCHEMAS.get_or_init(|| async {
        let chat_completion_schema_json: Value = json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" },
                "object": { "type": "string", "enum": ["chat.completion"] },
                "created": { "type": "integer" },
                "model": { "type": "string" },
                "choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": { "type": "integer" },
                            "message": {
                                "type": "object",
                                "properties": {
                                    "role": { "type": "string" },
                                    "content": { "type": ["string", "null"] },
                                    "refusal": { "type": ["string", "null"] },
                                    "tool_calls": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": { "type": "string" },
                                                "type": { "type": "string", "enum": ["function"] },
                                                "function": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": { "type": "string" },
                                                        "arguments": { "type": "string" }
                                                    },
                                                    "required": ["name", "arguments"]
                                                }
                                            },
                                            "required": ["id", "type", "function"]
                                        }
                                    }
                                },
                                "required": ["role"]
                            },
                            "logprobs": {
                                "type": ["object", "null"],
                                "properties": {
                                    "content": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "token": { "type": "string" },
                                                "logprob": { "type": "number" }
                                            },
                                            "required": ["token", "logprob"]
                                        }
                                    }
                                },
                                "required": ["content"]
                            },
                            "finish_reason": { "type": "string", "enum": ["stop", "length", "tool_calls", "content_filter"] }
                        },
                        "required": ["index", "message", "finish_reason"]
                    }
                },
                "audio": {
                    "type": "object",
                    "properties": {
                        "bytesBase64Encoded": { "type": "string" },
                        "mimeType": { "type": "string" },
                        "fileSizeBytes": { "type": "integer" }
                    },
                    "required": ["bytesBase64Encoded", "mimeType", "fileSizeBytes"]
                },
                "usage": { "type": "object" }
            },
            "required": ["id", "object", "created", "model", "choices"]
        });

        // Embedded schema for the chat completion stream payload
        let chat_completion_stream_schema_json: Value = json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" },
                "object": { "type": "string", "enum": ["chat.completion.chunk"] },
                "created": { "type": "integer" },
                "model": { "type": "string" },
                "choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": { "type": "integer" },
                            "delta": {
                                "type": "object",
                                "properties": {
                                    "role": { "type": "string" },
                                    "content": { "type": "string" },
                                    "refusal": { "type": "string" },
                                    "tool_calls": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": { "type": ["string", "null"] },
                                                "type": { "type": ["string", "null"], "enum": ["function", null] },
                                                "function": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": { "type": ["string", "null"] },
                                                        "arguments": { "type": "string" }
                                                    },
                                                    "required": ["arguments"]
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "logprobs": {
                                "type": ["object", "null"],
                                "properties": {
                                    "content": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "token": { "type": "string" },
                                                "logprob": { "type": "number" }
                                            },
                                            "required": ["token", "logprob"]
                                        }
                                    }
                                },
                                "required": ["content"]
                            },
                            "finish_reason": { "type": ["string", "null"] }
                        },
                        "required": ["index", "delta"]
                    }
                }
            },
            "required": ["id", "object", "created", "model", "choices"]
        });

        // Embedded schema for the error payload
        let error_schema_json: Value = json!({
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" },
                        "type": { "type": "string" },
                        "param": { "type": "string" },
                        "code": { "type": "string" }
                    },
                    "required": ["message", "type", "param", "code"]
                }
            },
            "required": ["error"]
        });

        let chat_completion_schema = Arc::new(
            JSONSchema::options()
                .with_draft(Draft::Draft7)
                .compile(&chat_completion_schema_json)
                .expect("Failed to compile chat completion schema"),
        );

        let chat_completion_stream_schema = Arc::new(
            JSONSchema::options()
                .with_draft(Draft::Draft7)
                .compile(&chat_completion_stream_schema_json)
                .expect("Failed to compile chat completion stream schema"),
        );

        let error_schema = Arc::new(
            JSONSchema::options()
                .with_draft(Draft::Draft7)
                .compile(&error_schema_json)
                .expect("Failed to compile error schema"),
        );

        Schemas {
            chat_completion_schema,
            chat_completion_stream_schema,
            error_schema,
        }
    })
    .await
}

// test for standard successful response, including metadata, choices, and usage statistics.
#[tokio::test] async fn validate_full_response_format() {
    let schemas = get_schemas().await;
    let full_response = json!({
        "id": "chatcmpl-12345",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?",
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    });

    let result = schemas.chat_completion_schema.validate(&full_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Full non-streaming response validation failed: {:#?}\nOriginal payload: {}", error_messages, full_response);
    }
}

// test ensures that a single chunk from a streamed response, containing
// a delta of the message content, conforms to the chat completion stream JSON schema.
#[tokio::test] async fn validate_stream_response_format() {
    let schemas = get_schemas().await;
    let full_response = json!({
        "id": "chatcmpl-12345",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "logprobs": null,
                "finish_reason": null
            }
        ]
    });

    let result = schemas.chat_completion_stream_schema.validate(&full_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Full stream response validation failed: {:#?}\nOriginal payload: {}", error_messages, full_response);
    }
}

// This test checks the format of a response where the model requests to call
// a single function, including the function name and arguments.
#[tokio::test] async fn validate_single_tool_call_response_format() {
    let schemas = get_schemas().await;
    let tool_call_response = json!({
        "id": "chatcmpl-tool-call-123",
        "object": "chat.completion",
        "created": 1677652289,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1a2b3c4d",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "{\"location\": \"New York\", \"unit\": \"celsius\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let result = schemas.chat_completion_schema.validate(&tool_call_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Single tool call response validation failed: {:#?}\nOriginal payload: {}", error_messages, tool_call_response);
    }
}

// Validates a response with multiple tool calls.
#[tokio::test] async fn validate_multiple_tool_calls_response_format() {
    let schemas = get_schemas().await;
    let multiple_tool_calls_response = json!({
        "id": "chatcmpl-multi-tool-123",
        "object": "chat.completion",
        "created": 1677652290,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1a2b3c4d",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "{\"location\": \"New York\", \"unit\": \"celsius\"}"
                            }
                        },
                        {
                            "id": "call_2e5f6g7h",
                            "type": "function",
                            "function": {
                                "name": "get_stock_price",
                                "arguments": "{\"symbol\": \"GOOG\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let result = schemas.chat_completion_schema.validate(&multiple_tool_calls_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Multiple tool calls response validation failed: {:#?}\nOriginal payload: {}", error_messages, multiple_tool_calls_response);
    }
}

// Validates a chat completion response that includes audio data.
// This test checks the format of a response containing embedded audio,
// ensuring that the base64-encoded data, MIME type, and file size are correctly structured.
#[tokio::test] async fn validate_audio_response_format() {
    let schemas = get_schemas().await;
    let audio_response = json!({
        "id": "chatcmpl-audio-12345",
        "object": "chat.completion",
        "created": 1677652291,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here is the audio response.",
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ],
        "audio": {
            "bytesBase64Encoded": "SGVyZSBpcyBhdWRpbyBkYXRhIGluIGJhc2U2NCBlbmNvZGluZw==",
            "mimeType": "audio/L16;rate=24000",
            "fileSizeBytes": 45
        }
    });

    let result = schemas.chat_completion_schema.validate(&audio_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Audio response validation failed: {:#?}\nOriginal payload: {}", error_messages, audio_response);
    }
}

// Validates a chat completion response that includes log probabilities.
// This test ensures that the `logprobs` object, containing tokens and their
// corresponding log probabilities, is correctly formatted.
#[tokio::test] async fn validate_logprobs_response_format() {
    let schemas = get_schemas().await;
    let logprobs_response = json!({
        "id": "chatcmpl-logprobs-12345",
        "object": "chat.completion",
        "created": 1677652292,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world."
                },
                "logprobs": {
                    "content": [
                        { "token": "Hello", "logprob": -0.01 },
                        { "token": " world", "logprob": -0.02 },
                        { "token": ".", "logprob": -0.03 }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    });

    let result = schemas.chat_completion_schema.validate(&logprobs_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Logprobs response validation failed: {:#?}\nOriginal payload: {}", error_messages, logprobs_response);
    }
}

// This test checks that each chunk in a streamed tool call response, from the
// initial function definition to the streaming of arguments, conforms to the
// stream schema.
#[tokio::test] async fn validate_stream_tool_calls_response_format() {
    let schemas = get_schemas().await;
    let stream_tool_call_chunks = vec![
        json!({
            "id": "chatcmpl-stream-tool-call-123",
            "object": "chat.completion.chunk",
            "created": 1677652293,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1a2b3c4d",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": ""
                                }
                            }
                        ]
                    },
                    "logprobs": null,
                    "finish_reason": null
                }
            ]
        }),
        json!({
            "id": "chatcmpl-stream-tool-call-123",
            "object": "chat.completion.chunk",
            "created": 1677652293,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": null,
                                "type": null,
                                "function": {
                                    "name": null,
                                    "arguments": "{\"location\":\""
                                }
                            }
                        ]
                    },
                    "logprobs": null,
                    "finish_reason": null
                }
            ]
        }),
        json!({
            "id": "chatcmpl-stream-tool-call-123",
            "object": "chat.completion.chunk",
            "created": 1677652293,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": null,
                                "type": null,
                                "function": {
                                    "name": null,
                                    "arguments": "New York\"}"
                                }
                            }
                        ]
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }
            ]
        })
    ];

    for (i, chunk) in stream_tool_call_chunks.iter().enumerate() {
        let result = schemas.chat_completion_stream_schema.validate(chunk);
        if let Err(errors) = result {
            let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
            panic!("Streaming tool calls validation failed for chunk {}: {:#?}\nOriginal payload: {}", i, error_messages, chunk);
        }
    }
}

// Validates a response where the content was filtered.
// This test ensures that a response with a `content_filter` finish reason,
// which includes a `refusal` message and null content, is correctly formatted.
#[tokio::test] async fn validate_content_filtered_response_format() {
    let schemas = get_schemas().await;
    let content_filtered_response = json!({
        "id": "chatcmpl-content-filter-123",
        "object": "chat.completion",
        "created": 1677652294,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "I cannot provide a response to that request.",
                },
                "logprobs": null,
                "finish_reason": "content_filter"
            }
        ]
    });

    let result = schemas.chat_completion_schema.validate(&content_filtered_response);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("Content filtered response validation failed: {:#?}\nOriginal payload: {}", error_messages, content_filtered_response);
    }
}

// Validates the format of a 401 Unauthorized error response.
#[tokio::test] async fn validate_401_unauthorized_error() {
    let schemas = get_schemas().await;

    let error_body = json!({
        "error": {
            "message": "Incorrect API key provided: sk-1234...5678. You can find your API key at https://platform.openai.com/account/api-keys.",
            "type": "invalid_request_error",
            "param": "",
            "code": "invalid_api_key"
        }
    });

    let result = schemas.error_schema.validate(&error_body);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("401 Unauthorized error validation failed: {:#?}", error_messages);
    }
}

// Validates the format of a 400 Bad Request error response.
#[tokio::test] async fn validate_400_bad_request_error() {
    let schemas = get_schemas().await;

    let error_body = json!({
        "error": {
            "message": "Invalid JSON",
            "type": "invalid_request_error",
            "param": "",
            "code": "invalid_json"
        }
    });

    let result = schemas.error_schema.validate(&error_body);
    if let Err(errors) = result {
        let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!("400 Bad Request error validation failed: {:#?}", error_messages);
    }
}
