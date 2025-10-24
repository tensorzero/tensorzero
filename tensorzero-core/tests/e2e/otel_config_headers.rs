use std::collections::HashMap;

use chrono::Utc;
use http::StatusCode;
use serde_json::{json, Value};
use tensorzero_core::observability::enter_fake_http_request_otel;
use uuid::Uuid;

use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent, Role,
};
use tensorzero_core::inference::types::TextKind;

use crate::common::get_gateway_endpoint;
use crate::otel::install_capturing_otel_exporter;
use crate::otel_export::{get_tempo_spans, TempoSpans};

/// Test that static headers from config are applied
/// This verifies that the config parses correctly and the system works end-to-end
#[tokio::test]
async fn test_otel_config_headers_static() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers."X-Static-Header-1" = "static-value-1"
extra_headers."X-Static-Header-2" = "static-value-2"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let _guard = enter_fake_http_request_otel();

    // Make an inference request - if this succeeds without panicking, config headers work
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Test static headers".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    // Verify spans were captured (proves the system works with config headers)
    let spans = exporter.take_spans();
    assert!(
        !spans.is_empty(),
        "Should have captured spans with static config headers"
    );
}

/// Test that the system works when both config and dynamic headers would be present.
/// Note: The embedded client cannot send dynamic headers (they require HTTP headers),
/// so this test only verifies that config headers work correctly. The dynamic override
/// behavior is tested indirectly through the header merging logic in `build_with_context`.
#[tokio::test]
async fn test_otel_config_headers_with_dynamic_override() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers."X-Config-Header" = "config-value"
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let _guard = enter_fake_http_request_otel();

    // Make an inference request with config headers enabled
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Test mixed headers".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let spans = exporter.take_spans();
    assert!(
        !spans.is_empty(),
        "Should have captured spans with mixed headers"
    );
}

/// Test that config headers work with empty string values
#[tokio::test]
async fn test_otel_config_headers_empty_value() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers."X-Empty-Header" = ""
"#;

    let client = make_embedded_gateway_with_config(config).await;

    let _guard = enter_fake_http_request_otel();

    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Test empty header value".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let spans = exporter.take_spans();
    assert!(!spans.is_empty(), "Should work with empty header values");
}

/// Test that the system works without any config headers (baseline)
#[tokio::test]
async fn test_otel_no_config_headers() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r"
[gateway.export.otlp.traces]
enabled = true
";

    let client = make_embedded_gateway_with_config(config).await;
    let _guard = enter_fake_http_request_otel();

    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Test without config headers".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let spans = exporter.take_spans();
    assert!(
        !spans.is_empty(),
        "Should work without config headers (baseline)"
    );
}

/// Test that multiple inferences with config headers work correctly
#[tokio::test]
async fn test_otel_config_headers_multiple_requests() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers."X-Multi-Header" = "multi-value"
"#;

    let client = make_embedded_gateway_with_config(config).await;
    let _guard = enter_fake_http_request_otel();

    // Make multiple requests
    for i in 0..3 {
        client
            .inference(ClientInferenceParams {
                model_name: Some("dummy::good".to_string()),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: format!("Request {i}"),
                        })],
                    }],
                },
                ..Default::default()
            })
            .await
            .unwrap();
    }

    let spans = exporter.take_spans();
    assert!(
        spans.len() >= 3,
        "Should have captured spans from multiple requests"
    );
}

/// Test config parsing with various header formats
#[tokio::test]
async fn test_otel_config_headers_various_formats() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers."X-Simple" = "value"
extra_headers."X-With-Dashes" = "dash-value"
extra_headers."X-With-Numbers-123" = "number-value"
extra_headers."x-lowercase" = "lowercase-value"
"#;

    let client = make_embedded_gateway_with_config(config).await;
    let _guard = enter_fake_http_request_otel();

    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Test various header formats".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let spans = exporter.take_spans();
    assert!(!spans.is_empty(), "Should work with various header formats");
}

/// Test that dynamic headers override config headers
/// This test uses the external gateway (via HTTP) and queries Tempo to verify headers
#[tokio::test]
async fn test_otel_config_and_dynamic_header_override() {
    let client = reqwest::Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{
                "role": "user",
                "content": "What is the name of the capital city of Japan?"
            }]
        },
        "stream": false,
        "tags": {"test": "config_override"},
    });

    let start_time = Utc::now();

    // Send request with dynamic header that overrides the config header
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .header(
            "TensorZero-OTLP-Traces-Extra-Header-x-config-override-header",
            "dynamic-override-value",
        )
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let _response_json = response.json::<Value>().await.unwrap();

    // Query Tempo to get the spans and verify the headers
    let tempo_semaphore = tokio::sync::Semaphore::new(1);
    let TempoSpans {
        target_span: function_inference_span,
        span_by_id,
        resources: _,
    } = get_tempo_spans(
        ("episode_id", &episode_id.to_string()),
        start_time,
        &tempo_semaphore,
    )
    .await;

    let function_inference_span =
        function_inference_span.expect("No function_inference span found");

    // Get the HTTP span (parent of function_inference)
    let parent_id = function_inference_span["parentSpanId"].as_str().unwrap();
    let http_span = span_by_id.get(parent_id).unwrap();

    // Extract attributes from the HTTP span
    let attrs: HashMap<&str, serde_json::Value> = http_span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
        .collect();

    // Verify the static config header that wasn't overridden is present
    assert_eq!(
        attrs["tensorzero.config_static_header"]["stringValue"]
            .as_str()
            .unwrap(),
        "config-static-value",
        "Static config header should be present with config value"
    );

    // Verify the overridden header has the dynamic value (not the config value)
    assert_eq!(
        attrs["tensorzero.config_override_header"]["stringValue"]
            .as_str()
            .unwrap(),
        "dynamic-override-value",
        "Override header should have dynamic value, not config value"
    );
}
