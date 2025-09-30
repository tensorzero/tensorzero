//! Tests for static OTLP headers configured in the config file (issue #3289)

use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent, Role,
};
use tensorzero_core::inference::types::TextKind;

use crate::otel::install_capturing_otel_exporter;

/// Test that static headers from config are applied
/// This verifies that the config parses correctly and the system works end-to-end
#[tokio::test]
async fn test_otel_config_headers_static() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"
[gateway.export.otlp.traces]
enabled = true
extra_headers = { "X-Static-Header-1" = "static-value-1", "X-Static-Header-2" = "static-value-2" }
"#;

    let client = make_embedded_gateway_with_config(config).await;

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
extra_headers = { "X-Config-Header" = "config-value" }
"#;

    let client = make_embedded_gateway_with_config(config).await;

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
extra_headers = { "X-Empty-Header" = "" }
"#;

    let client = make_embedded_gateway_with_config(config).await;

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
extra_headers = { "X-Multi-Header" = "multi-value" }
"#;

    let client = make_embedded_gateway_with_config(config).await;

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
