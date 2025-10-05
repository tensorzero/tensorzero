#![expect(clippy::print_stderr)]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use opentelemetry::{trace::Status, KeyValue, SpanId, Value};
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
};
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    FeedbackParams, InferenceOutput, InferenceResponse, InferenceResponseChunk, Role,
};
use tensorzero_core::observability::setup_observability_with_exporter_override;
use tensorzero_core::observability::LogFormat;
use tensorzero_core::{config::OtlpTracesFormat, inference::types::TextKind};
use tokio_stream::StreamExt;
use uuid::Uuid;

type CapturedSpans = Arc<Mutex<Option<Vec<SpanData>>>>;

#[derive(Clone, Debug)]
pub struct CapturingOtelExporter {
    pub spans: CapturedSpans,
}

impl SpanExporter for CapturingOtelExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.spans
            .lock()
            .as_mut()
            .expect("Failed to lock spans mutex")
            .as_mut()
            .expect("CapturingOtelExporter is shut down")
            .extend(batch);
        Ok(())
    }
}

impl CapturingOtelExporter {
    pub fn take_spans(&self) -> Vec<SpanData> {
        let spans = self
            .spans
            .lock()
            .expect("Failed to lock spans mutex")
            .replace(Vec::new())
            .expect("CapturingExporter is already shut down");
        spans
    }
}

#[derive(Debug)]
pub struct SpanMap {
    pub root_spans: Vec<SpanData>,
    pub span_children: HashMap<SpanId, Vec<SpanData>>,
}

pub async fn install_capturing_otel_exporter() -> CapturingOtelExporter {
    let exporter = CapturingOtelExporter {
        spans: Arc::new(Mutex::new(Some(vec![]))),
    };
    let handle =
        setup_observability_with_exporter_override(LogFormat::Pretty, Some(exporter.clone()))
            .await
            .unwrap();
    handle.delayed_otel.unwrap().enable_otel().unwrap();
    exporter
}

pub fn build_span_map(spans: Vec<SpanData>) -> SpanMap {
    let mut map = SpanMap {
        root_spans: vec![],
        span_children: HashMap::new(),
    };
    for span in spans {
        if span.parent_span_id == SpanId::INVALID {
            map.root_spans.push(span);
        } else {
            map.span_children
                .entry(span.parent_span_id)
                .or_default()
                .push(span);
        }
    }
    map
}

pub fn attrs_to_map(attrs: &[KeyValue]) -> HashMap<String, Value> {
    let mut map = HashMap::with_capacity(attrs.len());
    for attr in attrs {
        if let Some(old) = map.insert(attr.key.as_str().to_string(), attr.value.clone()) {
            panic!(
                "Duplicate attribute key: {key} (old={old}, new={new})",
                key = attr.key,
                new = attr.value
            );
        }
    }
    map
}

// The tracing bug (https://github.com/tokio-rs/tracing/issues/2519) is sufficiently subtle that
// we want to ensure that we know how to reproduce it (so that we know our fix is actually doing something).
// See https://github.com/tensorzero/tensorzero/issues/3715
#[tokio::test]
pub async fn test_reproduce_tracing_bug() {
    // Set this test-only global variable to disable our fix.
    tensorzero_core::observability::tracing_bug::DISABLE_TRACING_BUG_WORKAROUND
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let exporter = install_capturing_otel_exporter().await;

    let config = "
    [gateway.export.otlp.traces]
    enabled = true
    "
    .to_string();

    let client = make_embedded_gateway_with_config(&config).await;

    // Make an inference to initialize the tracing call-site cache
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            tags: HashMap::from([("first_request".to_string(), "first_value".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    // Ignore the spans we just created
    exporter.take_spans();

    // Trigger the tracing bug by calling `log::log_enabled!` with a disabled event
    // The tracing integration with the `log` crate will cause the next span we emit to be discarded
    // For some reason, the macro returns 'true' even though the event is disabled - most likely,
    // the `tracing` crate is being overly conservative in computing the value to return to the `log` crate
    assert!(log::log_enabled!(target: "fake-target", log::Level::Info));

    // Make a new inference, which should cause a span to get lost
    client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            tags: HashMap::from([("first_request".to_string(), "first_value".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    // We should now see 'variant_inference' as a root span, and have the 'function_inference' span missing
    let all_spans = exporter.take_spans();
    let spans = build_span_map(all_spans);
    assert_eq!(spans.root_spans[0].name, "variant_inference");
    assert_eq!(spans.root_spans.len(), 1);
}

#[tokio::test]
pub async fn test_capture_simple_inference_spans_genai_tags_non_streaming() {
    test_capture_simple_inference_spans(OtlpTracesFormat::OpenTelemetry, "opentelemetry", false)
        .await;
}
#[tokio::test]
pub async fn test_capture_simple_inference_spans_genai_tags_streaming() {
    test_capture_simple_inference_spans(OtlpTracesFormat::OpenTelemetry, "opentelemetry", true)
        .await;
}

#[tokio::test]
pub async fn test_capture_simple_inference_spans_openinference_tags_non_streaming() {
    test_capture_simple_inference_spans(OtlpTracesFormat::OpenInference, "openinference", false)
        .await;
}

#[tokio::test]
pub async fn test_capture_simple_inference_spans_openinference_tags_streaming() {
    test_capture_simple_inference_spans(OtlpTracesFormat::OpenInference, "openinference", true)
        .await;
}

struct ResponseData {
    inference_id: Uuid,
    episode_id: Uuid,
    input_tokens: i64,
    output_tokens: i64,
    total_tokens: i64,
}

async fn make_non_streaming_inference(client: &Client) -> ResponseData {
    let res: InferenceOutput = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
            ]),
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(output) = res else {
        panic!("Expected non-streaming output, got: {res:#?}");
    };

    let InferenceResponse::Chat(response) = output else {
        panic!("Expected chat response, got: {output:#?}");
    };

    ResponseData {
        inference_id: response.inference_id,
        episode_id: response.episode_id,
        input_tokens: response.usage.input_tokens as i64,
        output_tokens: response.usage.output_tokens as i64,
        total_tokens: (response.usage.input_tokens + response.usage.output_tokens) as i64,
    }
}

async fn make_streaming_inference(client: &Client) -> ResponseData {
    let res: InferenceOutput = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
            ]),
            stream: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::Streaming(mut stream) = res else {
        panic!("Expected streaming output, got: {res:#?}");
    };

    let mut inference_id = None;
    let mut episode_id = None;
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut total_tokens = 0;
    while let Some(chunk) = stream.next().await {
        let InferenceResponseChunk::Chat(response) = chunk.clone().unwrap() else {
            panic!("Expected chat response, got: {chunk:#?}");
        };
        inference_id = Some(response.inference_id);
        episode_id = Some(response.episode_id);
        if let Some(usage) = response.usage {
            input_tokens += usage.input_tokens as i64;
            output_tokens += usage.output_tokens as i64;
            total_tokens += (usage.input_tokens + usage.output_tokens) as i64;
        }
    }

    ResponseData {
        inference_id: inference_id.unwrap(),
        episode_id: episode_id.unwrap(),
        input_tokens,
        output_tokens,
        total_tokens,
    }
}

pub async fn test_capture_simple_inference_spans(
    mode: OtlpTracesFormat,
    config_mode: &str,
    streaming: bool,
) {
    let exporter = install_capturing_otel_exporter().await;

    let config = format!(
        "
    [gateway.export.otlp.traces]
    enabled = true
    format = \"{config_mode}\"
    "
    );

    let client = make_embedded_gateway_with_config(&config).await;
    let response_data = if streaming {
        make_streaming_inference(&client).await
    } else {
        make_non_streaming_inference(&client).await
    };
    let ResponseData {
        inference_id,
        episode_id,
        input_tokens,
        output_tokens,
        total_tokens,
    } = response_data;

    let all_spans = exporter.take_spans();
    let num_spans = all_spans.len();
    let spans = build_span_map(all_spans);
    eprintln!("Spans: {spans:#?}");
    let [root_span] = spans.root_spans.as_slice() else {
        panic!("Expected one root span: {:#?}", spans.root_spans);
    };
    // Since we're using the embedded gateway, the root span will be `function_inference`
    // (we won't have a top-level HTTP span)
    assert_eq!(root_span.name, "function_inference");
    assert_eq!(root_span.status, Status::Unset);
    let root_attr_map = attrs_to_map(&root_span.attributes);
    assert_eq!(root_attr_map["model_name"], "dummy::good".into());
    assert_eq!(
        root_attr_map["inference_id"],
        inference_id.to_string().into()
    );
    assert_eq!(root_attr_map["episode_id"], episode_id.to_string().into());
    assert_eq!(root_attr_map.get("function_name"), None);
    assert_eq!(root_attr_map.get("variant_name"), None);
    assert_eq!(
        root_attr_map.get("tags.first_tag").cloned(),
        Some("first_value".to_string().into())
    );
    assert_eq!(
        root_attr_map.get("tags.second_tag").cloned(),
        Some("second_value".to_string().into())
    );
    // Check that there are no other takes
    let tag_count = root_attr_map
        .iter()
        .filter(|(k, _)| k.starts_with("tags."))
        .count();
    assert_eq!(tag_count, 2);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let [variant_span] = root_children.as_slice() else {
        panic!("Expected one child span: {root_children:#?}");
    };

    assert_eq!(variant_span.name, "variant_inference");
    assert_eq!(variant_span.status, Status::Unset);
    let variant_attr_map = attrs_to_map(&variant_span.attributes);
    assert_eq!(
        variant_attr_map["function_name"],
        "tensorzero::default".into()
    );
    assert_eq!(variant_attr_map["variant_name"], "dummy::good".into());
    assert_eq!(variant_attr_map["stream"], streaming.into());

    let variant_children = &spans.span_children[&variant_span.span_context.span_id()];
    let [model_span] = variant_children.as_slice() else {
        panic!("Expected one child span: {variant_children:#?}");
    };

    assert_eq!(model_span.name, "model_inference");
    assert_eq!(model_span.status, Status::Unset);
    let model_attr_map = attrs_to_map(&model_span.attributes);
    assert_eq!(model_attr_map["model_name"], "dummy::good".into());
    assert_eq!(model_attr_map["stream"], streaming.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
    assert_eq!(model_provider_span.status, Status::Unset);
    let model_provider_attr_map = attrs_to_map(&model_provider_span.attributes);
    assert_eq!(model_provider_attr_map["provider_name"], "dummy".into());

    match mode {
        OtlpTracesFormat::OpenTelemetry => {
            assert_eq!(
                model_provider_attr_map["gen_ai.operation.name"],
                "chat".into()
            );
            assert_eq!(model_provider_attr_map["gen_ai.system"], "dummy".into());
            assert_eq!(
                model_provider_attr_map["gen_ai.request.model"],
                "good".into()
            );
            assert!(!model_provider_attr_map.contains_key("openinference.span.kind"));
            assert!(!model_provider_attr_map.contains_key("llm.system"));
            assert!(!model_provider_attr_map.contains_key("llm.model_name"));

            assert_eq!(
                model_provider_attr_map["gen_ai.usage.input_tokens"],
                input_tokens.into()
            );
            assert_eq!(
                model_provider_attr_map["gen_ai.usage.output_tokens"],
                output_tokens.into()
            );
            assert_eq!(
                model_provider_attr_map["gen_ai.usage.total_tokens"],
                total_tokens.into()
            );
            assert!(!model_provider_attr_map.contains_key("llm.token_count.prompt"));
            assert!(!model_provider_attr_map.contains_key("llm.token_count.completion"));
            assert!(!model_provider_attr_map.contains_key("llm.token_count.total"));
        }
        OtlpTracesFormat::OpenInference => {
            assert_eq!(
                model_provider_attr_map["openinference.span.kind"],
                "LLM".into()
            );
            assert_eq!(model_provider_attr_map["llm.system"], "dummy".into());
            assert_eq!(model_provider_attr_map["llm.model_name"], "good".into());
            assert!(!model_provider_attr_map.contains_key("gen_ai.operation.name"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.system"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.request.model"));

            assert_eq!(
                model_provider_attr_map["llm.token_count.prompt"],
                input_tokens.into()
            );
            assert_eq!(
                model_provider_attr_map["llm.token_count.completion"],
                output_tokens.into()
            );
            assert_eq!(
                model_provider_attr_map["llm.token_count.total"],
                total_tokens.into()
            );
            // We currently don't have input/output attributes implemented for streaming inferences
            if streaming {
                // When we implement input/output attributes for streaming inferences, remove these checks
                assert!(!model_provider_attr_map.contains_key("input.mime_type"));
                assert!(!model_provider_attr_map.contains_key("output.mime_type"));
                assert!(!model_provider_attr_map.contains_key("input.value"));
                assert!(!model_provider_attr_map.contains_key("output.value"));
            } else {
                assert_eq!(
                    model_provider_attr_map["input.mime_type"],
                    "application/json".into()
                );
                assert_eq!(
                    model_provider_attr_map["output.mime_type"],
                    "application/json".into()
                );
                assert_eq!(model_provider_attr_map["input.value"], "raw request".into());
                assert_eq!(
                    model_provider_attr_map["output.value"],
                    "{\n  \"id\": \"id\",\n  \"object\": \"text.completion\",\n  \"created\": 1618870400,\n  \"model\": \"text-davinci-002\",\n  \"choices\": [\n    {\n      \"text\": \"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\",\n      \"index\": 0,\n      \"logprobs\": null,\n      \"finish_reason\": null\n    }\n  ]\n}".into()
                );
            }

            assert!(!model_provider_attr_map.contains_key("gen_ai.usage.input_tokens"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.usage.output_tokens"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.usage.total_tokens"));
        }
    }
    assert_eq!(model_attr_map["stream"], streaming.into());

    assert_eq!(
        spans
            .span_children
            .get(&model_provider_span.span_context.span_id()),
        None
    );

    assert_eq!(num_spans, 4);
}

#[test]
pub fn test_capture_model_error_genai_tags() {
    test_capture_model_error(OtlpTracesFormat::OpenTelemetry, "opentelemetry");
}

#[test]
pub fn test_capture_model_error_openinference_tags() {
    test_capture_model_error(OtlpTracesFormat::OpenInference, "openinference");
}

pub fn test_capture_model_error(mode: OtlpTracesFormat, config_mode: &str) {
    let episode_uuid = Uuid::now_v7();

    let config = format!(
        "
    [gateway.export.otlp.traces]
    enabled = true
    format = \"{config_mode}\"
    "
    );

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (exporter, _err) = runtime.block_on(async {
        let exporter = install_capturing_otel_exporter().await;
        let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;
        let _err = client
            .inference(ClientInferenceParams {
                episode_id: Some(episode_uuid),
                model_name: Some("openai::missing-model-name".to_string()),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: "What is your name?".to_string(),
                        })],
                    }],
                },
                ..Default::default()
            })
            .await
            .unwrap_err();
        (exporter, _err)
    });
    // Shut down the runtime to wait for all `tokio::spawn` tasks to finish
    // (so that all spans are exported)
    drop(runtime);

    let all_spans = exporter.take_spans();
    let num_spans = all_spans.len();
    let spans = build_span_map(all_spans);

    let [root_span] = spans.root_spans.as_slice() else {
        panic!("Expected one root span: {:#?}", spans.root_spans);
    };
    // Since we're using the embedded gateway, the root span will be `function_inference`
    // (we won't have a top-level HTTP span)
    assert_eq!(root_span.name, "function_inference");
    assert_eq!(
        root_span.status,
        Status::Error {
            description: "".into()
        }
    );
    let root_attr_map = attrs_to_map(&root_span.attributes);
    assert_eq!(
        root_attr_map["model_name"],
        "openai::missing-model-name".into()
    );
    assert_eq!(root_attr_map["episode_id"], episode_uuid.to_string().into());
    assert_eq!(root_attr_map.get("function_name"), None);
    assert_eq!(root_attr_map.get("variant_name"), None);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let [variant_span] = root_children.as_slice() else {
        panic!("Expected one child span: {root_children:#?}");
    };

    assert_eq!(variant_span.name, "variant_inference");
    assert_eq!(variant_span.status, Status::Unset);
    let variant_attr_map = attrs_to_map(&variant_span.attributes);
    assert_eq!(
        variant_attr_map["function_name"],
        "tensorzero::default".into()
    );
    assert_eq!(
        variant_attr_map["variant_name"],
        "openai::missing-model-name".into()
    );
    assert_eq!(variant_attr_map["stream"], false.into());

    let variant_children = &spans.span_children[&variant_span.span_context.span_id()];
    let [model_span] = variant_children.as_slice() else {
        panic!("Expected one child span: {variant_children:#?}");
    };

    assert_eq!(model_span.name, "model_inference");
    assert_eq!(
        model_span.status,
        Status::Error {
            description: "".into()
        }
    );
    let model_attr_map = attrs_to_map(&model_span.attributes);
    assert_eq!(
        model_attr_map["model_name"],
        "openai::missing-model-name".into()
    );
    assert_eq!(model_attr_map["stream"], false.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
    assert_eq!(
        model_provider_span.status,
        Status::Error {
            description: "".into()
        }
    );
    assert_eq!(
        model_provider_span.events.len(),
        1,
        "Unexpected number of events: {model_provider_span:#?}",
    );
    assert!(
        model_provider_span.events[0]
            .name
            .starts_with("Error from openai server:"),
        "Unexpected span event: {:?}",
        model_provider_span.events[0]
    );
    let model_provider_attr_map = attrs_to_map(&model_provider_span.attributes);
    assert_eq!(model_provider_attr_map["provider_name"], "openai".into());

    match mode {
        OtlpTracesFormat::OpenTelemetry => {
            assert_eq!(
                model_provider_attr_map["gen_ai.operation.name"],
                "chat".into()
            );
            assert_eq!(model_provider_attr_map["gen_ai.system"], "openai".into());
            assert_eq!(
                model_provider_attr_map["gen_ai.request.model"],
                "missing-model-name".into()
            );
        }
        OtlpTracesFormat::OpenInference => {
            assert_eq!(
                model_provider_attr_map["openinference.span.kind"],
                "LLM".into()
            );
            assert_eq!(model_provider_attr_map["llm.system"], "openai".into());
            assert_eq!(
                model_provider_attr_map["input.mime_type"],
                "application/json".into()
            );
            assert_eq!(
                model_provider_attr_map["output.mime_type"],
                "application/json".into()
            );
            assert_eq!(
                model_provider_attr_map["input.value"],
                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is your name?\"}],\"model\":\"missing-model-name\",\"stream\":false}".into()
            );
            // Don't check the exact error message from OpenAI, to prevent this test from breaking whenever OpenAI changes the error details
            assert!(
                model_provider_attr_map["output.value"]
                    .as_str()
                    .contains("model_not_found"),
                "Unexpected output value: {:?}",
                model_provider_attr_map["output.value"]
            );
        }
    }

    assert_eq!(model_attr_map["stream"], false.into());

    assert_eq!(
        spans
            .span_children
            .get(&model_provider_span.span_context.span_id()),
        None
    );

    assert_eq!(num_spans, 4);
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_capture_feedback_spans() {
    let exporter = install_capturing_otel_exporter().await;

    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let res = client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
            ]),
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::NonStreaming(output) = res else {
        panic!("Expected non-streaming output, got: {res:#?}");
    };

    let _feedback_res = client
        .feedback(FeedbackParams {
            inference_id: Some(output.inference_id()),
            metric_name: "task_success".to_string(),
            value: true.into(),
            tags: HashMap::from([
                ("my_tag".to_string(), "my_value".to_string()),
                ("my_tag2".to_string(), "my_value2".to_string()),
            ]),
            ..Default::default()
        })
        .await
        .unwrap();

    let all_spans = exporter.take_spans();
    let mut spans = build_span_map(all_spans);
    // We should have a feedback span and a function_inference span
    assert_eq!(spans.root_spans.len(), 2);
    spans.root_spans.sort_by_key(|span| span.name.clone());
    assert_eq!(spans.root_spans[0].name, "feedback");
    assert_eq!(spans.root_spans[1].name, "function_inference");

    // We've already checked the function_inference span in the previous test,
    // so just check the feedback span
    let feedback_span = &spans.root_spans[0];
    let feedback_attr_map = attrs_to_map(&feedback_span.attributes);
    assert_eq!(
        feedback_attr_map["inference_id"],
        output.inference_id().to_string().into()
    );
    assert_eq!(
        feedback_attr_map["tags.my_tag"],
        "my_value".to_string().into()
    );
    assert_eq!(
        feedback_attr_map["tags.my_tag2"],
        "my_value2".to_string().into()
    );
    assert!(!feedback_attr_map.contains_key("episode_id"));
    assert_eq!(feedback_attr_map["metric_name"], "task_success".into());

    assert_eq!(
        spans
            .span_children
            .get(&feedback_span.span_context.span_id()),
        None,
        "feedback span should have no children"
    );
}
