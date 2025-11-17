#![expect(clippy::print_stderr)]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use opentelemetry::{trace::Status, KeyValue, SpanId, Value};
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
};
use tensorzero::{
    test_helpers::{
        make_embedded_gateway_with_config, make_embedded_gateway_with_config_and_postgres,
    },
    InferenceParams,
};
use tensorzero::{
    Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    FeedbackParams, InferenceOutput, InferenceResponse, InferenceResponseChunk, Role, Usage,
};
use tensorzero_core::observability::{
    enter_fake_http_request_otel, setup_observability_with_exporter_override,
};
use tensorzero_core::{config::OtlpTracesFormat, inference::types::TextKind};
use tensorzero_core::{
    endpoints::inference::ChatCompletionInferenceParams, observability::LogFormat,
};
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

#[derive(Clone, Copy, Debug, Default)]
struct OTelUsage {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

impl OTelUsage {
    fn zero() -> Self {
        OTelUsage {
            input_tokens: Some(0),
            output_tokens: Some(0),
        }
    }

    fn total_tokens(&self) -> Option<i64> {
        match (self.input_tokens, self.output_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt + completion),
            _ => None,
        }
    }

    /// Sum `OTelUsage` and `Usage` instances.
    /// `None` contaminates on both sides.
    fn sum_usage_strict(&mut self, other: &Usage) {
        self.input_tokens = match (self.input_tokens, other.input_tokens) {
            (Some(a), Some(b)) => Some(a + b as i64),
            _ => None,
        };

        self.output_tokens = match (self.output_tokens, other.output_tokens) {
            (Some(a), Some(b)) => Some(a + b as i64),
            _ => None,
        };
    }
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

    let _guard = enter_fake_http_request_otel();

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
    model_name: String,
    streaming: bool,
    inference_id: Uuid,
    episode_id: Uuid,
    usage: OTelUsage,
    estimated_tokens: i64,
    underestimate: bool,
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
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    max_tokens: Some(1000),
                    ..Default::default()
                },
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
                ("user_id".to_string(), Uuid::now_v7().to_string()),
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
        model_name: "dummy::good".to_string(),
        streaming: false,
        inference_id: response.inference_id,
        episode_id: response.episode_id,
        usage: OTelUsage {
            input_tokens: response.usage.input_tokens.map(|x| x as i64),
            output_tokens: response.usage.output_tokens.map(|x| x as i64),
        },
        underestimate: false,
        estimated_tokens: 1009,
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
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    max_tokens: Some(1000),
                    ..Default::default()
                },
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
                ("user_id".to_string(), Uuid::now_v7().to_string()),
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
    // `usage` is `None` until we receive a chunk with usage information
    let mut usage: Option<OTelUsage> = None;
    while let Some(chunk) = stream.next().await {
        let InferenceResponseChunk::Chat(response) = chunk.clone().unwrap() else {
            panic!("Expected chat response, got: {chunk:#?}");
        };
        inference_id = Some(response.inference_id);
        episode_id = Some(response.episode_id);
        if let Some(chunk_usage) = response.usage {
            // `usage` will be `None` if this is the first chunk with usage information....
            if usage.is_none() {
                // ... so initialize it to zero ...
                usage = Some(OTelUsage::zero());
            }
            // ...and then add the chunk usage to it (handling `None` fields)
            if let Some(ref mut u) = usage {
                u.sum_usage_strict(&chunk_usage);
            }
        }
    }

    // `usage` will be None if we don't see usage in any chunks, in which case we take the default value (fields as `None`)
    let usage = usage.unwrap_or_default();

    ResponseData {
        model_name: "dummy::good".to_string(),
        streaming: true,
        inference_id: inference_id.unwrap(),
        episode_id: episode_id.unwrap(),
        usage,
        estimated_tokens: 1009,
        underestimate: false,
    }
}

#[tokio::test]
async fn test_stream_fatal_error_usage() {
    let exporter = install_capturing_otel_exporter().await;

    let config = r#"

    [rate_limiting]
    enabled = true

    [[rate_limiting.rules]]
    priority = 0
    model_inferences_per_minute = 1000
    tokens_per_minute = 1000000
    scope = [
    { tag_key = "user_id", tag_value = "tensorzero::each" }
    ]

    [gateway.export.otlp.traces]
    enabled = true
    "#
    .to_string();

    let _guard = enter_fake_http_request_otel();

    let client = make_embedded_gateway_with_config_and_postgres(&config).await;
    let model_name = "dummy::fatal_stream_error";
    let res: InferenceOutput = client
        .inference(ClientInferenceParams {
            model_name: Some(model_name.to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "What is your name?".to_string(),
                    })],
                }],
            },
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    max_tokens: Some(1000),
                    ..Default::default()
                },
            },
            tags: HashMap::from([
                ("first_tag".to_string(), "first_value".to_string()),
                ("second_tag".to_string(), "second_value".to_string()),
                ("user_id".to_string(), Uuid::now_v7().to_string()),
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
    // `usage` is `None` until we receive a chunk with usage information
    let mut usage: Option<OTelUsage> = None;
    let mut all_chunks = vec![];
    while let Some(chunk) = stream.next().await {
        all_chunks.push(chunk.clone());
        match chunk {
            Ok(InferenceResponseChunk::Chat(response)) => {
                inference_id = Some(response.inference_id);
                episode_id = Some(response.episode_id);

                if let Some(response_usage) = response.usage {
                    // `usage` will be `None` if this is the first chunk with usage information....
                    if usage.is_none() {
                        // ... so initialize it to zero ...
                        usage = Some(OTelUsage::zero());
                    }
                    // ...and then add the chunk usage to it (handling `None` fields)
                    if let Some(ref mut u) = usage {
                        u.sum_usage_strict(&response_usage);
                    }
                }
            }
            Ok(_) => panic!("Expected chat response, got: {chunk:#?}"),
            Err(e) => {
                // Once we encounter a fatal error, the stream should end, and spans should be reported
                // The 'dummy::fatal_stream_error' model will try to produce more chunks after 5 seconds,
                // but the client should not see them, and the OTEL span should get reported anyway.
                assert!(
                    e.to_string().contains("Dummy fatal error"),
                    "Unexpected error: {e:#?}"
                );
                let start = Instant::now();
                let next_chunk = stream.next().await;
                let elapsed = start.elapsed();
                assert!(
                    elapsed < Duration::from_secs(1),
                    "Stream should end within 1 second of fatal error, but took {elapsed:?}"
                );
                assert!(
                    next_chunk.is_none(),
                    "Expected stream to end after fatal error, got: {next_chunk:#?}"
                );
                check_spans(
                    &exporter,
                    ResponseData {
                        model_name: model_name.to_string(),
                        streaming: true,
                        inference_id: inference_id.unwrap(),
                        episode_id: episode_id.unwrap(),
                        usage: usage.unwrap_or_default(),
                        estimated_tokens: 1009,
                        underestimate: true,
                    },
                    OtlpTracesFormat::OpenTelemetry,
                );
                return;
            }
        }
    }
    panic!("Expected to see fatal error in stream, but saw: {all_chunks:#?}");
}

fn check_spans(
    exporter: &CapturingOtelExporter,
    response_data: ResponseData,
    mode: OtlpTracesFormat,
) {
    let ResponseData {
        inference_id,
        episode_id,
        usage,
        estimated_tokens,
        model_name,
        streaming,
        underestimate,
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
    assert_eq!(root_span.status, Status::Ok);
    let root_attr_map = attrs_to_map(&root_span.attributes);
    assert_eq!(root_attr_map["model_name"], model_name.clone().into());
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
    assert!(root_attr_map.contains_key("tags.user_id"));
    // Check that there are no other tags
    let tag_count = root_attr_map
        .iter()
        .filter(|(k, _)| k.starts_with("tags."))
        .count();
    assert_eq!(tag_count, 3);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let [variant_span] = root_children.as_slice() else {
        panic!("Expected one child span: {root_children:#?}");
    };

    assert_eq!(variant_span.name, "variant_inference");
    assert_eq!(variant_span.status, Status::Ok);
    let variant_attr_map = attrs_to_map(&variant_span.attributes);
    assert_eq!(
        variant_attr_map["function_name"],
        "tensorzero::default".into()
    );
    assert_eq!(variant_attr_map["variant_name"], model_name.clone().into());
    assert_eq!(variant_attr_map["stream"], streaming.into());

    let variant_children = &spans.span_children[&variant_span.span_context.span_id()];
    let [model_span] = variant_children.as_slice() else {
        panic!("Expected one child span: {variant_children:#?}");
    };

    assert_eq!(model_span.name, "model_inference");
    assert_eq!(model_span.status, Status::Ok);
    let model_attr_map = attrs_to_map(&model_span.attributes);
    assert_eq!(model_attr_map["model_name"], model_name.clone().into());
    assert_eq!(model_attr_map["stream"], streaming.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
    assert_eq!(model_provider_span.status, Status::Ok);
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
                model_name
                    .strip_prefix("dummy::")
                    .unwrap()
                    .to_string()
                    .into()
            );
            assert!(!model_provider_attr_map.contains_key("openinference.span.kind"));
            assert!(!model_provider_attr_map.contains_key("llm.system"));
            assert!(!model_provider_attr_map.contains_key("llm.model_name"));

            if let Some(input_tokens) = usage.input_tokens {
                assert_eq!(
                    model_provider_attr_map["gen_ai.usage.input_tokens"],
                    input_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("gen_ai.usage.input_tokens"));
            }
            if let Some(output_tokens) = usage.output_tokens {
                assert_eq!(
                    model_provider_attr_map["gen_ai.usage.output_tokens"],
                    output_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("gen_ai.usage.output_tokens"));
            }
            if let Some(total_tokens) = usage.total_tokens() {
                assert_eq!(
                    model_provider_attr_map["gen_ai.usage.total_tokens"],
                    total_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("gen_ai.usage.total_tokens"));
            }
            assert!(!model_provider_attr_map.contains_key("llm.token_count.prompt"));
            assert!(!model_provider_attr_map.contains_key("llm.token_count.completion"));
            assert!(!model_provider_attr_map.contains_key("llm.token_count.total"));
        }
        OtlpTracesFormat::OpenInference => {
            assert_eq!(root_attr_map["openinference.span.kind"], "CHAIN".into());
            assert_eq!(variant_attr_map["openinference.span.kind"], "CHAIN".into());
            assert_eq!(
                model_provider_attr_map["openinference.span.kind"],
                "LLM".into()
            );
            assert_eq!(model_provider_attr_map["llm.system"], "dummy".into());
            assert_eq!(
                model_provider_attr_map["llm.model_name"],
                model_name
                    .strip_prefix("dummy::")
                    .unwrap()
                    .to_string()
                    .into()
            );
            assert!(!model_provider_attr_map.contains_key("gen_ai.operation.name"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.system"));
            assert!(!model_provider_attr_map.contains_key("gen_ai.request.model"));

            if let Some(input_tokens) = usage.input_tokens {
                assert_eq!(
                    model_provider_attr_map["llm.token_count.prompt"],
                    input_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("llm.token_count.prompt"));
            }
            if let Some(output_tokens) = usage.output_tokens {
                assert_eq!(
                    model_provider_attr_map["llm.token_count.completion"],
                    output_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("llm.token_count.completion"));
            }
            if let Some(total_tokens) = usage.total_tokens() {
                assert_eq!(
                    model_provider_attr_map["llm.token_count.total"],
                    total_tokens.into()
                );
            } else {
                assert!(!model_provider_attr_map.contains_key("llm.token_count.total"));
            }
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

    let rate_limit_spans = spans
        .span_children
        .get(&model_provider_span.span_context.span_id())
        .unwrap();
    let [consume_ticket_span, return_ticket_span] = rate_limit_spans.as_slice() else {
        panic!("Expected two rate limit spans: {rate_limit_spans:#?}");
    };

    assert_eq!(consume_ticket_span.name, "rate_limiting_consume_tickets");
    assert_eq!(consume_ticket_span.status, Status::Ok);
    let mut consume_ticket_attr_map = attrs_to_map(&consume_ticket_span.attributes);
    remove_unstable_attrs(&mut consume_ticket_attr_map);
    assert_eq!(
        consume_ticket_attr_map,
        HashMap::from([
            (
                "scope_info.tags.first_tag".to_string(),
                "first_value".into()
            ),
            (
                "scope_info.tags.second_tag".to_string(),
                "second_value".into()
            ),
            (
                "estimated_usage.tokens".to_string(),
                estimated_tokens.into()
            ),
            ("estimated_usage.model_inferences".to_string(), 1.into()),
            ("level".to_string(), "INFO".into()),
        ])
    );

    assert_eq!(return_ticket_span.name, "rate_limiting_return_tickets");
    assert_eq!(return_ticket_span.status, Status::Ok);
    let mut return_ticket_attr_map = attrs_to_map(&return_ticket_span.attributes);
    remove_unstable_attrs(&mut return_ticket_attr_map);
    assert_eq!(
        return_ticket_attr_map,
        HashMap::from([
            (
                "actual_usage.tokens".to_string(),
                usage.total_tokens().unwrap_or_default().into()
            ),
            ("actual_usage.model_inferences".to_string(), 1.into()),
            ("underestimate".to_string(), underestimate.into()),
            ("level".to_string(), "INFO".into()),
        ])
    );

    assert_eq!(num_spans, 6);
}

fn remove_unstable_attrs(attrs: &mut HashMap<String, Value>) {
    // These values are either random or can easily change between commits,
    // so remove them (and assert that they exist)
    attrs.remove("code.namespace");
    attrs.remove("code.module.name");
    attrs.remove("code.file.path");
    attrs.remove("code.line.number");
    attrs.remove("thread.id");
    attrs.remove("thread.name");
    attrs.remove("code.filepath");
    attrs.remove("code.lineno");
    attrs.remove("busy_ns");
    attrs.remove("idle_ns");
    attrs.remove("target");
    // We use a random value to prevent concurrent tests from trampling on each others' rate limiting buckets
    attrs.remove("tags.user_id");
    attrs.remove("scope_info.tags.user_id");
}

pub async fn test_capture_simple_inference_spans(
    mode: OtlpTracesFormat,
    config_mode: &str,
    streaming: bool,
) {
    let exporter = install_capturing_otel_exporter().await;

    let config = format!(
        r#"
    [rate_limiting]
    enabled = true

    [[rate_limiting.rules]]
    priority = 0
    model_inferences_per_minute = 1000
    tokens_per_minute = 1000000
    scope = [
    {{ tag_key = "user_id", tag_value = "tensorzero::each" }}
    ]

    [gateway.export.otlp.traces]
    enabled = true
    format = "{config_mode}"
    "#
    );

    let client = make_embedded_gateway_with_config_and_postgres(&config).await;
    let _guard = enter_fake_http_request_otel();
    let response_data = if streaming {
        make_streaming_inference(&client).await
    } else {
        make_non_streaming_inference(&client).await
    };
    // Explicitly drop the client, which will block on all OpenTelemetry spans being exported
    drop(client);
    check_spans(&exporter, response_data, mode);
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
        r#"
    [rate_limiting]
    enabled = true

    [[rate_limiting.rules]]
    priority = 0
    model_inferences_per_minute = 1000
    tokens_per_minute = 1000000
    scope = [
    {{ tag_key = "user_id", tag_value = "tensorzero::each" }}
    ]

    [gateway.export.otlp.traces]
    enabled = true
    format = "{config_mode}"
    "#
    );

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (exporter, _err) = runtime.block_on(async {
        let exporter = install_capturing_otel_exporter().await;
        let _guard = enter_fake_http_request_otel();
        let client =
            tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
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
                params: InferenceParams {
                    chat_completion: ChatCompletionInferenceParams {
                        max_tokens: Some(1000),
                        ..Default::default()
                    },
                },
                tags: HashMap::from([
                    ("first_tag".to_string(), "first_value".into()),
                    ("second_tag".to_string(), "second_value".into()),
                    ("user_id".to_string(), Uuid::now_v7().to_string()),
                ]),
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
    assert_eq!(variant_span.status, Status::Ok);
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
               "{\"messages\":[{\"role\":\"user\",\"content\":\"What is your name?\"}],\"model\":\"missing-model-name\",\"max_completion_tokens\":1000,\"stream\":false}".into()
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
    let rate_limit_spans = spans
        .span_children
        .get(&model_provider_span.span_context.span_id())
        .unwrap();
    // The model provider errored, so we shouldn't try to return tickets
    let [consume_ticket_span] = rate_limit_spans.as_slice() else {
        panic!("Expected one rate limit span: {rate_limit_spans:#?}");
    };
    assert_eq!(consume_ticket_span.name, "rate_limiting_consume_tickets");
    assert_eq!(consume_ticket_span.status, Status::Ok);
    let mut consume_ticket_attr_map = attrs_to_map(&consume_ticket_span.attributes);
    remove_unstable_attrs(&mut consume_ticket_attr_map);

    assert_eq!(
        consume_ticket_attr_map,
        HashMap::from([
            (
                "scope_info.tags.first_tag".to_string(),
                "first_value".into()
            ),
            (
                "scope_info.tags.second_tag".to_string(),
                "second_value".into()
            ),
            ("estimated_usage.tokens".to_string(), 1009.into()),
            ("estimated_usage.model_inferences".to_string(), 1.into()),
            ("level".to_string(), "INFO".into()),
        ])
    );

    assert_eq!(num_spans, 5);
}

#[test]
pub fn test_capture_rate_limit_error() {
    let episode_uuid = Uuid::now_v7();

    let config = r#"
    [rate_limiting]
    enabled = true

    [[rate_limiting.rules]]
    priority = 0
    model_inferences_per_minute = 1000
    tokens_per_minute = 2
    scope = [
    {tag_key = "user_id", tag_value = "tensorzero::each" }
    ]

    [gateway.export.otlp.traces]
    enabled = true
    "#
    .to_string();

    let user_id = Uuid::now_v7().to_string();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (exporter, _res) = runtime.block_on(async {
        let exporter = install_capturing_otel_exporter().await;
        let _guard = enter_fake_http_request_otel();
        let client =
            tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
        let err = client
            .inference(ClientInferenceParams {
                episode_id: Some(episode_uuid),
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
                params: InferenceParams {
                    chat_completion: ChatCompletionInferenceParams {
                        max_tokens: Some(1000),
                        ..Default::default()
                    },
                },
                tags: HashMap::from([
                    ("first_tag".to_string(), "first_value".into()),
                    ("second_tag".to_string(), "second_value".into()),
                    ("user_id".to_string(), user_id.clone()),
                ]),
                ..Default::default()
            })
            .await
            .unwrap_err();
        (exporter, err)
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
    assert_eq!(root_attr_map["model_name"], "dummy::good".into());
    assert_eq!(root_attr_map["episode_id"], episode_uuid.to_string().into());
    assert_eq!(root_attr_map.get("function_name"), None);
    assert_eq!(root_attr_map.get("variant_name"), None);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let [variant_span] = root_children.as_slice() else {
        panic!("Expected one child span: {root_children:#?}");
    };

    assert_eq!(variant_span.name, "variant_inference");
    assert_eq!(variant_span.status, Status::Ok);
    let variant_attr_map = attrs_to_map(&variant_span.attributes);
    assert_eq!(
        variant_attr_map["function_name"],
        "tensorzero::default".into()
    );
    assert_eq!(variant_attr_map["variant_name"], "dummy::good".into());
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
    assert_eq!(model_attr_map["model_name"], "dummy::good".into());
    assert_eq!(model_attr_map["stream"], false.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
    assert_eq!(model_provider_span.status, Status::Ok);
    assert_eq!(
        model_provider_span.events.len(),
        0,
        "Unexpected number of events: {model_provider_span:#?}",
    );
    let model_provider_attr_map = attrs_to_map(&model_provider_span.attributes);
    assert_eq!(model_provider_attr_map["provider_name"], "dummy".into());

    assert_eq!(
        model_provider_attr_map["gen_ai.operation.name"],
        "chat".into()
    );
    assert_eq!(model_provider_attr_map["gen_ai.system"], "dummy".into());
    assert_eq!(
        model_provider_attr_map["gen_ai.request.model"],
        "good".into()
    );

    assert_eq!(model_attr_map["stream"], false.into());
    let rate_limit_spans = spans
        .span_children
        .get(&model_provider_span.span_context.span_id())
        .unwrap();
    // We failed to consume tickets, so we shouldn't have a 'rate_limiting_return_tickets' span
    let [consume_ticket_span] = rate_limit_spans.as_slice() else {
        panic!("Expected one rate limit span: {rate_limit_spans:#?}");
    };
    assert_eq!(consume_ticket_span.name, "rate_limiting_consume_tickets");
    assert_eq!(
        consume_ticket_span.status,
        Status::Error {
            description: format!(
                r#"TensorZero rate limit exceeded for `token` resource.
Scope: tag_key="user_id", tag_value="tensorzero::each" (matched: "{user_id}")
Requested: 1009
Available: 2"#
            )
            .into()
        }
    );
    let mut consume_ticket_attr_map = attrs_to_map(&consume_ticket_span.attributes);
    remove_unstable_attrs(&mut consume_ticket_attr_map);

    assert_eq!(
        consume_ticket_attr_map,
        HashMap::from([
            (
                "scope_info.tags.first_tag".to_string(),
                "first_value".into()
            ),
            (
                "scope_info.tags.second_tag".to_string(),
                "second_value".into()
            ),
            ("estimated_usage.tokens".to_string(), 1009.into()),
            ("estimated_usage.model_inferences".to_string(), 1.into()),
            ("level".to_string(), "INFO".into()),
        ])
    );

    assert_eq!(num_spans, 5);
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_suppress_otel_spans() {
    let exporter = install_capturing_otel_exporter().await;

    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    // We do *not* call `enter_fake_http_request_otel` before calling `inference`.
    // As a result, otel reporting should get suppressed entirely.
    // This is the behavior of an embedded client if we somehow turned on real otel exporting. (which we don't support at the moment)
    // The main purpose of this test is to verify that otel span suppression works correctly - in a real gateway,
    // we want to ensure that non-instrumented routes (e.g. ui routes) cannot cause otel spans to be reported,
    // even if they call into instrumented code.
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
    assert!(all_spans.is_empty(), "Should have suppressed all spans");
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_capture_feedback_spans() {
    let exporter = install_capturing_otel_exporter().await;
    let _guard = enter_fake_http_request_otel();

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
