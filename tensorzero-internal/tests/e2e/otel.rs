use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use opentelemetry::{KeyValue, SpanId, Value};
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, Role,
};
use tensorzero_internal::inference::types::TextKind;
use tensorzero_internal::observability::build_opentelemetry_layer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::providers::common::make_embedded_gateway_no_config;

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
    pub fn take_spans(self) -> Vec<SpanData> {
        let spans = self
            .spans
            .lock()
            .expect("Failed to lock spans mutex")
            .replace(Vec::new())
            .expect("CapturingExporter is already shut down");
        spans
    }
}

pub struct SpanMap {
    pub root_spans: Vec<SpanData>,
    pub span_children: HashMap<SpanId, Vec<SpanData>>,
}

pub fn install_capturing_otel_exporter() -> CapturingOtelExporter {
    let exporter = CapturingOtelExporter {
        spans: Arc::new(Mutex::new(Some(vec![]))),
    };
    let (enable_otel, layer) = build_opentelemetry_layer(Some(exporter.clone()))
        .expect("Failed to build OpenTelemetry layer");

    tracing_subscriber::registry().with(layer).init();
    enable_otel
        .enable_otel()
        .expect("Failed to enable OpenTelemetry");
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

#[tokio::test]
pub async fn test_capture_simple_inference_spans() {
    let exporter = install_capturing_otel_exporter();

    let client = make_embedded_gateway_no_config().await;
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
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::NonStreaming(output) = res else {
        panic!("Expected non-streaming output, got: {res:#?}");
    };

    let all_spans = exporter.take_spans();
    let num_spans = all_spans.len();
    let spans = build_span_map(all_spans);
    let [root_span] = spans.root_spans.as_slice() else {
        panic!("Expected one root span: {:#?}", spans.root_spans);
    };
    // Since we're using the embedded gateway, the root span will be `function_inference`
    // (we won't have a top-level HTTP span)
    assert_eq!(root_span.name, "function_inference");
    let root_attr_map = attrs_to_map(&root_span.attributes);
    assert_eq!(root_attr_map["model_name"], "dummy::good".into());
    assert_eq!(
        root_attr_map["inference_id"],
        output.inference_id().to_string().into()
    );
    assert_eq!(
        root_attr_map["episode_id"],
        output.episode_id().to_string().into()
    );
    assert_eq!(root_attr_map.get("function_name"), None);
    assert_eq!(root_attr_map.get("variant_name"), None);

    let root_children = &spans.span_children[&root_span.span_context.span_id()];
    let [variant_span] = root_children.as_slice() else {
        panic!("Expected one child span: {root_children:#?}");
    };

    assert_eq!(variant_span.name, "variant_inference");
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
    let model_attr_map = attrs_to_map(&model_span.attributes);
    assert_eq!(model_attr_map["model_name"], "dummy::good".into());
    assert_eq!(model_attr_map["stream"], false.into());

    let model_children = &spans.span_children[&model_span.span_context.span_id()];
    let [model_provider_span] = model_children.as_slice() else {
        panic!("Expected one child span: {model_children:#?}");
    };
    assert_eq!(model_provider_span.name, "model_provider_inference");
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

    assert_eq!(
        spans
            .span_children
            .get(&model_provider_span.span_context.span_id()),
        None
    );

    assert_eq!(num_spans, 4);
}
