use std::collections::HashMap;
use std::sync::Arc;

use base64::prelude::*;
use chrono::DateTime;
use chrono::Utc;
use http::StatusCode;
use opentelemetry::KeyValue;
use opentelemetry::SpanId;
use opentelemetry::TraceId;
use opentelemetry_sdk::trace::IdGenerator;
use opentelemetry_sdk::trace::RandomIdGenerator;
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use url::Url;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[derive(Debug, Copy, Clone)]
struct ExistingTraceData {
    trace_id: TraceId,
    span_id: SpanId,
}

#[tokio::test]
async fn test_otel_export_trace_export_no_parent() {
    test_otel_export_trace_export(None, None, None, None, Arc::new(Semaphore::new(1))).await;
}

#[tokio::test]
async fn test_otel_export_trace_export_with_parent() {
    let id_gen = RandomIdGenerator::default();
    let trace_id = id_gen.new_trace_id();
    let span_id = id_gen.new_span_id();
    test_otel_export_trace_export(
        Some(ExistingTraceData { trace_id, span_id }),
        None,
        None,
        None,
        Arc::new(Semaphore::new(1)),
    )
    .await;
}

#[tokio::test]
async fn test_otel_export_trace_export_with_custom_header() {
    // Prevent overloading Tempo (it gives us 'job queue full' errors if we send too many requests at once)
    let semaphore = Arc::new(Semaphore::new(10));
    let mut futures = JoinSet::new();
    let num_tasks = 100;
    for _ in 0..num_tasks {
        futures.spawn(test_otel_export_trace_export(
            None,
            Some((
                "TensorZero-OTLP-Traces-Extra-Header-x-dummy-tensorzero".to_string(),
                Uuid::now_v7().to_string(),
            )),
            Some(KeyValue::new(
                "my-custom-resource",
                Uuid::now_v7().to_string(),
            )),
            Some(KeyValue::new(
                "my-custom-attribute",
                format!("My attr value: {}", Uuid::now_v7()),
            )),
            semaphore.clone(),
        ));
    }
    let mut i = 1;
    while let Some(task) = futures.join_next().await {
        let () = task.unwrap();
        println!("Completed task {i}/{num_tasks}");
        i += 1;
    }
}

#[tokio::test]
async fn test_otel_reject_invalid_attribute_values() {
    let client = reqwest::Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "episode_id": episode_id,
        "model_name": "openai::missing-model-name",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .header(
            "tensorzero-otlp-traces-extra-attribute-my-custom-attribute",
            "non-quoted-string",
        )
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let res = response.text().await.unwrap();
    assert!(res.contains("Failed to parse `tensorzero-otlp-traces-extra-attribute-` header `my-custom-attribute` value as valid JSON: expected ident at line 1 column 2"), "Unexpected error message: {res}");
}

#[tokio::test]
async fn test_otel_export_http_error() {
    let client = reqwest::Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "episode_id": episode_id,
        "model_name": "openai::missing-model-name",
        "input":
            {
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let start_time = Utc::now();

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let _response_json = response.json::<Value>().await.unwrap();

    let TempoSpans {
        target_span: function_inference_span,
        span_by_id,
        resources: _,
    } = get_tempo_spans(
        ("episode_id", &episode_id.to_string()),
        start_time,
        &Semaphore::new(1),
    )
    .await;

    let function_inference_span =
        function_inference_span.expect("No function_inference span found");

    let parent_id = function_inference_span["parentSpanId"].as_str().unwrap();
    let parent_span = span_by_id.get(parent_id).unwrap();

    let parent_attrs: HashMap<&str, serde_json::Value> = parent_span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
        .collect();

    println!("Parent attrs: {parent_attrs:?}");

    assert_eq!(parent_span["name"], "POST /inference");
    assert_eq!(parent_attrs["level"]["stringValue"], "INFO");
    assert_eq!(parent_attrs["http.response.status_code"]["intValue"], "502");
    assert_eq!(parent_span["status"]["code"], "STATUS_CODE_ERROR");
    assert!(
        parent_span["status"]["message"]
            .as_str()
            .unwrap()
            .starts_with("All variants failed with errors: openai::missing-model-name"),
        "Unexpected span error message: {}",
        parent_span["status"]["message"].as_str().unwrap()
    );
}

pub struct TempoSpans {
    pub target_span: Option<Value>,
    pub span_by_id: HashMap<String, Value>,
    pub resources: Vec<Value>,
}

pub async fn get_tempo_spans(
    (tag_key, tag_value): (&str, &str),
    start_time: DateTime<Utc>,
    tempo_semaphore: &Semaphore,
) -> TempoSpans {
    // It takes some time for the span to show up in Tempo
    tokio::time::sleep(std::time::Duration::from_secs(25)).await;

    let start_time = start_time.timestamp();
    let now = Utc::now().timestamp();

    let client = reqwest::Client::new();

    let tempo_base_url = std::env::var("TENSORZERO_TEMPO_URL")
        .unwrap_or_else(|_| "http://localhost:3200".to_string());

    let get_url = Url::parse(&format!(
        "{tempo_base_url}/api/search?tags={tag_key}={tag_value}&start={start_time}&end={now}"
    ))
    .unwrap();
    println!("Requesting URL: {get_url}");

    let permit = tempo_semaphore.acquire().await.unwrap();

    let jaeger_result = client.get(get_url).send().await.unwrap();
    let res = jaeger_result.text().await.unwrap();
    println!("Tempo result: {res}");
    let tempo_traces = serde_json::from_str::<Value>(&res).unwrap();

    if tempo_traces["traces"].as_array().unwrap().is_empty() {
        return TempoSpans {
            target_span: None,
            span_by_id: HashMap::new(),
            resources: Vec::new(),
        };
    }
    let trace_id = tempo_traces["traces"][0]["traceID"].as_str().unwrap();

    let trace_res = client
        .get(format!("{tempo_base_url}/api/traces/{trace_id}"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    drop(permit);

    println!("Trace res: {trace_res}");
    let trace_data = serde_json::from_str::<Value>(&trace_res).unwrap();

    let mut span_by_id = HashMap::new();
    let mut target_span = None;
    let mut resources = Vec::new();

    for batch in trace_data["batches"].as_array().unwrap() {
        resources.push(batch["resource"].clone());
        for scope_span in batch["scopeSpans"].as_array().unwrap() {
            for span in scope_span["spans"].as_array().unwrap() {
                span_by_id.insert(span["spanId"].as_str().unwrap().to_string(), span.clone());
                if span["name"] == "function_inference" {
                    //println!("Found function_inference span: {span:?}");
                    let attrs = span["attributes"].as_array().unwrap();
                    for attr in attrs {
                        if attr["key"].as_str().unwrap() == "function_name" {
                            assert!(
                                attr["value"].get("intValue").is_none(),
                                "Bad span: {span:?}"
                            );
                        }
                        if attr["key"].as_str().unwrap() == tag_key {
                            let inference_id_jaeger =
                                attr["value"]["stringValue"].as_str().unwrap();
                            if tag_value == inference_id_jaeger {
                                if target_span.is_some() {
                                    panic!("Found multiple function_inference spans with `{tag_key}`: {tag_value}");
                                } else {
                                    target_span = Some(span.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    TempoSpans {
        target_span,
        span_by_id,
        resources,
    }
}

#[tokio::test]
async fn test_otel_health_not_exported() {
    let client = reqwest::Client::new();

    let start_time = Utc::now();
    let tempo_semaphore = Arc::new(Semaphore::new(1));

    // Check that the /health endpoint is not exported to OTEL
    let response = client
        .get(get_gateway_endpoint("/health"))
        .header(
            "tensorzero-otlp-traces-extra-attribute-my-health-attr",
            "\"my-attr-value\"",
        )
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let spans = get_tempo_spans(
        ("my-health-attr", "my-attr-value"),
        start_time,
        &tempo_semaphore,
    )
    .await;

    assert_eq!(spans.target_span, None, "Target span should be none");
    assert_eq!(
        spans.span_by_id,
        HashMap::new(),
        "Span by ID should be empty"
    );
    assert_eq!(
        spans.resources,
        Vec::<Value>::new(),
        "Resources should be empty"
    );
}

#[tokio::test]
async fn test_otel_export_custom_attribute_override() {
    let client = reqwest::Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let start_time = Utc::now();
    let tempo_semaphore = Arc::new(Semaphore::new(1));

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .header(
            "tensorzero-otlp-traces-extra-attribute-function_name",
            "\"my-overridden-function-name\"",
        )
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let _response_json = response.json::<Value>().await.unwrap();
    let TempoSpans {
        target_span: function_inference_span,
        span_by_id: _,
        resources: _,
    } = get_tempo_spans(
        ("episode_id", &episode_id.to_string()),
        start_time,
        &tempo_semaphore,
    )
    .await;

    let function_inference_span =
        function_inference_span.expect("No function_inference span found");

    assert_eq!(function_inference_span["name"], "function_inference");
    assert_eq!(function_inference_span["kind"], "SPAN_KIND_INTERNAL");
    let attrs: HashMap<&str, serde_json::Value> = function_inference_span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
        .collect();

    println!("Attrs: {attrs:?}");

    assert_eq!(
        attrs["function_name"]["stringValue"],
        "my-overridden-function-name"
    );
}

// TODO - investigate why this test is sometimes flaky when running locally
async fn test_otel_export_trace_export(
    existing_trace_parent: Option<ExistingTraceData>,
    custom_header: Option<(String, String)>,
    custom_resource: Option<KeyValue>,
    custom_attribute: Option<KeyValue>,
    tempo_semaphore: Arc<Semaphore>,
) {
    let client = reqwest::Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"foo": "bar"},
    });

    let start_time = Utc::now();

    let mut builder = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);

    if let Some((custom_key, custom_value)) = &custom_header {
        builder = builder.header(custom_key, custom_value);
    }
    if let Some(custom_resource) = &custom_resource {
        builder = builder.header(
            format!(
                "tensorzero-otlp-traces-extra-resource-{}",
                custom_resource.key
            ),
            custom_resource.value.to_string(),
        );
    }

    if let Some(custom_attribute) = &custom_attribute {
        builder = builder.header(
            format!(
                "tensorzero-otlp-traces-extra-attribute-{}",
                custom_attribute.key
            ),
            serde_json::to_string(&serde_json::Value::String(
                custom_attribute.value.to_string(),
            ))
            .unwrap(),
        );
    }

    let existing_trace_header = existing_trace_parent
        // Version 00, with the 'sampled' flag set to 1
        .map(|trace_parent| format!("00-{}-{}-01", trace_parent.trace_id, trace_parent.span_id));

    if let Some(existing_trace_header) = &existing_trace_header {
        builder = builder.header("traceparent", existing_trace_header);
    }

    let response = builder.send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let _response_json = response.json::<Value>().await.unwrap();
    let TempoSpans {
        target_span: function_inference_span,
        span_by_id,
        resources,
    } = get_tempo_spans(
        ("episode_id", &episode_id.to_string()),
        start_time,
        &tempo_semaphore,
    )
    .await;

    let function_inference_span =
        function_inference_span.expect("No function_inference span found");

    if let Some(custom_attribute) = &custom_attribute {
        for span in span_by_id.values() {
            let attrs: HashMap<&str, serde_json::Value> = span["attributes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
                .collect();

            assert_eq!(
                attrs[custom_attribute.key.as_str()]["stringValue"]
                    .as_str()
                    .unwrap(),
                custom_attribute.value.to_string()
            );
        }
    }

    // Each tempo 'batch' has its own resources object
    for batch_resources in resources {
        let attrs: HashMap<&str, serde_json::Value> = batch_resources["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
            .collect();
        assert_eq!(attrs["service.name"]["stringValue"], "tensorzero-gateway");
        if let Some(custom_resource) = &custom_resource {
            assert_eq!(
                attrs[custom_resource.key.as_str()]["stringValue"]
                    .as_str()
                    .unwrap(),
                custom_resource.value.to_string()
            );
        }
    }

    // Just check a couple of spans - we already have more comprehensive tests that check the exact spans
    // send to the global `opentelemetry` exporter.
    assert_eq!(function_inference_span["name"], "function_inference");
    assert_eq!(function_inference_span["kind"], "SPAN_KIND_INTERNAL");
    let attrs: HashMap<&str, serde_json::Value> = function_inference_span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
        .collect();

    println!("Attrs: {attrs:?}");

    assert_eq!(attrs["function_name"]["stringValue"], "basic_test");

    let parent_id = function_inference_span["parentSpanId"].as_str().unwrap();
    let parent_span = span_by_id.get(parent_id).unwrap();

    assert_eq!(parent_span["name"], "POST /inference");
    assert_eq!(attrs["level"]["stringValue"], "INFO");

    let parent_attrs: HashMap<&str, serde_json::Value> = parent_span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap(), a["value"].clone()))
        .collect();

    if let Some((_, custom_value)) = &custom_header {
        assert_eq!(
            &parent_attrs["tensorzero.custom_key"]["stringValue"]
                .as_str()
                .expect("custom_key should be a string"),
            custom_value,
            "Bad parent attrs: {parent_attrs:?}"
        );
    }
    println!("Parent attrs: {parent_attrs:?}");
    assert_eq!(parent_attrs["http.response.status_code"]["intValue"], "200");

    if let Some(existing_trace_parent) = existing_trace_parent {
        // Tempo returns a base64-encoded binary trace ID, so we need to decode it
        // and convert it to a hex string
        let trace_id_decoded = hex::encode(
            BASE64_STANDARD
                .decode(parent_span["traceId"].as_str().unwrap())
                .unwrap(),
        );
        assert_eq!(trace_id_decoded, existing_trace_parent.trace_id.to_string());
        let parent_span_id_decoded = hex::encode(
            BASE64_STANDARD
                .decode(parent_span["parentSpanId"].as_str().unwrap())
                .unwrap(),
        );
        assert_eq!(
            parent_span_id_decoded,
            existing_trace_parent.span_id.to_string()
        );
    }
}
