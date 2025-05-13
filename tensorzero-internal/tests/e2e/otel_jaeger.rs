#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use chrono::Duration;
use chrono::SecondsFormat;
use chrono::Utc;
use http::StatusCode;
use opentelemetry::SpanId;
use opentelemetry::TraceId;
use opentelemetry_sdk::trace::IdGenerator;
use opentelemetry_sdk::trace::RandomIdGenerator;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[derive(Debug, Copy, Clone)]
struct ExistingTraceData {
    trace_id: TraceId,
    span_id: SpanId,
}

#[tokio::test]
async fn test_jaeger_trace_export_no_parent() {
    test_jaeger_trace_export(None).await;
}

#[tokio::test]
async fn test_jaeger_trace_export_with_parent() {
    let id_gen = RandomIdGenerator::default();
    let trace_id = id_gen.new_trace_id();
    let span_id = id_gen.new_span_id();
    test_jaeger_trace_export(Some(ExistingTraceData { trace_id, span_id })).await;
}

// TODO - investigate why this test is sometimes flaky when running locally
async fn test_jaeger_trace_export(existing_trace_parent: Option<ExistingTraceData>) {
    let episode_id = Uuid::now_v7();
    let client = reqwest::Client::new();

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

    let mut builder = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);

    let existing_trace_header = existing_trace_parent
        // Version 00, with the 'sampled' flag set to 1
        .map(|trace_parent| format!("00-{}-{}-01", trace_parent.trace_id, trace_parent.span_id));

    if let Some(existing_trace_header) = &existing_trace_header {
        builder = builder.header("traceparent", existing_trace_header);
    }

    let response = builder.send().await.unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    let inference_id = response_json["inference_id"]
        .as_str()
        .expect("inference_id should be a string");

    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let now = Utc::now();
    let one_minute_ago = now - Duration::minutes(1);

    let now_str = now.to_rfc3339_opts(SecondsFormat::Secs, true);
    let one_minute_ago_str = one_minute_ago.to_rfc3339_opts(SecondsFormat::Secs, true);

    let jaeger_result = client.get(format!("http://localhost:16686/api/v3/traces?&query.start_time_min={one_minute_ago_str}&query.start_time_max={now_str}&query.service_name=tensorzero-gateway")).send().await.unwrap();
    let jaeger_traces = jaeger_result.json::<Value>().await.unwrap();
    println!("Response: {jaeger_traces}");
    let mut target_span = None;

    let mut span_by_id = HashMap::new();

    for resource_span in jaeger_traces["result"]["resourceSpans"].as_array().unwrap() {
        for scope_span in resource_span["scopeSpans"].as_array().unwrap() {
            for span in scope_span["spans"].as_array().unwrap() {
                span_by_id.insert(span["spanId"].as_str().unwrap(), span.clone());
                if span["name"] == "function_inference" {
                    println!("Found function_inference span: {span:?}");
                    let attrs = span["attributes"].as_array().unwrap();
                    for attr in attrs {
                        if attr["key"].as_str().unwrap() == "inference_id" {
                            let inference_id_jaeger =
                                attr["value"]["stringValue"].as_str().unwrap();
                            if inference_id == inference_id_jaeger {
                                if target_span.is_some() {
                                    panic!("Found multiple function_inference spans with inference id: {inference_id}");
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

    // Just check a couple of spans - we already have more comprehensive tests that check the exact spans
    // send to the global `opentelemetry` exporter.
    let function_inference_span = target_span.unwrap_or_else(|| {
        panic!("No function_inference span found with matching inference_id: {inference_id}")
    });
    assert_eq!(function_inference_span["name"], "function_inference");
    assert_eq!(function_inference_span["kind"], 1);
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

    if let Some(existing_trace_parent) = existing_trace_parent {
        assert_eq!(
            parent_span["traceId"],
            existing_trace_parent.trace_id.to_string()
        );
        assert_eq!(
            parent_span["parentSpanId"],
            existing_trace_parent.span_id.to_string()
        );
    }
}
