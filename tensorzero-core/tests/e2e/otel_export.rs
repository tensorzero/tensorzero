#![allow(clippy::print_stdout)]
use std::collections::HashMap;
use std::sync::Arc;

use base64::prelude::*;
use chrono::Utc;
use http::StatusCode;
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
    test_otel_export_trace_export(None, None, Arc::new(Semaphore::new(1))).await;
}

#[tokio::test]
async fn test_otel_export_trace_export_with_parent() {
    let id_gen = RandomIdGenerator::default();
    let trace_id = id_gen.new_trace_id();
    let span_id = id_gen.new_span_id();
    test_otel_export_trace_export(
        Some(ExistingTraceData { trace_id, span_id }),
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
        let random_id = Uuid::now_v7();
        futures.spawn(test_otel_export_trace_export(
            None,
            Some((
                "TensorZero-OTLP-Traces-Extra-Header-x-dummy-tensorzero".to_string(),
                random_id.to_string(),
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

// TODO - investigate why this test is sometimes flaky when running locally
async fn test_otel_export_trace_export(
    existing_trace_parent: Option<ExistingTraceData>,
    custom_header: Option<(String, String)>,
    tempo_semaphore: Arc<Semaphore>,
) {
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

    let start_time = Utc::now().timestamp();

    let mut builder = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);

    if let Some((custom_key, custom_value)) = &custom_header {
        builder = builder.header(custom_key, custom_value);
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
    let response_json = response.json::<Value>().await.unwrap();

    let inference_id = response_json["inference_id"]
        .as_str()
        .expect("inference_id should be a string");

    // It takes some time for the span to show up in Tempo
    tokio::time::sleep(std::time::Duration::from_secs(25)).await;

    let now = Utc::now().timestamp();

    let jaeger_base_url = std::env::var("TENSORZERO_TEMPO_URL")
        .unwrap_or_else(|_| "http://localhost:3200".to_string());

    let get_url = Url::parse(&format!(
        "{jaeger_base_url}/api/search?tags=inference_id={inference_id}&start={start_time}&end={now}"
    ))
    .unwrap();
    println!("Requesting URL: {get_url}");

    let permit = tempo_semaphore.acquire().await.unwrap();

    let jaeger_result = client.get(get_url).send().await.unwrap();
    let res = jaeger_result.text().await.unwrap();
    println!("Tempo result: {res}");
    let tempo_traces = serde_json::from_str::<Value>(&res).unwrap();
    let trace_id = tempo_traces["traces"][0]["traceID"].as_str().unwrap();

    let trace_res = client
        .get(format!("{jaeger_base_url}/api/traces/{trace_id}"))
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

    for batch in trace_data["batches"].as_array().unwrap() {
        for scope_span in batch["scopeSpans"].as_array().unwrap() {
            for span in scope_span["spans"].as_array().unwrap() {
                span_by_id.insert(span["spanId"].as_str().unwrap(), span.clone());
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
