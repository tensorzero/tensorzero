#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

mod common;

use common::start_gateway_on_random_port;
use flate2::{Compression, write::GzEncoder};
use reqwest::header::{CONTENT_ENCODING, CONTENT_TYPE};
use serde_json::json;
use std::io::Write;
use uuid::Uuid;

fn gzip_compress(input: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

fn zstd_compress(input: &[u8]) -> Vec<u8> {
    // Level 0 == zstd default.
    zstd::stream::encode_all(input, 0).unwrap()
}

fn brotli_compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        // Use a small buffer size (4096) and reasonably strong compression params.
        // These values don't need to match any production client; we just need a valid `br` payload.
        let mut writer = brotli::CompressorWriter::new(&mut out, 4096, 5, 22);
        writer.write_all(input).unwrap();
    }
    out
}

async fn run_compression_test(
    compress: fn(&[u8]) -> Vec<u8>,
    encoding: &str,
    message: &str,
) {
    let gateway = start_gateway_on_random_port("", None).await;

    let body = json!({
        "model_name": "dummy::good",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        "stream": false
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let compressed = compress(&body_bytes);

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{}/inference", gateway.addr))
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_ENCODING, encoding)
        .body(compressed)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let text = response.text().await.unwrap();
    assert_eq!(
        status,
        http::StatusCode::OK,
        "Expected {encoding}-compressed request body to be accepted and decoded. Status: {status}, body: {text}"
    );

    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
        json.get("inference_id").is_some(),
        "Expected successful inference response to include inference_id. Body: {json}"
    );
}

#[tokio::test]
async fn test_request_decompression_gzip_inference() {
    run_compression_test(gzip_compress, "gzip", "Hello (gzip)").await;
}

#[tokio::test]
async fn test_request_decompression_zstd_inference() {
    run_compression_test(zstd_compress, "zstd", "Hello (zstd)").await;
}

#[tokio::test]
async fn test_request_decompression_brotli_inference() {
    run_compression_test(brotli_compress, "br", "Hello (br)").await;
}
