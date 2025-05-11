#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

use std::{net::SocketAddr, process::Stdio, time::Duration};

use tempfile::NamedTempFile;
use tokio::{
    io::{AsyncBufReadExt, BufReader, Lines},
    process::{Child, ChildStdout, Command},
};

const GATEWAY_PATH: &str = env!("CARGO_BIN_EXE_gateway");

/// Test the gateway does not log '/health' requests when RUST_LOG and [gateway.debug] are not set
#[tokio::test]
async fn test_logging_no_rust_log_default_debug() {
    let mut child_data = start_gateway_on_random_port("", None).await;
    let health_response = child_data.call_health_endpoint().await;
    assert_eq!(health_response, r#"{"gateway":"ok","clickhouse":"ok"}"#);
    let _err: tokio::time::error::Elapsed =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect_err("Gateway wrote to stdout after /health endpoint in non-debug mode");
}

/// Test that the gateway logs '/health' requests when [gateway.debug] is set
#[tokio::test]
async fn test_logging_no_rust_log_debug_on() {
    let mut child_data = start_gateway_on_random_port(
        r#"
    debug = true
    "#,
        None,
    )
    .await;
    let health_response = child_data.call_health_endpoint().await;
    assert_eq!(health_response, r#"{"gateway":"ok","clickhouse":"ok"}"#);

    let gateway_log_line =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect("Gateway didn't write to stdout after /health endpoint in debug mode")
            .expect("Error reading gateway log line")
            .expect("Gateway stdout was closed");
    println!("gateway log line: {gateway_log_line}");
    assert!(
        gateway_log_line.contains("/health"),
        "Unexpected log line: {gateway_log_line}",
    );
    assert!(
        gateway_log_line.contains("tower_http::trace"),
        "Missing tower_http::trace in log line: {gateway_log_line}",
    );
}

/// Test that the gateway does logs '/health' requests when `RUST_LOG` blocks it
#[tokio::test]
async fn test_logging_rust_log_debug_on() {
    let mut child_data = start_gateway_on_random_port(
        r#"
    debug = true
    "#,
        Some("gateway=debug"),
    )
    .await;
    let health_response = child_data.call_health_endpoint().await;
    assert_eq!(health_response, r#"{"gateway":"ok","clickhouse":"ok"}"#);

    let _err: tokio::time::error::Elapsed =
        tokio::time::timeout(Duration::from_secs(1), child_data.stdout.next_line())
            .await
            .expect_err("Gateway wrote to stdout after /health endpoint in non-debug mode");
}

struct ChildData {
    addr: SocketAddr,
    #[expect(dead_code)]
    output: Vec<String>,
    stdout: Lines<BufReader<ChildStdout>>,
    #[expect(dead_code)] // This kills the child on drop
    child: Child,
}

impl ChildData {
    async fn call_health_endpoint(&self) -> String {
        reqwest::Client::new()
            .get(format!("http://{}/health", self.addr))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }
}

async fn start_gateway_on_random_port(config_suffix: &str, rust_log: Option<&str>) -> ChildData {
    let config_str = format!(
        r#"
        [gateway]
        observability.enabled = false
        bind_address = "0.0.0.0:0"
        {config_suffix}
    "#
    );

    let tmpfile = NamedTempFile::new().unwrap();
    std::fs::write(tmpfile.path(), config_str).unwrap();

    let mut builder = Command::new(GATEWAY_PATH);
    builder
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .args([
            "--config-file",
            tmpfile.path().to_str().unwrap(),
            "--log-format",
            "json",
        ])
        // Make sure we don't inherit `RUST_LOG` from the outer `cargo test/nextest` invocation
        .env_remove("RUST_LOG")
        .kill_on_drop(true);

    if let Some(rust_log) = rust_log {
        builder.env("RUST_LOG", rust_log);
    }

    let mut child = builder.spawn().unwrap();
    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();

    let mut output = Vec::new();
    let mut listening_line = None;
    while let Some(line) = stdout.next_line().await.unwrap() {
        println!("gateway output line: {line}");
        output.push(line.clone());
        if line.contains("listening on 0.0.0.0:") {
            listening_line = Some(line);
            break;
        }
    }

    let port = listening_line
        .expect("Gateway exited before listening")
        .split_once("listening on 0.0.0.0:")
        .expect("Gateway didn't log listening line")
        .1
        .split(" ")
        .next()
        .unwrap()
        .parse::<u16>()
        .unwrap();

    ChildData {
        addr: format!("0.0.0.0:{port}").parse::<SocketAddr>().unwrap(),
        output,
        stdout,
        child,
    }
}
