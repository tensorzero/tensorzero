#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
use std::{net::SocketAddr, process::Stdio};

use reqwest::Response;
use tempfile::NamedTempFile;
use tokio::{
    io::{AsyncBufReadExt, BufReader, Lines},
    process::{Child, ChildStdout, Command},
};

pub fn gateway_path() -> String {
    // Compatibility with 'cargo nextest archive': https://nexte.st/docs/ci-features/archiving/#making-tests-relocatable
    std::env::var("NEXTEST_BIN_EXE_gateway")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_gateway").to_string())
}

pub async fn start_gateway_on_random_port(
    config_suffix: &str,
    rust_log: Option<&str>,
) -> ChildData {
    let config_str = format!(
        r#"
        [gateway]
        bind_address = "0.0.0.0:0"
        {config_suffix}
    "#
    );

    let tmpfile = NamedTempFile::new().unwrap();
    std::fs::write(tmpfile.path(), config_str).unwrap();

    let mut builder = Command::new(gateway_path());
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
        if line.contains("{\"message\":\"└") {
            // We're done logging the startup message
            break;
        }
        if line.contains("listening on 0.0.0.0:") {
            listening_line = Some(line);
        }
    }

    let port = listening_line
        .expect("Gateway exited before listening")
        .split_once("listening on 0.0.0.0:")
        .expect("Gateway didn't log listening line")
        .1
        .split("\"")
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

#[expect(dead_code)] // Not all e2e tests use all fields
pub struct ChildData {
    pub addr: SocketAddr,
    pub output: Vec<String>,
    pub stdout: Lines<BufReader<ChildStdout>>,
    // This kills the child on drop
    pub child: Child,
}

impl ChildData {
    #[allow(dead_code, clippy::allow_attributes)]
    pub async fn call_health_endpoint(&self) -> Response {
        reqwest::Client::new()
            .get(format!("http://{}/health", self.addr))
            .send()
            .await
            .unwrap()
    }
}
