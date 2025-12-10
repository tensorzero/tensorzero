#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::allow_attributes
)]
use std::{net::SocketAddr, process::Stdio};

use reqwest::Response;
use tempfile::NamedTempFile;
use tokio::{
    io::AsyncBufReadExt,
    process::{Child, Command},
    sync::mpsc::UnboundedReceiver,
};

/// `#[sqlx::test]` doesn't work here because it needs to share the DB with `start_gateway_on_random_port`.
#[allow(dead_code)]
pub async fn get_postgres_pool_for_testing() -> sqlx::PgPool {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for auth tests");

    sqlx::PgPool::connect(&postgres_url)
        .await
        .expect("Failed to connect to PostgreSQL")
}

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

    let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel();
    let line_tx = Some(line_tx);
    #[allow(clippy::disallowed_methods)]
    tokio::spawn(async move {
        while let Some(line) = stdout.next_line().await.unwrap() {
            println!("{line}");
            if let Some(line_tx) = &line_tx {
                let _ = line_tx.send(line.clone());
            }
        }
    });

    let mut listening_line = None;
    let mut output = Vec::new();
    while let Some(line) = line_rx.recv().await {
        if line.contains("listening on 0.0.0.0:") {
            listening_line = Some(line.clone());
        }
        output.push(line.clone());
        if line.contains("{\"message\":\"└") {
            // We're done logging the startup message
            break;
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
        stdout: line_rx,
        child,
    }
}

#[expect(dead_code)] // Not all e2e tests use all fields
pub struct ChildData {
    pub addr: SocketAddr,
    pub output: Vec<String>,
    pub stdout: UnboundedReceiver<String>,
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
