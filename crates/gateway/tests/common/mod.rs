#![expect(
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

pub mod relay;

/// `#[sqlx::test]` doesn't work here because it needs to share the DB with `start_gateway_on_random_port`.
#[allow(dead_code)]
pub async fn get_postgres_pool_for_testing() -> sqlx::PgPool {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for auth tests");

    sqlx::PgPool::connect(&postgres_url)
        .await
        .expect("Failed to connect to Postgres")
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
    start_gateway_impl(
        Some("0.0.0.0:0"), // config bind_address
        None,              // cli bind_address
        config_suffix,
        rust_log,
    )
    .await
}

/// Start gateway with a CLI --bind-address flag.
/// If `config_bind_address` is Some, it will be included in the config file.
/// The `cli_bind_address` is passed via --bind-address CLI arg.
#[allow(dead_code)]
pub async fn start_gateway_with_cli_bind_address(
    config_bind_address: Option<&str>,
    cli_bind_address: &str,
    config_suffix: &str,
) -> ChildData {
    start_gateway_impl(
        config_bind_address,
        Some(cli_bind_address),
        config_suffix,
        None,
    )
    .await
}

/// Start gateway expecting it to fail during startup.
/// Returns the output lines captured before the process exited.
#[allow(dead_code)]
pub async fn start_gateway_expect_failure(
    config_bind_address: Option<&str>,
    cli_bind_address: &str,
    config_suffix: &str,
) -> Vec<String> {
    let bind_address_config = config_bind_address
        .map(|addr| format!("bind_address = \"{addr}\""))
        .unwrap_or_default();

    let config_str = format!(
        r"
        [gateway]
        {bind_address_config}
        {config_suffix}
    "
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
            "--bind-address",
            cli_bind_address,
        ])
        .env_remove("RUST_LOG")
        .env_remove("TENSORZERO_GATEWAY_BIND_ADDRESS")
        .kill_on_drop(true);

    let mut child = builder.spawn().unwrap();
    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();

    let mut output = Vec::new();
    while let Some(line) = stdout.next_line().await.unwrap() {
        println!("{line}");
        output.push(line);
    }

    // Wait for the process to exit
    let status = child.wait().await.unwrap();
    assert!(
        !status.success(),
        "Expected gateway to fail, but it exited with success"
    );

    output
}

async fn start_gateway_impl(
    config_bind_address: Option<&str>,
    cli_bind_address: Option<&str>,
    config_suffix: &str,
    rust_log: Option<&str>,
) -> ChildData {
    let bind_address_config = config_bind_address
        .map(|addr| format!("bind_address = \"{addr}\""))
        .unwrap_or_default();

    let config_str = format!(
        r"
        [gateway]
        {bind_address_config}
        {config_suffix}
    "
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
        .env_remove("TENSORZERO_GATEWAY_BIND_ADDRESS")
        .kill_on_drop(true);

    if let Some(cli_addr) = cli_bind_address {
        builder.args(["--bind-address", cli_addr]);
    }

    if let Some(rust_log) = rust_log {
        builder.env("RUST_LOG", rust_log);
    }

    let mut child = builder.spawn().unwrap();
    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();

    let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel();
    #[allow(clippy::disallowed_methods)]
    tokio::spawn(async move {
        while let Some(line) = stdout.next_line().await.unwrap() {
            println!("{line}");
            let _ = line_tx.send(line.clone());
        }
    });

    let mut listening_line = None;
    let mut output = Vec::new();
    while let Some(line) = line_rx.recv().await {
        if line.contains("listening on ") {
            listening_line = Some(line.clone());
        }
        output.push(line.clone());
        if line.contains("{\"message\":\"└") {
            // We're done logging the startup message
            break;
        }
    }

    // Parse the listening address from the log line.
    // The format is: "listening on <ip>:<port>"
    let listening_line = listening_line.expect("Gateway exited before listening");
    let addr_str = listening_line
        .split("listening on ")
        .nth(1)
        .expect("Gateway didn't log listening line")
        .split("\"")
        .next()
        .unwrap();
    let bound_addr: SocketAddr = addr_str.parse().expect("Failed to parse bound address");

    // For 0.0.0.0, tests must choose an appropriate address to connect to.
    // On Windows, connecting to 0.0.0.0 fails with AddrNotAvailable, so use loopback there.
    let connect_addr = if bound_addr.ip().is_unspecified() {
        SocketAddr::new(
            if cfg!(target_os = "windows") {
                "127.0.0.1".parse().unwrap()
            } else {
                bound_addr.ip()
            },
            bound_addr.port(),
        )
    } else {
        bound_addr
    };

    ChildData {
        addr: connect_addr,
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
