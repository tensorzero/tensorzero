#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod common;

use common::{gateway_path, relay::start_relay_test_environment, start_gateway_on_random_port};
use reqwest::Client;
use serde_json::json;
use std::process::Stdio;
use tempfile::NamedTempFile;
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use uuid::Uuid;

/// Try to start gateway and return whether it failed to start
async fn try_start_gateway_expect_failure(config_suffix: &str) -> (bool, Vec<String>) {
    let config_str = format!(
        r#"
        [gateway]
        bind_address = "0.0.0.0:0"
        {config_suffix}
    "#
    );

    let tmpfile = NamedTempFile::new().unwrap();
    std::fs::write(tmpfile.path(), config_str).unwrap();

    let mut child = Command::new(gateway_path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .args([
            "--config-file",
            tmpfile.path().to_str().unwrap(),
            "--log-format",
            "json",
        ])
        .env_remove("RUST_LOG")
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();

    let mut output = Vec::new();
    let mut failed = false;

    // Read output for up to 2 seconds
    let timeout = tokio::time::sleep(tokio::time::Duration::from_secs(2));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            line = stdout.next_line() => {
                if let Ok(Some(line)) = line {
                    println!("{line}");
                    output.push(line.clone());
                    if line.contains("listening on 0.0.0.0:") {
                        // Gateway started successfully
                        break;
                    }
                } else {
                    // stdout closed, gateway exited
                    failed = true;
                    break;
                }
            }
            line = stderr.next_line() => {
                if let Ok(Some(line)) = line {
                    eprintln!("{line}");
                    output.push(line.clone());
                }
            }
            () = &mut timeout => {
                // Timeout, check if still running
                if let Ok(Some(_)) = child.try_wait() {
                    failed = true;
                }
                break;
            }
        }
    }

    // Ensure child is killed and waited for
    let _ = child.kill().await;
    let _ = child.wait().await;

    // Drop handles to close pipes
    drop(stdout);
    drop(stderr);

    (failed, output)
}

#[tokio::test]
async fn test_relay_mode_allows_invalid_credentials() {
    tensorzero_unsafe_helpers::set_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION", "1");

    let downstream_config = r#"
[models.test_model]
routing = ["good"]

[models.test_model.providers.good]
type = "dummy"
model_name = "good"
"#;

    let relay_config = r#"
[models.test_model]
routing = ["invalid_creds"]

[models.test_model.providers.invalid_creds]
type = "openai"
model_name = "gpt-4"
api_key_location = "env::NONEXISTENT_API_KEY"
"#;

    // Should start successfully - relay mode allows invalid credentials since they won't be used
    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Verify inference succeeds via relay (using downstream's valid credentials)
    let response = Client::new()
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "test_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "test"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(
        status, 200,
        "Relay mode should work with invalid local credentials, got: {status}, body: {body_text}"
    );

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}

#[tokio::test]
async fn test_skip_relay_with_valid_credentials_succeeds() {
    tensorzero_unsafe_helpers::set_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION", "1");

    let relay_config = r#"
[gateway.relay]
gateway_url = "http://tensorzero.invalid"

[models.local_model]
routing = ["good"]
skip_relay = true

[models.local_model.providers.good]
type = "dummy"
model_name = "good"
"#;

    // Should start successfully because dummy provider with no api_key_location is valid
    let relay = start_gateway_on_random_port(relay_config, None).await;

    // Verify inference succeeds using local provider
    let response = Client::new()
        .post(format!("http://{}/inference", relay.addr))
        .json(&json!({
            "model_name": "local_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "test"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    assert_eq!(
        status, 200,
        "Expected success with local provider, got: {status}, body: {body_text}"
    );

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}

#[tokio::test]
async fn test_skip_relay_with_invalid_credentials_fails() {
    tensorzero_unsafe_helpers::set_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION", "1");

    let relay_config = r#"
[gateway.relay]
gateway_url = "http://tensorzero.invalid"

[models.local_model]
routing = ["bad_creds"]
skip_relay = true

[models.local_model.providers.bad_creds]
type = "openai"
model_name = "gpt-4"
api_key_location = "env::MISSING_LOCAL_KEY"
"#;

    let (failed, output) = try_start_gateway_expect_failure(relay_config).await;
    let output_str = output.join("\n");

    assert!(
        failed,
        "Gateway should fail to start with invalid credentials for skip_relay model, got: {output_str}"
    );

    assert!(
        output_str.contains("MISSING_LOCAL_KEY") || output_str.contains("missing"),
        "Expected error about missing credentials, got: {output_str}"
    );

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}

#[tokio::test]
async fn test_invalid_credentials_fail_without_relay() {
    tensorzero_unsafe_helpers::set_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION", "1");

    let config = r#"
[models.test_model]
routing = ["bad_creds"]

[models.test_model.providers.bad_creds]
type = "openai"
model_name = "gpt-4"
api_key_location = "env::TOTALLY_NONEXISTENT_KEY"
"#;

    let (failed, output) = try_start_gateway_expect_failure(config).await;
    let output_str = output.join("\n");

    assert!(
        failed,
        "Gateway should fail to start with invalid credentials (no relay mode), got: {output_str}"
    );

    assert!(
        output_str.contains("TOTALLY_NONEXISTENT_KEY")
            || output_str.contains("missing")
            || output_str.contains("ApiKeyMissing"),
        "Expected error about missing credentials, got: {output_str}"
    );

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}
