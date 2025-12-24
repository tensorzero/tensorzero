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

/// Assert that the gateway fails to start, and return the output lines.
async fn try_start_gateway_expect_failure(config_suffix: &str) -> Vec<String> {
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

    // Read output for up to 20 seconds
    let timeout = tokio::time::sleep(tokio::time::Duration::from_secs(20));
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

    for line in &output {
        println!("{line}");
    }

    assert!(failed, "Gateway should have failed to start");

    output
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

/// Test that shorthand models (e.g., openai::gpt-4o-mini) work in relay mode
/// when the edge gateway lacks credentials but the downstream gateway has them.
///
/// Setup:
/// - Edge gateway: relay config, NO OPENAI_API_KEY
/// - Downstream gateway: empty config, HAS OPENAI_API_KEY
/// - Inference uses shorthand model openai::gpt-4o-mini
#[tokio::test]
async fn test_relay_mode_allows_invalid_credentials_shorthand() {
    tensorzero_unsafe_helpers::set_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION", "1");

    // Get the API key from environment - skip test if not available
    let openai_api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required");

    // Start downstream gateway WITH OPENAI_API_KEY
    // Empty config - shorthand models are resolved at request time
    let downstream_config = "";

    let downstream_config_str = format!(
        r#"
        [gateway]
        bind_address = "0.0.0.0:0"
        {downstream_config}
    "#
    );

    let downstream_tmpfile = NamedTempFile::new().unwrap();
    std::fs::write(downstream_tmpfile.path(), downstream_config_str).unwrap();

    let mut downstream_child = Command::new(gateway_path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .args([
            "--config-file",
            downstream_tmpfile.path().to_str().unwrap(),
            "--log-format",
            "json",
        ])
        .env_remove("RUST_LOG")
        .env("OPENAI_API_KEY", &openai_api_key) // Downstream HAS the API key
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let downstream_port = wait_for_gateway_port(&mut downstream_child).await;

    // Start edge/relay gateway WITHOUT OPENAI_API_KEY
    let relay_config = format!(
        r#"
        [gateway]
        bind_address = "0.0.0.0:0"

        [gateway.relay]
        gateway_url = "http://0.0.0.0:{downstream_port}"
    "#
    );

    let relay_tmpfile = NamedTempFile::new().unwrap();
    std::fs::write(relay_tmpfile.path(), &relay_config).unwrap();

    let mut relay_child = Command::new(gateway_path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .args([
            "--config-file",
            relay_tmpfile.path().to_str().unwrap(),
            "--log-format",
            "json",
        ])
        .env_remove("RUST_LOG")
        .env_remove("OPENAI_API_KEY") // Edge does NOT have the API key
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let relay_port = wait_for_gateway_port(&mut relay_child).await;

    // Verify inference succeeds via relay using shorthand model
    // Edge gateway skips credential validation (relay mode)
    // Request relays to downstream which has OPENAI_API_KEY
    let response = Client::new()
        .post(format!("http://0.0.0.0:{relay_port}/inference"))
        .json(&json!({
            "model_name": "openai::gpt-4o-mini",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'hello' and nothing else."
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
        "Shorthand model in relay mode should work when downstream has credentials, got: {status}, body: {body_text}"
    );

    // Cleanup
    let _ = downstream_child.kill().await;
    let _ = relay_child.kill().await;

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}

/// Helper to wait for gateway to start and extract its port
async fn wait_for_gateway_port(child: &mut tokio::process::Child) -> u16 {
    let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();

    let mut listening_line = None;
    while let Some(line) = stdout.next_line().await.unwrap() {
        println!("{line}");
        if line.contains("listening on 0.0.0.0:") {
            listening_line = Some(line.clone());
        }
        if line.contains("{\"message\":\"└") {
            break;
        }
    }

    // Spawn a task to continue draining stdout so the process doesn't block
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(async move {
        while let Ok(Some(line)) = stdout.next_line().await {
            println!("{line}");
        }
    });

    listening_line
        .expect("Gateway exited before listening")
        .split_once("listening on 0.0.0.0:")
        .expect("Gateway didn't log listening line")
        .1
        .split("\"")
        .next()
        .unwrap()
        .parse::<u16>()
        .unwrap()
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

    let output = try_start_gateway_expect_failure(relay_config).await;
    let output_str = output.join("\n");

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

    let output = try_start_gateway_expect_failure(config).await;
    let output_str = output.join("\n");

    assert!(
        output_str.contains("TOTALLY_NONEXISTENT_KEY")
            || output_str.contains("missing")
            || output_str.contains("ApiKeyMissing"),
        "Expected error about missing credentials, got: {output_str}"
    );

    tensorzero_unsafe_helpers::remove_env_var_tests_only("TENSORZERO_E2E_CREDENTIAL_VALIDATION");
}
