use crate::common::{ChildData, start_gateway_on_random_port};

/// Test environment with both downstream and relay gateways.
#[allow(dead_code, clippy::allow_attributes)]
pub struct RelayTestEnvironment {
    pub downstream: ChildData,
    pub relay: ChildData,
}

/// Spawns a relay test environment with both downstream and relay gateways.
#[allow(dead_code, clippy::allow_attributes)]
pub async fn start_relay_test_environment(
    downstream_config: &str,
    relay_config_suffix: &str,
) -> RelayTestEnvironment {
    // Start downstream gateway first
    let downstream = start_gateway_on_random_port(downstream_config, None).await;

    // Build relay configuration with downstream port injected
    let relay_config = format!(
        r#"
[gateway.relay]
gateway_url = "http://0.0.0.0:{}"

{}
"#,
        downstream.addr.port(),
        relay_config_suffix
    );

    // Start relay gateway
    let relay = start_gateway_on_random_port(&relay_config, None).await;

    RelayTestEnvironment { downstream, relay }
}
