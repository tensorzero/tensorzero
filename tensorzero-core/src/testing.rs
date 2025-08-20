#![cfg(test)]

use std::sync::Arc;

use crate::config::Config;
use crate::gateway_util::GatewayHandle;

pub fn get_unit_test_gateway_handle(
    config: Arc<Config>,
    clickhouse_healthy: bool,
) -> GatewayHandle {
    GatewayHandle::new_unit_test_data(config, clickhouse_healthy)
}
