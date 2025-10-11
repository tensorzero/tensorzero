#![cfg(test)]

use std::sync::Arc;

use crate::config::Config;
use crate::db::clickhouse::clickhouse_client::FakeClickHouseClient;
use crate::utils::gateway::{GatewayHandle, GatewayHandleTestOptions};

pub fn get_unit_test_gateway_handle(config: Arc<Config>) -> GatewayHandle {
    let fake = FakeClickHouseClient::new(/* healthy= */ true);
    get_unit_test_gateway_handle_with_options(
        config,
        GatewayHandleTestOptions {
            clickhouse_client: Arc::new(fake),
            postgres_healthy: true,
        },
    )
}

pub fn get_unit_test_gateway_handle_with_options(
    config: Arc<Config>,
    test_options: GatewayHandleTestOptions,
) -> GatewayHandle {
    GatewayHandle::new_unit_test_data(config, test_options)
}
