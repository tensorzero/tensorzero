#![cfg(test)]

use std::sync::Arc;

use crate::config_parser::Config;
use crate::gateway_util::AppStateData;

pub fn get_unit_test_app_state_data(config: Arc<Config>, clickhouse_healthy: bool) -> AppStateData {
    AppStateData::new_unit_test_data(config, clickhouse_healthy)
}
