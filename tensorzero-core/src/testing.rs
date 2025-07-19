#![cfg(test)]

use std::sync::Arc;

use crate::config_parser::Config;
use crate::gateway_util::AppStateData;

pub const ALL_DIGITS: u64 = 1_234_567_890;
pub const MAY_3_2021: u64 = 1_620_000_000;
pub const JAN_1_2000: u64 = 946_684_800;
pub const ONE_THRU_SIX: u64 = 123_456;

pub fn get_unit_test_app_state_data(config: Arc<Config>, clickhouse_healthy: bool) -> AppStateData {
    AppStateData::new_unit_test_data(config, clickhouse_healthy)
}
