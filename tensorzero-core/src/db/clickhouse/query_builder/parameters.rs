use crate::db::clickhouse::query_builder::{ClickhouseType, QueryParameter};

/// Helper to add a parameter and return its SQL placeholder {name:CHType}
/// The internal_name (e.g. p0, p1) is stored in params_map with its value.
/// Returns the sql placeholder string {param_name:ClickHouseType}
pub fn add_parameter<T: ToString>(
    value: T,
    clickhouse_type: ClickhouseType,
    params_map: &mut Vec<QueryParameter>,
    counter: &mut usize,
) -> String {
    let internal_name = format!("p{}", *counter);
    *counter += 1;
    params_map.push(QueryParameter {
        name: internal_name.clone(),
        value: value.to_string(),
    });
    format!("{{{internal_name}:{clickhouse_type}}}")
}
