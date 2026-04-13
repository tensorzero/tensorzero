use std::collections::HashMap;

use tensorzero_config_paths::{
    extract_target_path_contents, resolve_target_path_tables_from_contents,
};

use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};

use super::UninitializedConfig;

pub fn config_to_toml(
    config: &UninitializedConfig,
) -> Result<(String, HashMap<String, String>), Error> {
    let mut root = toml::Value::try_from(config).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize editable config TOML: {e}"),
        })
    })?;
    let mut path_contents = HashMap::new();
    extract_path_contents(&mut root, &mut path_contents)?;

    let table = match root {
        toml::Value::Table(table) => table,
        _ => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Editable config TOML root is not a table. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }));
        }
    };

    let toml = toml::to_string_pretty(&sort_toml_table(table)).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to render editable config TOML: {e}"),
        })
    })?;
    Ok((toml, path_contents))
}

pub fn toml_to_config(
    toml_str: &str,
    path_contents: &HashMap<String, String>,
) -> Result<UninitializedConfig, Error> {
    let table: toml::Table = toml_str.parse().map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to parse editable config TOML: {e}"),
        })
    })?;
    let mut root = toml::Value::Table(table);
    resolve_target_path_tables_from_contents(&mut root, path_contents).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: e.to_string(),
        })
    })?;
    let table = match root {
        toml::Value::Table(table) => table,
        _ => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Editable config TOML root is not a table. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }));
        }
    };
    UninitializedConfig::try_from(table)
}

fn extract_path_contents(
    root: &mut toml::Value,
    path_contents: &mut HashMap<String, String>,
) -> Result<(), Error> {
    extract_target_path_contents(root, path_contents).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: e.to_string(),
        })
    })
}

fn sort_toml_table(raw_config: toml::Table) -> toml::Table {
    let mut sorted_table = toml::Table::new();
    let mut keys: Vec<_> = raw_config.keys().cloned().collect();
    keys.sort();

    for key in keys {
        if let Some(value) = raw_config.get(&key) {
            let processed_value = match value {
                toml::Value::Table(table) => toml::Value::Table(sort_toml_table(table.clone())),
                toml::Value::Array(items) => toml::Value::Array(
                    items
                        .iter()
                        .map(sort_toml_value)
                        .collect::<Vec<toml::Value>>(),
                ),
                value => value.clone(),
            };
            sorted_table.insert(key, processed_value);
        }
    }

    sorted_table
}

fn sort_toml_value(value: &toml::Value) -> toml::Value {
    match value {
        toml::Value::Table(table) => toml::Value::Table(sort_toml_table(table.clone())),
        toml::Value::Array(values) => {
            toml::Value::Array(values.iter().map(sort_toml_value).collect())
        }
        value => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use googletest::prelude::*;
    use toml::de::DeTable;

    use crate::config::{SpanMap, path::resolve_toml_relative_paths};

    use super::*;

    fn sample_uninitialized_config() -> UninitializedConfig {
        let config_str = include_str!("../../fixtures/config/tensorzero.toml");
        tensorzero_unsafe_helpers::set_env_var_tests_only("OPENAI_API_KEY", "sk-something");
        tensorzero_unsafe_helpers::set_env_var_tests_only("ANTHROPIC_API_KEY", "sk-something");
        tensorzero_unsafe_helpers::set_env_var_tests_only("AZURE_API_KEY", "sk-something");

        let table = DeTable::parse(config_str).expect("sample config TOML should parse");
        let fake_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fake_path.toml");
        let resolved =
            resolve_toml_relative_paths(table.into_inner(), &SpanMap::new_single_file(fake_path))
                .expect("sample config paths should resolve");

        UninitializedConfig::try_from(resolved).expect("sample config should deserialize")
    }

    #[gtest]
    fn editable_toml_round_trips_sample_config() {
        let config = sample_uninitialized_config();

        let (toml, path_contents) =
            config_to_toml(&config).expect("editable TOML serialization should succeed");
        let round_trip = toml_to_config(&toml, &path_contents)
            .expect("editable TOML deserialization should succeed");

        expect_that!(&round_trip, eq(&config));
    }

    #[gtest]
    fn editable_toml_uses_plain_path_strings() {
        let config = sample_uninitialized_config();

        let (toml, path_contents) =
            config_to_toml(&config).expect("editable TOML serialization should succeed");
        let parsed = toml::Value::Table(
            toml::from_str::<toml::Table>(&toml).expect("editable TOML should parse"),
        );

        expect_that!(
            parsed
                .get("functions")
                .and_then(|value| value.get("generate_draft"))
                .and_then(|value| value.get("variants"))
                .and_then(|value| value.get("openai_promptA"))
                .and_then(|value| value.get("system_template"))
                .and_then(toml::Value::as_str),
            some(contains_substring("system_template.minijinja"))
        );
        expect_that!(
            parsed
                .get("tools")
                .and_then(|value| value.get("get_temperature"))
                .and_then(|value| value.get("parameters"))
                .and_then(toml::Value::as_str),
            some(contains_substring("get_temperature.json"))
        );
        expect_that!(
            path_contents
                .keys()
                .any(|key| key.contains("system_template.minijinja")),
            eq(true)
        );
        expect_that!(
            path_contents
                .keys()
                .any(|key| key.contains("get_temperature.json")),
            eq(true)
        );
    }

    #[gtest]
    fn editable_toml_errors_when_path_contents_are_missing() {
        let config = sample_uninitialized_config();
        let (toml, mut path_contents) =
            config_to_toml(&config).expect("editable TOML serialization should succeed");
        let removed_key = path_contents
            .keys()
            .find(|key| key.contains("system_template.minijinja"))
            .cloned()
            .expect("sample config should have a system template path");
        path_contents.remove(&removed_key);

        expect_that!(
            toml_to_config(&toml, &path_contents),
            err(displays_as(contains_substring(removed_key)))
        );
    }
}
