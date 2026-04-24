use std::collections::HashMap;

use tensorzero_config_paths::{
    extract_target_path_contents, resolve_target_path_tables_from_contents,
};

use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};

use super::{ConfigLoadingError, UninitializedConfig};

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

/// Like [`config_to_toml`], but also appends broken items as commented-out
/// TOML fragments so the editor shows them for debugging/fixing rather than
/// silently omitting them (which would cause them to be deleted on the next
/// save).
pub fn config_to_toml_with_errors(
    config: &UninitializedConfig,
    errors: &[ConfigLoadingError],
) -> Result<(String, HashMap<String, String>), Error> {
    let (mut toml, path_contents) = config_to_toml(config)?;

    for err in errors {
        let label = match &err.parent {
            Some(parent) => format!("{} / {}", parent, err.name),
            None => err.name.clone(),
        };
        // Split the error message on newlines so a multi-line error (e.g. a
        // serde path error) doesn't spill past the `#` and corrupt the TOML.
        // `{:?}` on `label` escapes any embedded newlines in the parent/name
        // fields themselves.
        let mut error_lines = err.error.lines();
        let first_error_line = error_lines.next().unwrap_or("");
        toml.push_str(&format!(
            "\n# BROKEN ({} {:?}): {}\n",
            err.kind, label, first_error_line
        ));
        for line in error_lines {
            toml.push_str(&format!("#   {line}\n"));
        }
        if let Some(fragment) = &err.raw_toml {
            for line in fragment.lines() {
                toml.push_str(&format!("# {line}\n"));
            }
        } else {
            toml.push_str("# (raw config not available)\n");
        }
    }

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

    // ─── config_to_toml_with_errors ────────────────────────────────────────

    #[gtest]
    fn config_to_toml_with_errors_with_empty_list_matches_plain_config_to_toml() {
        // No errors → output must be identical to config_to_toml.
        let config = sample_uninitialized_config();
        let (plain, plain_paths) =
            config_to_toml(&config).expect("plain serialization should succeed");
        let (annotated, annotated_paths) = config_to_toml_with_errors(&config, &[])
            .expect("annotated serialization should succeed");

        expect_that!(&annotated, eq(&plain));
        expect_that!(&annotated_paths, eq(&plain_paths));
    }

    #[gtest]
    fn config_to_toml_with_errors_appends_broken_header_and_raw_fragment() {
        let config = sample_uninitialized_config();
        let errors = vec![ConfigLoadingError {
            kind: "model",
            name: "gpt-broken".to_string(),
            parent: None,
            error: "unsupported schema revision 999".to_string(),
            raw_toml: Some("type = \"chat_completion\"\nmodel_name = \"gpt-broken\"".to_string()),
        }];
        let (annotated, _) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        // Header mentions kind, name, and the error message.
        expect_that!(
            &annotated,
            contains_substring(r#"# BROKEN (model "gpt-broken"): unsupported schema revision 999"#)
        );
        // Raw TOML is emitted as commented-out lines.
        expect_that!(
            &annotated,
            contains_substring(r#"# type = "chat_completion""#)
        );
        expect_that!(
            &annotated,
            contains_substring(r#"# model_name = "gpt-broken""#)
        );
    }

    #[gtest]
    fn config_to_toml_with_errors_uses_parent_slash_name_label() {
        let config = sample_uninitialized_config();
        let errors = vec![ConfigLoadingError {
            kind: "variant",
            name: "primary".to_string(),
            parent: Some("generate_draft".to_string()),
            error: "missing variant version".to_string(),
            raw_toml: None,
        }];
        let (annotated, _) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        expect_that!(
            &annotated,
            contains_substring(
                r#"# BROKEN (variant "generate_draft / primary"): missing variant version"#
            )
        );
    }

    #[gtest]
    fn config_to_toml_with_errors_marks_missing_raw_toml() {
        let config = sample_uninitialized_config();
        let errors = vec![ConfigLoadingError {
            kind: "tool",
            name: "broken_tool".to_string(),
            parent: None,
            error: "bad parameters file".to_string(),
            raw_toml: None,
        }];
        let (annotated, _) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        expect_that!(
            &annotated,
            contains_substring("# (raw config not available)")
        );
    }

    #[gtest]
    fn config_to_toml_with_errors_appends_multiple_errors_in_order() {
        let config = sample_uninitialized_config();
        let errors = vec![
            ConfigLoadingError {
                kind: "model",
                name: "first".to_string(),
                parent: None,
                error: "error one".to_string(),
                raw_toml: None,
            },
            ConfigLoadingError {
                kind: "function",
                name: "second".to_string(),
                parent: None,
                error: "error two".to_string(),
                raw_toml: None,
            },
        ];
        let (annotated, _) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        let first_idx = annotated
            .find(r#"# BROKEN (model "first"): error one"#)
            .expect("first error header should appear");
        let second_idx = annotated
            .find(r#"# BROKEN (function "second"): error two"#)
            .expect("second error header should appear");
        expect_that!(first_idx < second_idx, eq(true));
    }

    #[gtest]
    fn config_to_toml_with_errors_round_trips_back_to_original_config() {
        // The `# BROKEN:` annotations are TOML comments, so parsing the
        // annotated output must reproduce the same config. This guards against
        // regressions where the annotation format drifts into something that
        // the TOML parser treats as non-comment content.
        let config = sample_uninitialized_config();
        let errors = vec![
            ConfigLoadingError {
                kind: "model",
                name: "broken".to_string(),
                parent: None,
                error: "bad schema revision".to_string(),
                raw_toml: Some(
                    "type = \"chat_completion\"\nendpoint = \"https://x.test\"".to_string(),
                ),
            },
            ConfigLoadingError {
                kind: "variant",
                name: "alt".to_string(),
                parent: Some("generate_draft".to_string()),
                error: "missing variant version".to_string(),
                raw_toml: None,
            },
        ];
        let (annotated, path_contents) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        let round_trip =
            toml_to_config(&annotated, &path_contents).expect("annotated TOML should parse back");
        expect_that!(&round_trip, eq(&config));
    }

    #[gtest]
    fn config_to_toml_with_errors_handles_multiline_error_messages() {
        // Serde deserialization errors are frequently multi-line. Each line of
        // the error must stay on a commented line so the annotated TOML still
        // parses — otherwise the second line of a two-line error would become
        // uncommented TOML content after the header's trailing newline.
        let config = sample_uninitialized_config();
        let errors = vec![ConfigLoadingError {
            kind: "model",
            name: "broken".to_string(),
            parent: None,
            error: "missing field `type`\nat line 3 column 1\nin table `[models.broken]`"
                .to_string(),
            raw_toml: Some("foo = 1".to_string()),
        }];
        let (annotated, path_contents) = config_to_toml_with_errors(&config, &errors)
            .expect("annotated serialization should succeed");

        // The first line appears on the header, subsequent lines are each
        // prefixed with `#   `.
        expect_that!(
            &annotated,
            contains_substring(r#"# BROKEN (model "broken"): missing field `type`"#)
        );
        expect_that!(&annotated, contains_substring("#   at line 3 column 1"));
        expect_that!(
            &annotated,
            contains_substring("#   in table `[models.broken]`")
        );

        // And the annotated TOML must still round-trip: if a multi-line error
        // leaked past the `#`, the parser would either fail or pick up the
        // leaked text as config.
        let round_trip =
            toml_to_config(&annotated, &path_contents).expect("annotated TOML should parse back");
        expect_that!(&round_trip, eq(&config));
    }

    #[gtest]
    fn config_to_toml_with_errors_is_deterministic_for_cas_signature() {
        // The CAS signature in the apply handler is computed over the TOML
        // string returned by this function. Running it twice on the same
        // inputs must produce byte-identical output.
        let config = sample_uninitialized_config();
        let errors = vec![ConfigLoadingError {
            kind: "model",
            name: "gpt-broken".to_string(),
            parent: None,
            error: "schema revision 999".to_string(),
            raw_toml: Some("a = 1\nb = 2".to_string()),
        }];

        let (first, _) = config_to_toml_with_errors(&config, &errors)
            .expect("first serialization should succeed");
        let (second, _) = config_to_toml_with_errors(&config, &errors)
            .expect("second serialization should succeed");
        expect_that!(&first, eq(&second));
    }
}
