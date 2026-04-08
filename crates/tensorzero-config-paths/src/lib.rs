//! Shared definitions for config path patterns used by tensorzero-core (config loading)
//! and config-applier (config serialization).

pub mod walker;

pub use walker::{
    TomlPathVisitor, TomlTableMut, TomlTreeMut, WalkError, walk_pattern, walk_patterns,
    walk_target_paths, walk_target_paths_from_prefix,
};

use std::collections::HashMap;

use thiserror::Error;

/// A component in a path pattern for matching config keys.
#[derive(Debug, Copy, Clone)]
pub enum PathComponent {
    /// Matches a specific key name exactly.
    Literal(&'static str),
    /// Matches any key name (used for user-defined names like function names).
    Wildcard,
}

pub const REMAPPED_PATH_KEY: &str = "__tensorzero_remapped_path";
pub const DATA_KEY: &str = "__data";

/// This stores patterns for every possible file path that can be stored in a TensorZero config file.
/// For example, `functions.my_function.system_schema = "some/relative/schema_path.json"` is matched by
/// `&[PathComponent::Literal("functions"), PathComponent::Wildcard, PathComponent::Literal("system_schema")]`
///
/// Any config struct that stores a `ResolvedTomlPath` needs a corresponding entry in this array.
/// If an entry is missing, then deserializing the struct will fail, as the `ResolvedTomlPath` deserializer
/// expects a table produced by `resolve_paths`.
///
/// During config loading, we pre-process the `toml::de::DeTable`, and convert all entries located at
/// `TARGET_PATH_COMPONENTS` (which should be strings) into absolute paths, using the source TOML file
/// as the base path. For example, `functions.my_function.system_schema = "some/relative/schema_path.json"
/// will become `functions.my_function.system_schema = { __tensorzero_remapped_path = "base/directory/some/relative/schema_path.json" }`
///
/// This allows us to abstract over config file globbing, and allow almost all of the codebase to work with
/// absolute paths, without needing to know which TOML file a particular path was originally written in.
///
/// You should avoid declaring a `PathBuf` inside any TensorZero config structs, unless you're certain
/// that the path should not be relative to the TOML file that it's written in.
///
/// One alternative we considered was use `Spanned<PathBuf>` in our config structs, and deserialize from
/// a `toml::de::DeTable`. Unfortunately, this breaks whenever serde has an internal 'boundary'
/// (internally-tagged enums, `#[serde(flatten)]`, and possible other attributes). In these cases, serde
/// will deserialize into its own custom type (consuming the original `Deserializer`), and continue
/// deserializing with the internal serde `Deserializer`. This causes any extra information to get
/// lost (including the span information held by the `toml::de::DeTable` deserializer).
/// While it might be possible to work around this (similar to what we do for error messages in
/// the `TensorZeroDeserialize` macro), this is a load-bearing part of the codebase, and implicitly
/// depends on internal Serde implementation details.
pub static TARGET_PATH_COMPONENTS: &[&[PathComponent]] = &[
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("output_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("output_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("schemas"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("assistant_template"),
    ],
    // Function-scoped evaluators
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("assistant_template"),
    ],
    // Top-level evaluations
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("tools"),
        PathComponent::Wildcard,
        PathComponent::Literal("parameters"),
    ],
    &[
        PathComponent::Literal("gateway"),
        PathComponent::Literal("template_filesystem_access"),
        PathComponent::Literal("base_path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
];

#[derive(Debug, Error)]
pub enum RemappedTomlPathError {
    #[error("`{path}`: Expected a table, found {found}")]
    ExpectedTable { path: String, found: String },
    #[error("`{path}`: Expected a string, found {found}")]
    ExpectedString { path: String, found: String },
    #[error("`{path}`: Expected string `{key}` in remapped path table")]
    MissingStringKey { path: String, key: &'static str },
    #[error("`{path}`: Missing editable file contents for path `{target_path}`")]
    MissingPathContents { path: String, target_path: String },
    #[error(
        "`{path}`: Path `{target_path}` was provided with conflicting contents in editable config serialization"
    )]
    ConflictingPathContents { path: String, target_path: String },
    #[error(transparent)]
    Walk(#[from] WalkError),
}

struct LeafFnVisitor<F> {
    visit: F,
}

impl<E, F> TomlPathVisitor<toml::Value> for LeafFnVisitor<F>
where
    E: From<WalkError>,
    F: FnMut(&mut toml::Value, &[String]) -> Result<(), E>,
{
    type Error = E;

    fn visit_leaf(&mut self, value: &mut toml::Value, path: &[String]) -> Result<(), Self::Error> {
        (self.visit)(value, path)
    }
}

pub fn visit_target_path_values<E, F>(root: &mut toml::Value, visit_leaf: F) -> Result<(), E>
where
    E: From<WalkError>,
    F: FnMut(&mut toml::Value, &[String]) -> Result<(), E>,
{
    let mut visitor = LeafFnVisitor { visit: visit_leaf };
    walk_target_paths(root, &mut visitor)
}

pub fn is_directory_path(path: &[String]) -> bool {
    matches!(
        path,
        [gateway, access, base_path]
            if gateway == "gateway"
                && access == "template_filesystem_access"
                && base_path == "base_path"
    )
}

pub fn extract_target_path_contents(
    root: &mut toml::Value,
    path_contents: &mut HashMap<String, String>,
) -> Result<(), RemappedTomlPathError> {
    visit_target_path_values(root, |value, error_path| {
        let table = value
            .as_table()
            .ok_or_else(|| RemappedTomlPathError::ExpectedTable {
                path: error_path.join("."),
                found: value.type_str().to_string(),
            })?;

        let path = table_string_field(table, error_path, REMAPPED_PATH_KEY)?.to_string();
        if is_directory_path(error_path) {
            *value = toml::Value::String(path);
            return Ok(());
        }

        let contents = table_string_field(table, error_path, DATA_KEY)?.to_string();
        if let Some(existing) = path_contents.get(&path) {
            if existing != &contents {
                return Err(RemappedTomlPathError::ConflictingPathContents {
                    path: error_path.join("."),
                    target_path: path,
                });
            }
        } else {
            path_contents.insert(path.clone(), contents);
        }

        *value = toml::Value::String(path);
        Ok(())
    })
}

pub fn resolve_target_path_tables_from_contents(
    root: &mut toml::Value,
    path_contents: &HashMap<String, String>,
) -> Result<(), RemappedTomlPathError> {
    visit_target_path_values(root, |value, error_path| {
        let toml::Value::String(path) = value else {
            return Err(RemappedTomlPathError::ExpectedString {
                path: error_path.join("."),
                found: value.type_str().to_string(),
            });
        };

        let mut inner_table = toml::Table::new();
        inner_table.insert(
            REMAPPED_PATH_KEY.to_string(),
            toml::Value::String(path.clone()),
        );

        if !is_directory_path(error_path) {
            let contents = path_contents.get(path).ok_or_else(|| {
                RemappedTomlPathError::MissingPathContents {
                    path: error_path.join("."),
                    target_path: path.clone(),
                }
            })?;
            inner_table.insert(DATA_KEY.to_string(), toml::Value::String(contents.clone()));
        }

        *value = toml::Value::Table(inner_table);
        Ok(())
    })
}

fn table_string_field<'a>(
    table: &'a toml::Table,
    error_path: &[String],
    key: &'static str,
) -> Result<&'a str, RemappedTomlPathError> {
    table.get(key).and_then(toml::Value::as_str).ok_or_else(|| {
        RemappedTomlPathError::MissingStringKey {
            path: error_path.join("."),
            key,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use googletest::prelude::*;

    use super::*;

    fn sample_root() -> toml::Value {
        toml::Value::Table(toml::toml! {
            [functions.test_function.variants.test_variant]
            system_template = { __tensorzero_remapped_path = "/tmp/system.minijinja", __data = "system" }

            [gateway.template_filesystem_access]
            base_path = { __tensorzero_remapped_path = "/tmp/templates" }
        })
    }

    #[gtest]
    fn extract_target_path_contents_rewrites_tables_to_strings() {
        let mut root = sample_root();
        let mut path_contents = HashMap::new();

        extract_target_path_contents(&mut root, &mut path_contents)
            .expect("extracting target path contents should succeed");

        expect_that!(
            root.get("functions")
                .and_then(|value| value.get("test_function"))
                .and_then(|value| value.get("variants"))
                .and_then(|value| value.get("test_variant"))
                .and_then(|value| value.get("system_template"))
                .and_then(toml::Value::as_str),
            some(eq("/tmp/system.minijinja"))
        );
        expect_that!(
            root.get("gateway")
                .and_then(|value| value.get("template_filesystem_access"))
                .and_then(|value| value.get("base_path"))
                .and_then(toml::Value::as_str),
            some(eq("/tmp/templates"))
        );
        expect_that!(
            path_contents.get("/tmp/system.minijinja"),
            some(eq(&"system".to_string()))
        );
    }

    #[gtest]
    fn resolve_target_path_tables_from_contents_restores_file_and_directory_tables() {
        let mut root = toml::Value::Table(toml::toml! {
            [functions.test_function.variants.test_variant]
            system_template = "/tmp/system.minijinja"

            [gateway.template_filesystem_access]
            base_path = "/tmp/templates"
        });
        let path_contents =
            HashMap::from([("/tmp/system.minijinja".to_string(), "system".to_string())]);

        resolve_target_path_tables_from_contents(&mut root, &path_contents)
            .expect("restoring target path tables should succeed");

        expect_that!(
            root.get("functions")
                .and_then(|value| value.get("test_function"))
                .and_then(|value| value.get("variants"))
                .and_then(|value| value.get("test_variant"))
                .and_then(|value| value.get("system_template"))
                .and_then(toml::Value::as_table)
                .and_then(|table| table.get(REMAPPED_PATH_KEY))
                .and_then(toml::Value::as_str),
            some(eq("/tmp/system.minijinja"))
        );
        expect_that!(
            root.get("functions")
                .and_then(|value| value.get("test_function"))
                .and_then(|value| value.get("variants"))
                .and_then(|value| value.get("test_variant"))
                .and_then(|value| value.get("system_template"))
                .and_then(toml::Value::as_table)
                .and_then(|table| table.get(DATA_KEY))
                .and_then(toml::Value::as_str),
            some(eq("system"))
        );
        expect_that!(
            root.get("gateway")
                .and_then(|value| value.get("template_filesystem_access"))
                .and_then(|value| value.get("base_path"))
                .and_then(toml::Value::as_table)
                .map(|table| table.contains_key(DATA_KEY)),
            some(eq(false))
        );
    }
}
