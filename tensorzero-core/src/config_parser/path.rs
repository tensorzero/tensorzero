#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::error::{Error, ErrorDetails};
use serde::{Deserialize, Serialize};
use toml::{
    de::{DeTable, DeValue},
    Spanned, Table,
};

/// Wrapper type to enforce proper handling of toml-relative paths.
/// When we add support for config globbing, we'll require deserializing
/// all paths (e.g. `system_schema`) as `TomlRelativePath`s, which will
/// track the original `.toml` file in order to perform correct relative path resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
/// TODO: Add a pyclass attribute to the TomlRelativePath struct
#[cfg_attr(feature = "pyo3", pyclass(str, name = "TomlRelativePath"))]
pub struct TomlRelativePath {
    __tensorzero_remapped_path: PathBuf,
    /// This should be set for dynamic variants to indicate what the file contents would have been at this remapped path.
    #[serde(default)]
    __data: Option<String>,
}

impl std::fmt::Display for TomlRelativePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__tensorzero_remapped_path.display())
    }
}

impl TomlRelativePath {
    /// Creates a new 'fake path' - this is currently used to construct
    /// `tensorzero::llm_judge` template paths for evaluators
    pub fn new_fake_path(fake_path: String, data: String) -> Self {
        Self {
            __tensorzero_remapped_path: PathBuf::from(fake_path),
            __data: Some(data),
        }
    }

    /// Obtains the key for templating purposes (may not be a real path)
    pub fn get_template_key(&self) -> String {
        self.__tensorzero_remapped_path.display().to_string()
    }

    /// Obtains the real path for this path, if it is a real path.
    /// If it is a fake path, like those passed in from the dynamic variant config
    /// this returns an error.
    pub fn get_real_path(&self) -> Result<&Path, Error> {
        if self.__data.is_some() {
            return Err(ErrorDetails::InternalError {
                message: "Attempted to get real path for a fake path with data".to_string(),
            }
            .into());
        }
        Ok(self.__tensorzero_remapped_path.as_ref())
    }

    /// Obtains the data that this path contains.
    /// For a real path this will read from the file system.
    /// For a fake path this will return the data that was passed in.
    pub fn read(&self) -> Result<String, Error> {
        if let Some(data) = &self.__data {
            Ok(data.clone())
        } else {
            std::fs::read_to_string(&self.__tensorzero_remapped_path).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "Failed to read file at {}: {}",
                        self.__tensorzero_remapped_path.to_string_lossy(),
                        e
                    ),
                })
            })
        }
    }

    /// Test-only method for unit tests.
    /// This allows constructing a `TomlRelativePath` outside of deserializing from a toml file
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_for_tests(buf: PathBuf, data: Option<String>) -> Self {
        Self {
            __tensorzero_remapped_path: buf,
            __data: data,
        }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<&str> for TomlRelativePath {
    fn from(path: &str) -> Self {
        TomlRelativePath {
            __tensorzero_remapped_path: PathBuf::from(path),
            __data: None,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum PathComponent {
    Literal(&'static str),
    Wildcard,
}

/// This stores patterns for every possible file path that can be stored in a TensorZero config file.
/// For example, `functions.my_function.system_schema = "some/relative/schema_path.json"` is matched by
/// `&[PathComponent::Literal("functions"), PathComponent::Wildcard, PathComponent::Literal("system_schema")]`
///
/// Any config struct that stores a `TomlRelativePath` needs a corresponding entry in this array.
/// If an entry is missing, then deserializing the struct will fail, as the `TomlRelativePath` deserializer
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
static TARGET_PATH_COMPONENTS: &[&[PathComponent]] = &[
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
];

/// Converts a `toml::DeValue` to a `toml::Value`.
/// This just removes all of the `Spanned` wrappers, and leaves the value otherwise unchanged.
pub(super) fn de_value_to_value(value: DeValue<'_>) -> Result<toml::Value, Error> {
    let value = match value {
        DeValue::String(string) => toml::Value::String(string.to_string()),
        DeValue::Integer(integer) => {
            toml::Value::Integer(integer.to_string().parse().map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse integer: {e}"),
                })
            })?)
        }
        DeValue::Float(float) => toml::Value::Float(float.to_string().parse().map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse float: {e}"),
            })
        })?),
        DeValue::Boolean(boolean) => toml::Value::Boolean(boolean),
        DeValue::Array(array) => toml::Value::Array(
            array
                .into_iter()
                .map(|val| de_value_to_value(val.into_inner()))
                .collect::<Result<Vec<toml::Value>, Error>>()?,
        ),
        DeValue::Datetime(datetime) => toml::Value::Datetime(datetime),
        DeValue::Table(table) => toml::Value::Table(
            table
                .into_iter()
                .map(|(key, value)| {
                    let key = key.into_inner().to_string();
                    let value = de_value_to_value(value.into_inner())?;
                    Ok((key, value))
                })
                .collect::<Result<Table, Error>>()?,
        ),
    };
    Ok(value)
}

struct TargetData<'a, 'b> {
    /// The current PathComponent we're visiting
    component: PathComponent,
    /// The remaining components in our path
    tail: &'b [PathComponent],
    /// The entry we're visiting
    entry: &'b mut DeValue<'a>,
    /// The path in the user-specified config that we visited to reach `entry`
    /// We use this when reporting errors
    error_path: Vec<String>,
}

/// Visits all of the entries declared in `TARGET_PATH_COMPONENTS`, and resolves relative paths into
/// absolute paths. The original entries held string paths written by the user
/// (e.g. `functions.my_function.system_schema = "some/relative/schema_path.json"`),
/// while the output table stores nested tables of the form expected by the `TomlRelativePath` deserializer.
/// This ensures that missing entries in `TARGET_PATH_COMPONENTS` produce an error if we try to deserialize
/// a `TomlRelativePath`, rather than silently deserializing to an incorrect path.
///
/// This currently uses the provided `base_path` to resolve relative paths. Once we add config globbing support,
/// this will be removed in favor of using the `Spanned` data from the original `DeTable`
pub(super) fn resolve_toml_relative_paths(
    table: DeTable<'_>,
    base_path: &Path,
) -> Result<Table, Error> {
    let mut root = DeValue::Table(table);
    for path in TARGET_PATH_COMPONENTS {
        let mut targets = vec![TargetData {
            component: path[0],
            tail: &path[1..],
            entry: &mut root,
            error_path: vec![],
        }];
        while let Some(target_data) = targets.pop() {
            let mut error_path = target_data.error_path.clone();
            // We're reached the last component in our path - look up the key, and replace the value with
            // the `__tensorzero_remapped_path` table expected by the `TomlRelativePath` deserializer
            if target_data.tail.is_empty() {
                match target_data.component {
                    PathComponent::Literal(literal) => {
                        error_path.push(literal.to_string());
                        let DeValue::Table(entry) = target_data.entry else {
                            return Err(ErrorDetails::Config {
                                message: format!(
                                    "`{}`: Expected a table, found {}",
                                    error_path.join("."),
                                    target_data.entry.type_str()
                                ),
                            }
                            .into());
                        };
                        // Spanned ignores the span for Hash/PartialEq, so we can use a dummy span for the lookup
                        if let Some(entry) = entry.get_mut(literal) {
                            if let DeValue::String(target_string) = entry.get_mut() {
                                let target_path = Path::new(&**target_string);
                                let mut inner_table = DeTable::new();

                                // We use dummy spans for now - this may change when we implement globbing
                                inner_table.insert(
                                    Spanned::new(
                                        0..0,
                                        Cow::Owned("__tensorzero_remapped_path".to_string()),
                                    ),
                                    Spanned::new(
                                        0..0,
                                        DeValue::String(Cow::Owned(
                                            // Note - when we implement globbing, we'll obtain `base_path` using the span of the `entry`
                                            base_path
                                                .join(target_path)
                                                .to_str()
                                                .ok_or_else(|| {
                                                    Error::new(ErrorDetails::Config {
                                                        message: format!(
                                                            "`{}`: Path was not valid utf-8: base_path={base_path:?}, target_path={target_path:?}",
                                                            error_path.join(".")
                                                        ),
                                                    })
                                                })?
                                                .to_string(),
                                        )),
                                    ),
                                );
                                // Overwrite the original `"relative/schema_path.json"` value with
                                // a table that looks like `{"__tensorzero_remapped_path": "/my/base/path/relative/schema_path.json"}`
                                *entry = Spanned::new(0..0, DeValue::Table(inner_table));
                            } else {
                                return Err(ErrorDetails::Config {
                                    message: format!(
                                        "`{}`: Expected a string, found {}",
                                        error_path.join("."),
                                        entry.get_ref().type_str()
                                    ),
                                }
                                .into());
                            }
                        }
                    }
                    PathComponent::Wildcard => {
                        return Err(ErrorDetails::InternalError {
                            message: format!(
                                "`{}`: Path cannot end with a wildcard. {IMPOSSIBLE_ERROR_MESSAGE}",
                                error_path.join(".")
                            ),
                        }
                        .into())
                    }
                }
            } else {
                // We're not at the end of the path, so we push new entries to our 'targets' stack
                match target_data.component {
                    PathComponent::Literal(literal) => {
                        error_path.push(literal.to_string());
                        let DeValue::Table(entry) = target_data.entry else {
                            return Err(ErrorDetails::Config {
                                message: format!(
                                    "`{}`: Expected a table, found {}",
                                    error_path.join("."),
                                    target_data.entry.type_str()
                                ),
                            }
                            .into());
                        };
                        // If the literal is present in the user-provided table, traverse into the value
                        if let Some(entry) = entry.get_mut(literal) {
                            targets.push(TargetData {
                                component: target_data.tail[0],
                                tail: &target_data.tail[1..],
                                entry: entry.get_mut(),
                                error_path,
                            });
                        }
                    }
                    // For wildcards, push all of the table values onto our stack. This is used to process
                    // all entries within a table (e.g. `[functions.first_function]`, `[functions.second_function]`, etc.)
                    PathComponent::Wildcard => {
                        if let DeValue::Table(table) = target_data.entry {
                            for (key, value) in table.iter_mut() {
                                let mut error_path = error_path.clone();
                                error_path.push(key.get_ref().to_string());
                                targets.push(TargetData {
                                    component: target_data.tail[0],
                                    tail: &target_data.tail[1..],
                                    entry: value.get_mut(),
                                    error_path,
                                });
                            }
                        } else {
                            return Err(ErrorDetails::Config {
                                message: format!(
                                    "`{}`: Expected a table, found {}",
                                    target_data.error_path.join("."),
                                    target_data.entry.type_str()
                                ),
                            }
                            .into());
                        }
                    }
                }
            }
        }
    }
    let value = de_value_to_value(root)?;
    match value {
        toml::Value::Table(table) => Ok(table),
        _ => Err(ErrorDetails::InternalError {
            message: format!("Root is not a table. {IMPOSSIBLE_ERROR_MESSAGE}"),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_toml_relative_paths() {
        let table =
            DeTable::parse(r#"functions.my_function.system_schema = "relative/schema_path.json""#)
                .unwrap();

        let resolved =
            resolve_toml_relative_paths(table.into_inner(), Path::new("my/base/path")).unwrap();
        assert_eq!(
            resolved,
            toml::from_str(
                r#"functions.my_function.system_schema = { __tensorzero_remapped_path = "my/base/path/relative/schema_path.json" }"#
            )
            .unwrap()
        );
    }

    #[test]
    fn test_invalid_resolve_toml_relative_paths() {
        let table = DeTable::parse("functions.my_function.system_schema = 123").unwrap();
        let err =
            resolve_toml_relative_paths(table.into_inner(), Path::new("my/base/path")).unwrap_err();
        assert_eq!(
            *err.get_details(),
            ErrorDetails::Config {
                message: "`functions.my_function.system_schema`: Expected a string, found integer"
                    .to_string(),
            }
        );
    }
}
