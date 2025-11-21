use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use toml::{
    de::{DeTable, DeValue},
    map::Entry,
    Spanned, Table,
};

use crate::error::{Error, ErrorDetails};
use crate::{config::span_map::SpanMap, error::IMPOSSIBLE_ERROR_MESSAGE};

/// Wrapper type to enforce proper handling of toml-relative paths for files.
/// When we add support for config globbing, we'll require deserializing
/// all paths (e.g. `system_schema`) as `ResolvedTomlPath`s, which will
/// track the original `.toml` file in order to perform correct relative path resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ResolvedTomlPathData {
    __tensorzero_remapped_path: PathBuf,
    /// This should contain the data that was stored at the path above
    __data: String,
}

/// Wrapper type for directory paths (as opposed to file paths).
/// Currently only used for `gateway.template_filesystem_access.base_path`.
/// Unlike `ResolvedTomlPathData`, this doesn't eagerly load file contents since directories don't have contents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ResolvedTomlPathDirectory {
    __tensorzero_remapped_path: PathBuf,
}

impl ResolvedTomlPathData {
    /// Creates a new 'fake path' - this is currently used to construct
    /// `tensorzero::llm_judge` template paths for evaluators
    pub fn new_fake_path(fake_path: String, data: String) -> Self {
        Self {
            __tensorzero_remapped_path: PathBuf::from(fake_path),
            __data: data,
        }
    }

    /// Obtains the key for templating purposes (may not be a real path)
    pub fn get_template_key(&self) -> String {
        self.__tensorzero_remapped_path.display().to_string()
    }

    /// Returns true if this is a real filesystem path (not a synthetic "fake path")
    pub fn is_real_path(&self) -> bool {
        // Fake paths use a special prefix like "tensorzero::llm_judge"
        !self.get_template_key().starts_with("tensorzero::")
    }

    /// Obtains the real path for this path, if it is a real path.
    /// If it is a fake path, like those passed in from the dynamic variant config
    /// this returns an error.
    pub fn get_real_path(&self) -> Result<&Path, Error> {
        if !self.is_real_path() {
            return Err(ErrorDetails::InternalError {
                message: "Attempted to get real path for a fake path with data".to_string(),
            }
            .into());
        }
        Ok(self.__tensorzero_remapped_path.as_ref())
    }

    pub fn data(&self) -> &str {
        &self.__data
    }

    /// Test-only method for unit tests.
    /// This allows constructing a `ResolvedTomlPath` outside of deserializing from a toml file
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_for_tests(buf: PathBuf, data_override: Option<String>) -> Self {
        let data = if let Some(data) = data_override {
            data
        } else {
            // If no data override provided, try to read from filesystem
            std::fs::read_to_string(&buf).unwrap_or_else(|_| String::new())
        };
        Self {
            __tensorzero_remapped_path: buf,
            __data: data,
        }
    }
}

impl ResolvedTomlPathDirectory {
    /// Obtains the real path for this directory path.
    pub fn get_real_path(&self) -> &Path {
        self.__tensorzero_remapped_path.as_ref()
    }

    /// Test-only method for unit tests.
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_for_tests(buf: PathBuf) -> Self {
        Self {
            __tensorzero_remapped_path: buf,
        }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<&str> for ResolvedTomlPathData {
    fn from(path: &str) -> Self {
        let buf = PathBuf::from(path);
        let data = std::fs::read_to_string(&buf).unwrap_or_else(|_| String::new());
        ResolvedTomlPathData {
            __tensorzero_remapped_path: buf,
            __data: data,
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

/// Merges all of the keys from 'source' into 'target'.
/// This is conservative, and does not allow overwriting an existing keys in `target`
/// (if the same key is mapped to a table in both `source` and `target`, then we recursively merge)
pub(super) fn merge_tomls<'a>(
    target: &mut DeTable<'a>,
    source: &DeTable<'a>,
    span_map: &SpanMap,
    error_key_path: Vec<String>,
) -> Result<(), Error> {
    for (key, value) in source {
        let mut error_path = error_key_path.clone();
        error_path.push(key.get_ref().to_string());
        match target.entry(key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(value.clone());
            }
            Entry::Occupied(mut entry) => match entry.get_mut().get_mut() {
                DeValue::String(_)
                | DeValue::Integer(_)
                | DeValue::Float(_)
                | DeValue::Boolean(_)
                | DeValue::Array(_)
                | DeValue::Datetime(_) => {
                    let target_file = span_map
                        .lookup_range(entry.key().span())
                        .map(|f| f.path().clone())
                        .unwrap_or_else(|| PathBuf::from("<unknown TOML file>"));
                    let source_file = span_map
                        .lookup_range(key.span())
                        .map(|f| f.path().clone())
                        .unwrap_or_else(|| PathBuf::from("<unknown TOML file>"));
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "`{}`: Found duplicate values in globbed TOML config files `{}` and `{}`",
                            error_path.join("."),
                            target_file.display(),
                            source_file.display(),
                        ),
                    }
                    .into());
                }
                DeValue::Table(target_table) => {
                    if let DeValue::Table(source_table) = value.get_ref() {
                        merge_tomls(target_table, source_table, span_map, error_path)?;
                    } else {
                        let source_file = span_map
                            .lookup_range(key.span())
                            .map(|f| f.path().clone())
                            .unwrap_or_else(|| PathBuf::from("<unknown TOML file>"));
                        let target_file = span_map
                            .lookup_range(entry.key().span())
                            .map(|f| f.path().clone())
                            .unwrap_or_else(|| PathBuf::from("<unknown TOML file>"));
                        return Err(ErrorDetails::Config {
                            message: format!(
                                "`{}`: Cannot merge `{}` from file `{}` into a table from file `{}`",
                                error_path.join("."),
                                value.get_ref().type_str(),
                                source_file.display(),
                                target_file.display(),
                            ),
                        }
                        .into());
                    }
                }
            },
        }
    }
    Ok(())
}

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
/// while the output table stores nested tables of the form expected by the `ResolvedTomlPath` deserializer.
/// This ensures that missing entries in `TARGET_PATH_COMPONENTS` produce an error if we try to deserialize
/// a `ResolvedTomlPath`, rather than silently deserializing to an incorrect path.
///
/// Our `DeTable` was deserialized from a a string consisting of concatenated config files
/// (chosen from the glob passed in on the command line). We use the provided `SpanMap` to
/// map a particular entry back to its original TOML source file, to determine the base path
pub(super) fn resolve_toml_relative_paths(
    table: DeTable<'_>,
    span_map: &SpanMap,
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
            // the `__tensorzero_remapped_path` table expected by the `ResolvedTomlPath` deserializer
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
                            let span = entry.span();
                            if let DeValue::String(target_string) = entry.get_mut() {
                                let base_path = span_map
                                    .lookup_range(span)
                                    .ok_or_else(|| {
                                        Error::new(ErrorDetails::Config {
                                            message: format!(
                                            "`{}`: Failed to determine original TOML source file",
                                            error_path.join(".")
                                        ),
                                        })
                                    })?
                                    .base_path();

                                let target_path = Path::new(&**target_string);
                                let mut inner_table = DeTable::new();

                                // Resolve the absolute path
                                let resolved_path = base_path.join(target_path);
                                let resolved_path_str = resolved_path
                                    .to_str()
                                    .ok_or_else(|| {
                                        Error::new(ErrorDetails::Config {
                                            message: format!(
                                                "`{}`: Path was not valid utf-8: base_path={base_path:?}, target_path={target_path:?}",
                                                error_path.join(".")
                                            ),
                                        })
                                    })?
                                    .to_string();

                                // Check if this is a directory or file
                                let is_directory_path = error_path.as_slice()
                                    == ["gateway", "template_filesystem_access", "base_path"];

                                if resolved_path.is_dir() {
                                    if !is_directory_path {
                                        // All other paths should be files, not directories
                                        return Err(Error::new(ErrorDetails::Config {
                                            message: format!(
                                                "`{}`: Expected a file path, but '{}' is a directory. Please provide a path to a file.",
                                                error_path.join("."),
                                                resolved_path.display()
                                            ),
                                        }));
                                    }
                                    // For the directory path (gateway.template_filesystem_access.base_path),
                                    // create a table with only __tensorzero_remapped_path (no __data)
                                    inner_table.insert(
                                        Spanned::new(
                                            0..0,
                                            Cow::Owned("__tensorzero_remapped_path".to_string()),
                                        ),
                                        Spanned::new(
                                            0..0,
                                            DeValue::String(Cow::Owned(resolved_path_str)),
                                        ),
                                    );
                                } else {
                                    // For file paths, eagerly read the file contents
                                    let file_contents = std::fs::read_to_string(&resolved_path)
                                        .map_err(|e| {
                                            Error::new(ErrorDetails::Config {
                                                message: format!(
                                                    "`{}`: Failed to read file at {}: {}",
                                                    error_path.join("."),
                                                    resolved_path.display(),
                                                    e
                                                ),
                                            })
                                        })?;

                                    // We use dummy spans for now - this may change when we implement globbing
                                    inner_table.insert(
                                        Spanned::new(
                                            0..0,
                                            Cow::Owned("__tensorzero_remapped_path".to_string()),
                                        ),
                                        Spanned::new(
                                            0..0,
                                            DeValue::String(Cow::Owned(resolved_path_str)),
                                        ),
                                    );
                                    inner_table.insert(
                                        Spanned::new(0..0, Cow::Owned("__data".to_string())),
                                        Spanned::new(
                                            0..0,
                                            DeValue::String(Cow::Owned(file_contents)),
                                        ),
                                    );
                                }
                                // Overwrite the original path value with the appropriate table structure
                                // For files: `{"__tensorzero_remapped_path": "...", "__data": "..."}`
                                // For directories: `{"__tensorzero_remapped_path": "..."}`
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
