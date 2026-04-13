use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_config_paths::{
    DATA_KEY, REMAPPED_PATH_KEY, WalkError, is_directory_path, walk_target_paths,
};
use toml::{
    Spanned, Table,
    de::{DeTable, DeValue},
    map::Entry,
};

use crate::error::{Error, ErrorDetails};
use crate::{config::span_map::SpanMap, error::IMPOSSIBLE_ERROR_MESSAGE};

/// Wrapper type to enforce proper handling of toml-relative paths for files.
/// When we add support for config globbing, we'll require deserializing
/// all paths (e.g. `system_schema`) as `ResolvedTomlPath`s, which will
/// track the original `.toml` file in order to perform correct relative path resolution.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ResolvedTomlPathData {
    __tensorzero_remapped_path: PathBuf,
    /// This should contain the data that was stored at the path above
    __data: String,
}

/// Wrapper type for directory paths (as opposed to file paths).
/// Currently only used for `gateway.template_filesystem_access.base_path`.
/// Unlike `ResolvedTomlPathData`, this doesn't eagerly load file contents since directories don't have contents.
/// This path is always stored as an absolute filesystem path because runtime
/// code needs to read from that directory after config loading.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

struct VisitorError(Error);

impl From<WalkError> for VisitorError {
    fn from(err: WalkError) -> Self {
        let inner = match err {
            WalkError::ExpectedTable { .. } => Error::new(ErrorDetails::Config {
                message: err.to_string(),
            }),
            WalkError::WildcardAtEnd { path } => Error::new(ErrorDetails::InternalError {
                message: format!(
                    "`{path}`: Path cannot end with a wildcard. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }),
        };
        VisitorError(inner)
    }
}

impl From<Error> for VisitorError {
    fn from(err: Error) -> Self {
        VisitorError(err)
    }
}

struct ResolveRelativePathsVisitor<'a> {
    span_map: &'a SpanMap,
    shared_path_prefix_to_strip: Option<&'a Path>,
}

struct CollectResolvedFilePathsVisitor<'a> {
    span_map: &'a SpanMap,
    resolved_file_paths: Vec<PathBuf>,
}

fn resolve_target_path(
    span_map: &SpanMap,
    span: std::ops::Range<usize>,
    target_string: &str,
    error_path: &[String],
) -> Result<PathBuf, Error> {
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

    let target_path = Path::new(target_string);
    base_path.join(target_path).canonicalize().map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!(
                "`{}`: Failed to resolve path `{}` (base: `{}`): {e}",
                error_path.join("."),
                target_path.display(),
                base_path.display(),
            ),
        })
    })
}

fn path_to_utf8_string(path: &Path, error_path: &[String]) -> Result<String, Error> {
    path.to_str().map(ToOwned::to_owned).ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!(
                "`{}`: Path was not valid utf-8: {path:?}",
                error_path.join(".")
            ),
        })
    })
}

fn compute_shared_path_prefix(paths: &[PathBuf]) -> Option<PathBuf> {
    let mut dirs = paths.iter().filter_map(|path| path.parent());
    let first = dirs.next()?;
    let mut components = first.components().collect::<Vec<_>>();

    for dir in dirs {
        let dir_components = dir.components().collect::<Vec<_>>();
        let shared_len = components
            .iter()
            .zip(dir_components.iter())
            .take_while(|(left, right)| left == right)
            .count();
        components.truncate(shared_len);
        if components.is_empty() {
            return None;
        }
    }

    let prefix: PathBuf = components.iter().collect();
    if prefix == Path::new("/") || prefix.as_os_str().is_empty() {
        return None;
    }
    Some(prefix)
}

impl tensorzero_config_paths::TomlPathVisitor<Spanned<DeValue<'_>>>
    for ResolveRelativePathsVisitor<'_>
{
    type Error = VisitorError;

    fn visit_leaf(
        &mut self,
        value: &mut Spanned<DeValue<'_>>,
        error_path: &[String],
    ) -> Result<(), Self::Error> {
        let span = value.span();
        let DeValue::String(target_string) = value.get_mut() else {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "`{}`: Expected a string, found {}",
                    error_path.join("."),
                    value.get_ref().type_str()
                ),
            })
            .into());
        };

        let resolved_path =
            resolve_target_path(self.span_map, span, target_string.as_ref(), error_path)?;
        let remapped_path = self
            .shared_path_prefix_to_strip
            .and_then(|prefix| resolved_path.strip_prefix(prefix).ok())
            .unwrap_or(resolved_path.as_path());
        let remapped_path_str = path_to_utf8_string(remapped_path, error_path)?;

        let mut inner_table = DeTable::new();
        inner_table.insert(
            Spanned::new(0..0, Cow::Owned(REMAPPED_PATH_KEY.to_string())),
            Spanned::new(0..0, DeValue::String(Cow::Owned(remapped_path_str))),
        );

        if resolved_path.is_dir() {
            if !is_directory_path(error_path) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "`{}`: Expected a file path, but '{}' is a directory. Please provide a path to a file.",
                        error_path.join("."),
                        resolved_path.display()
                    ),
                })
                .into());
            }
        } else {
            let file_contents = std::fs::read_to_string(&resolved_path).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "`{}`: Failed to read file at {}: {}",
                        error_path.join("."),
                        resolved_path.display(),
                        e
                    ),
                })
            })?;
            inner_table.insert(
                Spanned::new(0..0, Cow::Owned(DATA_KEY.to_string())),
                Spanned::new(0..0, DeValue::String(Cow::Owned(file_contents))),
            );
        }

        *value = Spanned::new(0..0, DeValue::Table(inner_table));
        Ok(())
    }
}

impl tensorzero_config_paths::TomlPathVisitor<Spanned<DeValue<'_>>>
    for CollectResolvedFilePathsVisitor<'_>
{
    type Error = VisitorError;

    fn visit_leaf(
        &mut self,
        value: &mut Spanned<DeValue<'_>>,
        error_path: &[String],
    ) -> Result<(), Self::Error> {
        let span = value.span();
        let DeValue::String(target_string) = value.get_ref() else {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "`{}`: Expected a string, found {}",
                    error_path.join("."),
                    value.get_ref().type_str()
                ),
            })
            .into());
        };

        let resolved_path =
            resolve_target_path(self.span_map, span, target_string.as_ref(), error_path)?;
        if resolved_path.is_dir() {
            if !is_directory_path(error_path) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "`{}`: Expected a file path, but '{}' is a directory. Please provide a path to a file.",
                        error_path.join("."),
                        resolved_path.display()
                    ),
                })
                .into());
            }
            return Ok(());
        }

        self.resolved_file_paths.push(resolved_path);
        Ok(())
    }
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
    let mut root = Spanned::new(0..0, DeValue::Table(table));
    let mut collector = CollectResolvedFilePathsVisitor {
        span_map,
        resolved_file_paths: Vec::new(),
    };
    walk_target_paths(&mut root, &mut collector).map_err(|VisitorError(error)| error)?;

    let shared_path_prefix_to_strip = compute_shared_path_prefix(&collector.resolved_file_paths);
    let mut visitor = ResolveRelativePathsVisitor {
        span_map,
        shared_path_prefix_to_strip: shared_path_prefix_to_strip.as_deref(),
    };
    walk_target_paths(&mut root, &mut visitor).map_err(|VisitorError(error)| error)?;
    let value = de_value_to_value(root.into_inner())?;
    match value {
        toml::Value::Table(table) => Ok(table),
        _ => Err(ErrorDetails::InternalError {
            message: format!("Root is not a table. {IMPOSSIBLE_ERROR_MESSAGE}"),
        }
        .into()),
    }
}
