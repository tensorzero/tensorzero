use std::path::PathBuf;
use std::{cmp::Ordering, ops::Range};
use toml::de::DeTable;
use toml::Table;

use crate::config::path::{merge_tomls, resolve_toml_relative_paths};
use crate::config::ConfigFileGlob;
use crate::error::{Error, ErrorDetails};

/// Holds range information for a merged config file.
/// The merged file is built up from several different config files.
/// While we don't parse a single concatenated file, we do adjust the ranges
/// within each individually-parsed file to be disjoint, allowing us to map
/// a `Spanned` range back to the original file.
pub struct SpanMap {
    /// Each entry in this vector is a tuple of a byte range from the final merged config,
    /// and the `TomlFile` representing the original file.
    /// The ranges are disjoint, and are sorted in ascending order, which allows
    /// us to binary search to lookup a particular range.
    range_to_file: Vec<(Range<usize>, TomlFile)>,
}

pub struct TomlFile {
    path: PathBuf,
    base_path: PathBuf,
}

impl TomlFile {
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn base_path(&self) -> &PathBuf {
        &self.base_path
    }
}

impl SpanMap {
    #[cfg(test)]
    pub fn new_empty() -> Self {
        Self {
            range_to_file: vec![],
        }
    }

    #[cfg(test)]
    pub fn new_single_file(path: PathBuf) -> Self {
        Self {
            range_to_file: vec![(
                0..usize::MAX,
                TomlFile {
                    base_path: path.parent().unwrap().to_owned(),
                    path,
                },
            )],
        }
    }

    /// Loads all config files matching the given glob, and merges them into a single `Table`
    /// All of the `ResolvedTomlPath` entries in the resulting `Table` have been remapped to
    /// take their source toml file into account.
    /// As a result, consumers of the returned `Table` don't need to care about globbing.
    pub fn from_glob(glob: &ConfigFileGlob, allow_empty: bool) -> Result<Table, Error> {
        let mut found_file = false;
        let mut range_to_file = Vec::new();
        let mut previous_range_end: usize = 0;
        // Due to lifetime restriction from `toml::de::DeTable`, we need all of the globbed config
        // contents to have the same lifetime. This increases peak memory usage during config loading,
        // but all of these temporary allocations get freed before we construct our final loaded `Config`
        let mut padded_strs = Vec::new();
        for file in &glob.paths {
            found_file = true;
            let base_path = file
                .parent()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Failed to determine base path for file `{}`",
                            file.to_string_lossy()
                        ),
                    })
                })?
                .to_owned();
            let contents = std::fs::read_to_string(file).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "Failed to read config file `{}`: {e}",
                        file.to_string_lossy()
                    ),
                })
            })?;

            // Hack to get non-overlapping span ranges for each file - we pad out each file with whitespace
            // characters (by the length of all of the previous files). The alternative is to walk each
            // `DeTable` and adjust every `Spanned` that we encounter, which is more error-prone.
            // This unfortunately increases peak memory usage, but all of these strings get freed after
            // we merge the configs.
            let whitespace_file =
                std::iter::repeat_n('\n', previous_range_end).collect::<String>() + &contents;
            let whitespace_file_len = whitespace_file.len();
            padded_strs.push(whitespace_file);

            range_to_file.push((
                previous_range_end..whitespace_file_len,
                TomlFile {
                    path: file.clone(),
                    base_path,
                },
            ));
            previous_range_end = whitespace_file_len;
        }
        if !found_file && !allow_empty {
            return Err(ErrorDetails::Glob {
                glob: glob.glob.to_string(),
                message: "No config files matched glob".to_string(),
            }
            .into());
        }
        let mut target_config = DeTable::new();
        let span_map = SpanMap { range_to_file };
        for padded_str in &padded_strs {
            let parsed = DeTable::parse(padded_str).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse config file as valid TOML: {e}"),
                })
            })?;
            merge_tomls(&mut target_config, parsed.get_ref(), &span_map, vec![])?;
        }
        let final_table = resolve_toml_relative_paths(target_config, &span_map)?;
        Ok(final_table)
    }

    /// Obtains the base path for a given range. This range should come from a `Spanned` entry
    /// in the final `DeTable`
    pub(super) fn lookup_range(&self, range: Range<usize>) -> Option<&TomlFile> {
        if range.end == 0 {
            return None;
        }
        let idx = self
            .range_to_file
            .binary_search_by(|(r, _)| {
                if r.contains(&range.start) && r.contains(&(range.end - 1)) {
                    Ordering::Equal
                } else if r.start < range.start {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .ok()?;
        Some(&self.range_to_file[idx].1)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use toml::{de::DeValue, Spanned};

    use super::*;

    #[test]
    fn test_resolve_toml_relative_paths() {
        // Create a temporary file to test with
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, r#"{{"test": "data"}}"#).unwrap();
        let temp_path = temp_file.path().to_path_buf();

        // Get the directory containing the temp file
        let temp_dir = temp_path.parent().unwrap();

        // Create a config that references this temp file
        let config_str = format!(
            r#"functions.my_function.system_schema = "{}""#,
            temp_path.file_name().unwrap().to_str().unwrap()
        );
        let table = DeTable::parse(&config_str).unwrap();

        // Use the temp directory as the base path
        let config_path = temp_dir.join("fake_config.toml");
        let resolved =
            resolve_toml_relative_paths(table.into_inner(), &SpanMap::new_single_file(config_path))
                .unwrap();

        // Verify the path was resolved and the data was loaded
        let schema_table = resolved["functions"]["my_function"]["system_schema"]
            .as_table()
            .unwrap();
        assert_eq!(
            schema_table["__tensorzero_remapped_path"].as_str().unwrap(),
            temp_path.to_str().unwrap()
        );
        assert_eq!(
            schema_table["__data"].as_str().unwrap(),
            r#"{"test": "data"}"#
        );
    }

    #[test]
    fn test_invalid_resolve_toml_relative_paths() {
        let table = DeTable::parse("functions.my_function.system_schema = 123").unwrap();
        let err =
            resolve_toml_relative_paths(table.into_inner(), &SpanMap::new_empty()).unwrap_err();
        assert_eq!(
            *err.get_details(),
            ErrorDetails::Config {
                message: "`functions.my_function.system_schema`: Expected a string, found integer"
                    .to_string(),
            }
        );
    }

    #[test]
    fn test_merge_empty() {
        let mut target = DeTable::new();
        let source = DeTable::new();
        let span_map = SpanMap::new_empty();
        let error_path = vec![];
        merge_tomls(&mut target, &source, &span_map, error_path).unwrap();
        assert_eq!(target.len(), 0);
    }

    #[test]
    fn test_merge_invalid_type() {
        let mut target = DeTable::new();
        let mut inner_table = DeTable::new();
        inner_table.insert(
            Spanned::new(0..0, Cow::Borrowed("inner_key")),
            Spanned::new(0..0, DeValue::String(Cow::Borrowed("inner_value"))),
        );
        target.insert(
            Spanned::new(0..0, Cow::Borrowed("outer_table")),
            Spanned::new(0..0, DeValue::Table(inner_table)),
        );
        let mut source = DeTable::new();
        source.insert(
            Spanned::new(0..0, Cow::Borrowed("outer_table")),
            Spanned::new(0..0, DeValue::String(Cow::Borrowed("outer_value"))),
        );
        let span_map = SpanMap::new_empty();
        let error_path = vec![];
        let err = merge_tomls(&mut target, &source, &span_map, error_path).unwrap_err();
        assert_eq!(
            *err.get_details(),
            ErrorDetails::Config {
                message: "`outer_table`: Cannot merge `string` from file `<unknown TOML file>` into a table from file `<unknown TOML file>`".to_string(),
            }
        );
    }

    #[test]
    fn test_merge_duplicate_key() {
        let mut target = DeTable::new();
        let mut target_inner = DeTable::new();
        target_inner.insert(
            Spanned::new(0..1, Cow::Borrowed("inner_key")),
            Spanned::new(0..1, DeValue::String(Cow::Borrowed("target_value"))),
        );
        target.insert(
            Spanned::new(0..1, Cow::Borrowed("outer_table")),
            Spanned::new(0..1, DeValue::Table(target_inner)),
        );

        let mut source = DeTable::new();
        let mut source_inner = DeTable::new();
        source_inner.insert(
            Spanned::new(1..2, Cow::Borrowed("inner_key")),
            Spanned::new(1..2, DeValue::String(Cow::Borrowed("source_value"))),
        );
        source.insert(
            Spanned::new(1..2, Cow::Borrowed("outer_table")),
            Spanned::new(1..2, DeValue::Table(source_inner)),
        );
        let span_map = SpanMap {
            range_to_file: vec![
                (
                    0..1,
                    TomlFile {
                        path: PathBuf::from("target.toml"),
                        base_path: PathBuf::from("."),
                    },
                ),
                (
                    1..2,
                    TomlFile {
                        path: PathBuf::from("source.toml"),
                        base_path: PathBuf::from("."),
                    },
                ),
            ],
        };
        let error_path = vec![];
        let err = merge_tomls(&mut target, &source, &span_map, error_path).unwrap_err();
        assert_eq!(
            *err.get_details(),
            ErrorDetails::Config {
                message: "`outer_table.inner_key`: Found duplicate values in globbed TOML config files `target.toml` and `source.toml`".to_string(),
            }
        );
    }
}
