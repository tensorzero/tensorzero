use std::path::PathBuf;
use std::{cmp::Ordering, ops::Range};
use toml::de::DeTable;
use toml::Table;

use crate::config_parser::path::{merge_tomls, resolve_toml_relative_paths};
use crate::config_parser::ConfigFileGlob;
use crate::error::{Error, ErrorDetails};

pub struct SpanMap {
    /// Each entry in this vector is a tuple of a byte range from the final merged config,
    /// and the `TomlFile` representing the original file.
    /// The ranges are disjoint, and are sorted in ascending order, which allows
    /// us to binary search to lookup a particular range.
    range_to_file: Vec<(Range<usize>, TomlFile)>,
}

struct TomlFile {
    path: PathBuf,
    base_path: PathBuf,
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
    /// All of the `TomlRelativePath` entries in the resulting `Table` have been remapped to
    /// take their source toml file into account.
    /// As a result, almost all consumers of the returned `Table` shouldn't need to care
    /// about globbing (the exception being the fallback logic for `[gateway.template_filesystem_access]`,
    /// which needs to check if we globbed exactly one file)
    pub fn from_glob(glob: &ConfigFileGlob) -> Result<(Self, Table), Error> {
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
        if !found_file {
            return Err(ErrorDetails::Glob {
                glob: glob.glob.to_string(),
                message: "No config files matched glob".to_string(),
            }
            .into());
        }
        let mut target_config = DeTable::new();
        for padded_str in &padded_strs {
            let parsed = DeTable::parse(padded_str).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse config file as valid TOML: {e}"),
                })
            })?;
            merge_tomls(&mut target_config, parsed.get_ref(), vec![])?;
        }
        let span_map = SpanMap { range_to_file };
        let final_table = resolve_toml_relative_paths(target_config, &span_map)?;
        Ok((span_map, final_table))
    }

    /// Obtains the base path for a given range. This range should come from a `Spanned` entry
    /// in the final `DeTable`
    pub(super) fn lookup_range_base_path(&self, range: Range<usize>) -> Option<&PathBuf> {
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
        Some(&self.range_to_file[idx].1.base_path)
    }

    /// If the glob matched exactly one file, return the path to that file (*not* the base path)
    pub fn get_single_file(&self) -> Option<&PathBuf> {
        if let [(_range, single_file)] = self.range_to_file.as_slice() {
            Some(&single_file.path)
        } else {
            None
        }
    }
}
