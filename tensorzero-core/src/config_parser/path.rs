use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Wrapper type to enforce proper handling of toml-relative paths.
/// When we add support for config globbing, we'll require deserializing
/// all paths (e.g. `system_schema`) as `TomlRelativePath`s, which will
/// track the original `.toml` file in order to perform correct relative path resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(transparent)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TomlRelativePath {
    path: PathBuf,
}

impl TomlRelativePath {
    /// Creates a new 'fake path' - this is currently used to construct
    /// `tensorzero::llm_judge` template paths for evaluators
    pub fn new_fake_path(fake_path: String) -> Self {
        Self {
            path: PathBuf::from(fake_path),
        }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Test-only method for unit tests.
    /// This allows constructing a `TomlRelativePath` outside of deserializing from a toml file
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_for_tests(buf: PathBuf) -> Self {
        Self { path: buf }
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl From<&str> for TomlRelativePath {
    fn from(path: &str) -> Self {
        TomlRelativePath {
            path: PathBuf::from(path),
        }
    }
}
