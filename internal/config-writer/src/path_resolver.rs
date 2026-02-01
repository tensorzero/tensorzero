use std::path::{Path, PathBuf};

use crate::error::ConfigWriterError;

/// Information about a template or schema file that needs to be written.
#[derive(Debug, Clone)]
pub struct FileToWrite {
    /// The absolute path where the file should be written
    pub absolute_path: PathBuf,
    /// The content to write
    pub content: String,
}

/// Context for generating standardized paths for a variant.
pub struct VariantPathContext<'a> {
    pub glob_base: &'a Path,
    pub toml_file_dir: &'a Path,
    pub function_name: &'a str,
    pub variant_name: &'a str,
}

/// Context for generating standardized paths for an evaluator variant.
pub struct EvaluatorPathContext<'a> {
    pub glob_base: &'a Path,
    pub toml_file_dir: &'a Path,
    pub evaluation_name: &'a str,
    pub evaluator_name: &'a str,
    pub variant_name: &'a str,
}

impl<'a> VariantPathContext<'a> {
    /// Generate a standardized path for a variant template file.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn template_path(
        &self,
        template_kind: &str,
    ) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{template_kind}.minijinja"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for a variant schema file.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn schema_path(&self, schema_kind: &str) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{schema_kind}.json"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for system_instructions.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn system_instructions_path(&self) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join("system_instructions.txt");

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }
}

impl<'a> EvaluatorPathContext<'a> {
    /// Generate a standardized path for an evaluator variant's system_instructions.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn system_instructions_path(&self) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("evaluations")
            .join(self.evaluation_name)
            .join("evaluators")
            .join(self.evaluator_name)
            .join("variants")
            .join(self.variant_name)
            .join("system_instructions.txt");

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for an evaluator variant template.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn template_path(
        &self,
        template_kind: &str,
    ) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("evaluations")
            .join(self.evaluation_name)
            .join("evaluators")
            .join(self.evaluator_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{template_kind}.minijinja"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }
}

/// Compute a relative path from a base directory to a target file.
fn compute_relative_path(from_dir: &Path, to_file: &Path) -> Result<String, ConfigWriterError> {
    pathdiff::diff_paths(to_file, from_dir)
        .ok_or_else(|| ConfigWriterError::Path {
            message: format!(
                "Cannot compute relative path from `{}` to `{}`",
                from_dir.display(),
                to_file.display()
            ),
        })?
        .to_str()
        .ok_or_else(|| ConfigWriterError::Path {
            message: format!("Path contains invalid UTF-8: {}", to_file.display()),
        })
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_relative_path() {
        let from = Path::new("/project/config");
        let to = Path::new("/project/config/functions/my_fn/variants/v1/template.minijinja");
        let result = compute_relative_path(from, to).expect("should compute relative path");
        assert_eq!(result, "functions/my_fn/variants/v1/template.minijinja");
    }
}
