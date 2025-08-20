use std::{fmt::format, fs, path::PathBuf};

use fs2::FileExt;
use tracing::error;

use crate::inference::types::current_timestamp;
use crate::{
    error::{Error, ErrorDetails},
    variant::VariantConfig,
};

use super::{Config, ConfigFileGlob};

pub fn write_variant_config(
    variant: &VariantConfig,
    new_variant_name: &str,
    existing_function_name: &str,
    config_glob: &ConfigFileGlob,
    config: &Config,
) -> Result<(), Error> {
    // First, we'll figure out what file to write to
    let mut toml_write_path = None;
    for path in config_glob.paths.iter() {
        if check_file_for_function_type_key(path, existing_function_name)? {
            toml_write_path = Some(path.clone());
            // Since we assume here that the config has been parsed,
            // we can assume the key shows up at most once and break here.
            break;
        };
    }
    let Some(toml_write_path) = toml_write_path else {
        return Err(ErrorDetails::MissingFunctionTypeKey {
            function_name: existing_function_name.to_string(),
        }
        .into());
    };
    // Create lock file path in the directory containing the TOML file
    let lock_file_path = toml_write_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".tensorzero.lock");

    let lock_file = std::fs::File::create(&lock_file_path).map_err(|e| ErrorDetails::Lock {
        message: format!(
            "Failed to create lock file {}: {}",
            lock_file_path.display(),
            e
        ),
    })?;

    // Let's lock the directory containing the file
    // so we can do all our writes without contending for the config
    // TODO:
    // Non-blocking lock attempt
    // We need to do this in a loop with a timeout so we can fail gracefully if the lock is broken
    // For now I'm just locking it
    /*
    match lock_file.try_lock_exclusive() {
        Ok(()) => {
            // Got the lock, proceed with work
            // ...
            lock_file.unlock()?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            println!("Another process is using the directory");
            // Handle the contention case
        }
        Err(e) => return Err(e), // Other error
        }
        */
    // @Aaron not sure the best pattern to establish for this operation in our codebase. Open to suggestions!
    lock_file.lock_exclusive().map_err(|e| ErrorDetails::Lock {
        message: format!("Failed to lock file {}: {}", lock_file_path.display(), e),
    })?;
    let result = inner_write_variant_config(&toml_write_path, variant, existing_function_name)
        .map_err(|e| e.into());
    FileExt::unlock(&lock_file).map_err(|e| ErrorDetails::Lock {
        message: format!("Failed to unlock file {}: {}", lock_file_path.display(), e),
    })?;
    result
}

pub fn inner_write_variant_config(
    toml_write_path: &PathBuf,
    variant: &VariantConfig,
    existing_function_name: &str,
) -> Result<(), WriteFilesError> {
    // For each file in the variant configuration,
    // Blocked on variant changes. Can't do this now
    todo!()
}

fn check_file_for_function_type_key(
    path: &PathBuf,
    existing_function_name: &str,
) -> Result<bool, Error> {
    // First, let's parse the toml with serde_toml
    let toml_content = fs::read_to_string(path).map_err(|e| {
        Error::new(ErrorDetails::FileWrite {
            message: e.to_string(),
            file_path: path.display().to_string(),
        })
    })?;
    check_for_function_type_key(&toml_content, existing_function_name, path)
}

fn check_for_function_type_key(
    toml_content: &str,
    existing_function_name: &str,
    path: &PathBuf,
) -> Result<bool, Error> {
    let parsed_toml: toml::Value = toml::from_str(toml_content).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to parse TOML at {}: {}", path.display(), e),
        })
    })?;
    // Check if the path "functions.$function_name.type" exists in this file
    Ok(parsed_toml
        .get("functions")
        .and_then(|functions| {
            functions
                .get(existing_function_name)
                .and_then(|function| function.get("type"))
        })
        .is_some())
}

struct WriteFilesError {
    error: Error,
}

impl WriteFilesError {
    pub fn new(error: Error, files_written: Vec<PathBuf>) -> Self {
        for file in files_written.iter() {
            if let Err(e) = fs::remove_file(file) {
                error!("Failed to remove file {}: {}", file.display(), e);
            }
        }
        Self { error }
    }
}

impl From<WriteFilesError> for Error {
    fn from(err: WriteFilesError) -> Self {
        err.error
    }
}

/// Solves the problem of trampling by appending a timestamp to the filename
/// Returns the path of the actual file written
/// I am openly looking for a cleaner way to do this.
fn safe_write_auxiliary_file(path: &PathBuf, contents: &str) -> Result<PathBuf, Error> {
    // If the original path doesn't exist, use it directly
    if !path.exists() {
        std::fs::write(path, contents).map_err(|e| Error::new(ErrorDetails::File))
        return Ok(path.clone());
    }

    // Generate timestamp
    let timestamp = current_timestamp();

    // Create new path with timestamp
    let new_path = if let Some(extension) = path.extension() {
        let stem = path.file_stem().unwrap_or_default();
        let parent = path.parent().unwrap_or(std::path::Path::new(""));

        let mut new_name = stem.to_os_string();
        new_name.push(format!("_{}", timestamp));
        new_name.push(".");
        new_name.push(extension);

        parent.join(new_name)
    } else {
        let parent = path.parent().unwrap_or(std::path::Path::new(""));
        let file_name = path.file_name().unwrap_or_default();
        let mut new_name = file_name.to_os_string();
        new_name.push(format!("_{}", timestamp));

        parent.join(new_name)
    };

    // Write to the new path
    std::fs::write(&new_path, contents).map_err(|e| Error::new(ErrorDetails::FileWrite {
        message: e.to_string(),
        file_path: new_path.display().to_string(),
    }))?;
    Ok(new_path)
}
