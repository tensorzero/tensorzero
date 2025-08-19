use std::path::PathBuf;

use crate::{
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    variant::VariantConfig,
};

use super::{Config, ConfigFileGlob};

pub async fn write_variant_config(
    variant: &VariantConfig,
    new_variant_name: &str,
    existing_function_name: &str,
    config_glob: &ConfigFileGlob,
    config: &Config,
) -> Result<(), Error> {
    // First, we'll figure out what file to write to
    let mut toml_write_path = None;
    for path in config_glob.paths.iter() {
        if check_file_for_function_type_key(existing_function_name, path).await? {
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

    todo!()
}

async fn check_file_for_function_type_key(
    existing_function_name: &str,
    path: &PathBuf,
) -> Result<bool, Error> {
    todo!()
}

async fn write_variant()
