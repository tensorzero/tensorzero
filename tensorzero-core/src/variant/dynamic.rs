use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tempfile::tempdir;

use crate::config_parser::UninitializedVariantInfo;
use crate::error::{Error, ErrorDetails};

use super::VariantInfo;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DynamicVariantParams {
    pub config: UninitializedVariantInfo,
    // Since Variant configuration may require some files to be
    // read from the file sytem, we allow the caller to pass in a map
    // from "file names" to "file contents" here.
    pub paths: HashMap<String, String>,
}

impl TryFrom<(UninitializedVariantInfo, &HashMap<String, String>)> for VariantInfo {
    type Error = Error;

    fn try_from(
        (config, paths): (UninitializedVariantInfo, &HashMap<String, String>),
    ) -> Result<Self, Self::Error> {
        // Since we do all our validation and generation for variants from the file system,
        // we write the contents of the files to the temporary directory and construct
        // VariantInfo from that.
        let tmp_dir = tempdir().map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to create temporary directory for dynamic variant: {e}",),
            })
        })?;

        for (path, content) in paths {
            let file_path = tmp_dir.path().join(path);
            std::fs::write(&file_path, content).map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Failed to write file {} for dynamic variant: {e}",
                        file_path.display(),
                    ),
                })
            })?;
        }

        // Next, we construct the VariantInfo from the temporary directory.
        let mut variant_info = config.load(tmp_dir)?;
        variant_info.set_weight(Some(1.0));

        Ok(variant_info)
    }
}
