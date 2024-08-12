use std::path::Path;

use serde::Deserialize;

use crate::{error::Error, jsonschema_util::JSONSchemaFromPath};

#[derive(Clone, Debug, Deserialize)]
pub struct ToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
}

impl ToolConfig {
    pub fn load<P: AsRef<Path>>(&mut self, base_path: P) -> Result<(), Error> {
        self.parameters.load(base_path)
    }
}
