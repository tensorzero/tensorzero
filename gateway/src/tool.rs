use crate::jsonschema_util::JSONSchemaFromPath;

#[derive(Debug)]
pub struct ToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
}
