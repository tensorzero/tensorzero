use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::Error;
use crate::jsonschema_util::JSONSchemaFromPath;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfig {
    pub r#type: FunctionConfigType,
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub chat: Option<FunctionConfigChat>,
    pub tool: Option<FunctionConfigTool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionConfigType {
    Chat,
    Tool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FunctionConfigChat {
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FunctionConfigTool {
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariantConfig {
    pub weight: f64,
    pub generation: Option<GenerationConfig>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    pub model: String,
    pub system_template: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputMessage {
    role: InputMessageRole,
    content: serde_json::Value,
}

impl FunctionConfig {
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a tool function, the input is validated against the system, user, and assistant schemas.
    pub fn validate_input(&self, input: &Vec<InputMessage>) -> Result<(), Error> {
        match self.r#type {
            FunctionConfigType::Chat => {
                if let Some(ref chat) = self.chat {
                    FunctionConfig::validate_chat_like_input(
                        &chat.system_schema,
                        &chat.user_schema,
                        &chat.assistant_schema,
                        input,
                    )?;
                }
            }
            FunctionConfigType::Tool => {
                if let Some(ref tool) = self.tool {
                    FunctionConfig::validate_chat_like_input(
                        &tool.system_schema,
                        &tool.user_schema,
                        &tool.assistant_schema,
                        input,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Validate an input that is a chat-like function (i.e. chat or tool).
    /// The validation is done based on the input's role and the function's schemas.
    fn validate_chat_like_input(
        system_schema: &Option<JSONSchemaFromPath>,
        user_schema: &Option<JSONSchemaFromPath>,
        assistant_schema: &Option<JSONSchemaFromPath>,
        input: &Vec<InputMessage>,
    ) -> Result<(), Error> {
        for message in input {
            match (
                &message.role,
                &system_schema,
                &user_schema,
                &assistant_schema,
            ) {
                (InputMessageRole::System, Some(ref system_schema), _, _) => {
                    system_schema.validate(&message.content)
                }
                (InputMessageRole::User, _, Some(ref user_schema), _) => {
                    user_schema.validate(&message.content)
                }
                (InputMessageRole::Assistant, _, _, Some(ref assistant_schema)) => {
                    assistant_schema.validate(&message.content)
                }
                _ => Ok(()),
            }
            .map_err(|e| Error::InvalidInputSchema {
                messages: e
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>(),
            })?;
        }

        Ok(())
    }
}
