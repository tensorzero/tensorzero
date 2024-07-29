use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use crate::error::Error;
use crate::jsonschema_util::JSONSchemaFromPath;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Tool(FunctionConfigTool),
}

impl FunctionConfig {
    pub fn variants(&self) -> &HashMap<String, VariantConfig> {
        match self {
            FunctionConfig::Chat(params) => &params.variants,
            FunctionConfig::Tool(params) => &params.variants,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigTool {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(ChatCompletionConfig),
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
        }
    }

    // TODO: return a reference to the template itself (not the path)
    pub fn system_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.system_template.as_ref(),
        }
    }

    // TODO: return a reference to the template itself (not the path)
    pub fn user_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.user_template.as_ref(),
        }
    }

    // TODO: return a reference to the template itself (not the path)
    pub fn assistant_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.assistant_template.as_ref(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String,
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
}

impl fmt::Display for InputMessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    role: InputMessageRole,
    content: serde_json::Value,
}

impl FunctionConfig {
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a tool function, the input is validated against the system, user, and assistant schemas.
    pub fn validate_input(&self, input: &[InputMessage]) -> Result<(), Error> {
        match &self {
            FunctionConfig::Chat(params) => {
                FunctionConfig::validate_chat_like_input(
                    &params.system_schema,
                    &params.user_schema,
                    &params.assistant_schema,
                    input,
                )?;
            }
            FunctionConfig::Tool(params) => {
                FunctionConfig::validate_chat_like_input(
                    &params.system_schema,
                    &params.user_schema,
                    &params.assistant_schema,
                    input,
                )?;
            }
        }

        Ok(())
    }

    // TODO: implement output validation against the output schema

    /// Validate an input that is a chat-like function (i.e. chat or tool).
    /// The validation is done based on the input's role and the function's schemas.
    fn validate_chat_like_input(
        system_schema: &Option<JSONSchemaFromPath>,
        user_schema: &Option<JSONSchemaFromPath>,
        assistant_schema: &Option<JSONSchemaFromPath>,
        input: &[InputMessage],
    ) -> Result<(), Error> {
        for (index, message) in input.iter().enumerate() {
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
                _ => {
                    if !message.content.is_string() {
                        return Err(Error::InvalidMessage {
                            message: format!("Message at index {} has non-string content but there is no schema given for role {}.", index, message.role),
                        });
                    } else {
                        Ok(())
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_schema() -> JSONSchemaFromPath {
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        }
        "#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", schema).expect("Failed to write schema to temporary file");

        JSONSchemaFromPath::new(temp_file.path()).expect("Failed to create JSONSchemaFromPath")
    }

    #[test]
    fn test_validate_input_chat_no_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no schema given for role \"system\".".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_chat_system_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(create_test_schema()),
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_user_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(create_test_schema()),
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_assistant_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(create_test_schema()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_all_schemas() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(create_test_schema()),
            user_schema: Some(create_test_schema()),
            assistant_schema: Some(create_test_schema()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_no_schema() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system_content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 1 has non-string content but there is no schema given for role \"user\".".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_tool_system_schema() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: Some(create_test_schema()),
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_user_schema() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(create_test_schema()),
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_assistant_schema() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(create_test_schema()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_all_schemas() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: Some(create_test_schema()),
            user_schema: Some(create_test_schema()),
            assistant_schema: Some(create_test_schema()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidInputSchema {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()]
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }
}
