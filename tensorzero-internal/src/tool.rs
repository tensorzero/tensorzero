use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{Error, ErrorDetails},
    jsonschema_util::{DynamicJSONSchema, JSONSchemaFromPath},
};

/* A Tool is a function that can be called by an LLM
 * We represent them in various ways depending on how they are configured by the user.
 * The primary difficulty is that tools require an input signature that we represent as a JSONSchema.
 * JSONSchema compilation takes time so we want to do it at startup if the tool is in the config.
 * We also don't want to clone compiled JSON schemas.
 * If the tool is dynamic we want to run compilation while LLM inference is happening so that we can validate the tool call arguments.
 *
 * If we are doing an implicit tool call for JSON schema enforcement, we can use the compiled schema from the output signature.
 */

/// A Tool object describes how a tool can be dynamically configured by the user.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    pub description: String,
    pub parameters: Value,
    pub name: String,
    #[serde(default)]
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum ToolConfig {
    Static(Arc<StaticToolConfig>),
    Dynamic(DynamicToolConfig),
    Implicit(ImplicitToolConfig),
    DynamicImplicit(DynamicImplicitToolConfig),
}

/// Contains the configuration information for a specific tool
#[derive(Debug, PartialEq, Serialize)]
pub struct StaticToolConfig {
    pub description: String,
    pub parameters: JSONSchemaFromPath,
    pub name: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool defined at runtime
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct DynamicToolConfig {
    pub description: String,
    pub parameters: DynamicJSONSchema,
    pub name: String,
    pub strict: bool,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImplicitToolConfig {
    pub parameters: JSONSchemaFromPath,
}

/// Contains the configuration information for a tool used in implicit tool calling for
/// JSON schema enforcement for a JSON schema that is dynamically passed at inference time
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynamicImplicitToolConfig {
    pub parameters: DynamicJSONSchema,
}

/// Contains all information required to tell an LLM what tools it can call
/// and what sorts of tool calls (parallel, none, etc) it is allowed to respond with.
/// Most inference providers can convert this into their desired tool format.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ToolCallConfig {
    pub tools_available: Vec<ToolConfig>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
}

/// ToolCallConfigDatabaseInsert is a lightweight version of ToolCallConfig that can be serialized and cloned.
/// It is used to insert the ToolCallConfig into the database.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallConfigDatabaseInsert {
    pub tools_available: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
}

impl ToolCallConfig {
    pub fn new(
        function_tools: &[String],
        function_tool_choice: &ToolChoice,
        function_parallel_tool_calls: Option<bool>,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        dynamic_tool_params: DynamicToolParams,
    ) -> Result<Option<Self>, Error> {
        // If `allowed_tools` is not provided, use the function's configured tools.
        // This means we allow all tools for the function.
        let allowed_tools = dynamic_tool_params
            .allowed_tools
            .as_deref()
            .unwrap_or(function_tools);

        // Get each tool from the static tool config.
        let tools_available: Result<Vec<ToolConfig>, Error> = allowed_tools
            .iter()
            .map(|tool_name| {
                static_tools
                    .get(tool_name)
                    .cloned()
                    .map(ToolConfig::Static)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ToolNotFound {
                            name: tool_name.clone(),
                        })
                    })
            })
            .collect();

        // Throw an error if any tool was not found in the previous step.
        let mut tools_available = tools_available?;

        // Adds the additional tools to the list of available tools
        // (this kicks off async compilation in another thread for each)
        tools_available.extend(dynamic_tool_params.additional_tools.into_iter().flat_map(
            |tools| {
                tools.into_iter().map(|tool| {
                    ToolConfig::Dynamic(DynamicToolConfig {
                        description: tool.description,
                        parameters: DynamicJSONSchema::new(tool.parameters),
                        name: tool.name,
                        strict: tool.strict,
                    })
                })
            },
        ));

        let tool_choice = dynamic_tool_params
            .tool_choice
            .unwrap_or_else(|| function_tool_choice.clone());

        // If the tool choice is a specific tool, make sure it's in the list of available tools
        if let ToolChoice::Specific(tool_name) = &tool_choice {
            if !tools_available.iter().any(|tool| match tool {
                ToolConfig::Static(config) => config.name == *tool_name,
                ToolConfig::Dynamic(config) => config.name == *tool_name,
                ToolConfig::Implicit(_) => false,
                ToolConfig::DynamicImplicit(_) => false,
            }) {
                return Err(ErrorDetails::ToolNotFound {
                    name: tool_name.clone(),
                }
                .into());
            }
        }

        let parallel_tool_calls = dynamic_tool_params
            .parallel_tool_calls
            .or(function_parallel_tool_calls);

        let tool_call_config_option = match tools_available.is_empty() {
            true => None,
            false => Some(Self {
                tools_available,
                tool_choice,
                parallel_tool_calls,
            }),
        };

        Ok(tool_call_config_option)
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolConfig> {
        self.tools_available.iter().find(|tool_cfg| match tool_cfg {
            ToolConfig::Static(config) => config.name == name,
            ToolConfig::Dynamic(config) => config.name == name,
            ToolConfig::Implicit(_config) => false,
            ToolConfig::DynamicImplicit(_config) => false,
        })
    }
}

/// A struct to hold the dynamic tool parameters passed at inference time.
/// These should override the function-level tool parameters.
/// `allowed_tools` should be a subset of the configured tools for the function.
/// if `allowed_tools` is not provided, all tools are allowed.
/// `additional_tools` are the tools that are provided at runtime, which we compile on the fly.
/// `tool_choice` and `parallel_tool_calls` are optional and will override the function-level values.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicToolParams {
    pub allowed_tools: Option<Vec<String>>,
    pub additional_tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, derive(Default))]
pub struct BatchDynamicToolParams {
    pub allowed_tools: Option<Vec<Option<Vec<String>>>>,
    pub additional_tools: Option<Vec<Option<Vec<Tool>>>>,
    pub tool_choice: Option<Vec<Option<ToolChoice>>>,
    pub parallel_tool_calls: Option<Vec<Option<bool>>>,
}

// Helper type for converting BatchDynamicToolParams into a Vec<DynamicToolParams>
pub struct BatchDynamicToolParamsWithSize(pub BatchDynamicToolParams, pub usize);

/// A ToolCall is a request by a model to call a Tool
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// The input format that we accept from the clients.
/// This is like `ToolCallOutput`, but more relaxed (`raw_arguments` and `raw_name` are optional)
/// This allows round-tripping a `ToolCallOutput` without needing to modify the json object,
/// but also allows manually-constructed `ToolCallInput` where users don't care about the
/// name/raw_name distinction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallInput {
    pub name: Option<String>,
    pub arguments: Option<Value>,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_name: Option<String>,
}

impl TryFrom<ToolCallInput> for ToolCall {
    type Error = Error;
    fn try_from(value: ToolCallInput) -> Result<Self, Self::Error> {
        let name = value.name.or(value.raw_name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "ToolCall must have `name` or `raw_name` set".to_string(),
            })
        })?;

        let arguments = if let Some(arguments) = value.arguments {
            match arguments {
                Value::String(s) => {
                    tracing::warn!("Deprecation Warning: Treating string 'ToolCall.arguments' as a serialized JSON object. Please pass in a JSON object instead. Support for strings will be removed in a future release: https://github.com/tensorzero/tensorzero/issues/1410");
                    s
                }
                Value::Object(obj) => Value::Object(obj).to_string(),
                _ => {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: "ToolCall arguments must be a string or an object".to_string(),
                    }));
                }
            }
        } else if let Some(raw_arguments) = value.raw_arguments {
            raw_arguments
        } else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "ToolCall must have `arguments` or `raw_arguments` set".to_string(),
            }));
        };

        Ok(ToolCall {
            name,
            arguments,
            id: value.id,
        })
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<ToolCall, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = ToolCallInput::deserialize(deserializer)?;
        ToolCall::try_from(helper).map_err(serde::de::Error::custom)
    }
}

/// A ToolCallOutput is a request by a model to call a Tool
/// in the form that we return to the client / ClickHouse
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallOutput {
    pub arguments: Option<Value>,
    pub id: String,
    pub name: Option<String>,
    pub raw_arguments: String,
    pub raw_name: String,
}

impl ToolCallOutput {
    /// Validates that a ToolCall is compliant with the ToolCallConfig
    /// First, it finds the ToolConfig for the ToolCall
    /// Then, it validates the ToolCall arguments against the ToolConfig
    pub async fn new(tool_call: ToolCall, tool_cfg: Option<&ToolCallConfig>) -> Self {
        let tool = tool_cfg.and_then(|t| t.get_tool(&tool_call.name));
        let parsed_name = match tool {
            Some(_) => Some(tool_call.name.clone()),
            None => None,
        };
        let parsed_arguments = match &tool {
            Some(tool) => {
                if let Ok(arguments) = serde_json::from_str(&tool_call.arguments) {
                    if tool.validate_arguments(&arguments).await.is_ok() {
                        Some(arguments)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None,
        };
        Self {
            arguments: parsed_arguments,
            id: tool_call.id,
            name: parsed_name,
            raw_arguments: tool_call.arguments.clone(),
            raw_name: tool_call.name.clone(),
        }
    }
}

impl ToolCallConfig {
    #[cfg(test)]
    pub fn implicit_from_value(value: &Value) -> Self {
        let parameters = JSONSchemaFromPath::from_value(value).unwrap();
        let implicit_tool_config = ToolConfig::Implicit(ImplicitToolConfig { parameters });
        Self {
            tools_available: vec![implicit_tool_config],
            tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
            parallel_tool_calls: None,
        }
    }
}

/// A ToolResult is the outcome of a ToolCall, which we may want to present back to the model
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub name: String,
    pub result: String,
    pub id: String,
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
    // Forces the LLM to call a specific tool. The String is the name of the tool.
    Specific(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: String,
    pub raw_name: String,
    pub raw_arguments: String,
}

pub const IMPLICIT_TOOL_NAME: &str = "respond";
pub const IMPLICIT_TOOL_DESCRIPTION: &str = "Respond to the user using the output schema provided.";

impl ToolConfig {
    pub async fn validate_arguments(&self, arguments: &Value) -> Result<(), Error> {
        match self {
            ToolConfig::Static(config) => config.parameters.validate(arguments),
            ToolConfig::Dynamic(config) => config.parameters.validate(arguments).await,
            ToolConfig::Implicit(config) => config.parameters.validate(arguments),
            ToolConfig::DynamicImplicit(config) => config.parameters.validate(arguments).await,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.description,
            ToolConfig::Dynamic(config) => &config.description,
            ToolConfig::Implicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
            ToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_DESCRIPTION,
        }
    }

    pub fn parameters(&self) -> &Value {
        match self {
            ToolConfig::Static(config) => config.parameters.value,
            ToolConfig::Dynamic(config) => &config.parameters.value,
            ToolConfig::Implicit(config) => config.parameters.value,
            ToolConfig::DynamicImplicit(config) => &config.parameters.value,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ToolConfig::Static(config) => &config.name,
            ToolConfig::Dynamic(config) => &config.name,
            ToolConfig::Implicit(_config) => IMPLICIT_TOOL_NAME,
            ToolConfig::DynamicImplicit(_config) => IMPLICIT_TOOL_NAME,
        }
    }

    pub fn strict(&self) -> bool {
        match self {
            ToolConfig::Static(config) => config.strict,
            ToolConfig::Dynamic(config) => config.strict,
            ToolConfig::Implicit(_config) => false,
            ToolConfig::DynamicImplicit(_config) => false,
        }
    }
}

impl From<ToolCallConfig> for ToolCallConfigDatabaseInsert {
    fn from(tool_call_config: ToolCallConfig) -> Self {
        Self {
            tools_available: tool_call_config
                .tools_available
                .into_iter()
                .map(|tool| tool.into())
                .collect(),
            tool_choice: tool_call_config.tool_choice,
            parallel_tool_calls: tool_call_config.parallel_tool_calls,
        }
    }
}

impl From<ToolConfig> for Tool {
    fn from(tool_config: ToolConfig) -> Self {
        Self {
            description: tool_config.description().to_string(),
            parameters: tool_config.parameters().clone(),
            name: tool_config.name().to_string(),
            strict: tool_config.strict(),
        }
    }
}

pub fn create_dynamic_implicit_tool_config(schema: Value) -> ToolCallConfig {
    let tool_schema = DynamicJSONSchema::new(schema);
    let implicit_tool = ToolConfig::DynamicImplicit(DynamicImplicitToolConfig {
        parameters: tool_schema,
    });
    ToolCallConfig {
        tools_available: vec![implicit_tool],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        parallel_tool_calls: None,
    }
}

impl TryFrom<BatchDynamicToolParamsWithSize> for Vec<DynamicToolParams> {
    type Error = Error;

    fn try_from(
        batch_dynamic_tool_params_with_size: BatchDynamicToolParamsWithSize,
    ) -> Result<Self, Self::Error> {
        let BatchDynamicToolParamsWithSize(batch_dynamic_tool_params, num_inferences) =
            batch_dynamic_tool_params_with_size;
        if num_inferences == 0 {
            return Ok(vec![
                DynamicToolParams {
                    allowed_tools: None,
                    additional_tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                };
                num_inferences
            ]);
        }
        let BatchDynamicToolParams {
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
        } = batch_dynamic_tool_params;

        // Verify all provided Vecs have the same length
        if let Some(allowed_tools) = &allowed_tools {
            if allowed_tools.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "allowed_tools vector length ({}) does not match number of inferences ({})",
                        allowed_tools.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(additional_tools) = &additional_tools {
            if additional_tools.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "additional_tools vector length ({}) does not match number of inferences ({})",
                        additional_tools.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(tool_choice) = &tool_choice {
            if tool_choice.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "tool_choice vector length ({}) does not match number of inferences ({})",
                        tool_choice.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        if let Some(parallel_tool_calls) = &parallel_tool_calls {
            if parallel_tool_calls.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "parallel_tool_calls vector length ({}) does not match number of inferences ({})",
                        parallel_tool_calls.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }
        // Convert Option<Vec<Option<T>>> into Vec<Option<T>> by unwrapping or creating empty vec
        let allowed_tools = allowed_tools.unwrap_or_default();
        let additional_tools = additional_tools.unwrap_or_default();
        let tool_choice = tool_choice.unwrap_or_default();
        let parallel_tool_calls = parallel_tool_calls.unwrap_or_default();

        // Create iterators that take ownership
        let mut allowed_tools_iter = allowed_tools.into_iter();
        let mut additional_tools_iter = additional_tools.into_iter();
        let mut tool_choice_iter = tool_choice.into_iter();
        let mut parallel_tool_calls_iter = parallel_tool_calls.into_iter();

        // Build params using the iterators
        let mut all_dynamic_tool_params = Vec::with_capacity(num_inferences);
        // Since we already verified that the vectors that were Some were the same length,
        // it is safe to do .next().unwrap_or() since we'll either be taking real elements or using an empty vector.
        for _ in 0..num_inferences {
            all_dynamic_tool_params.push(DynamicToolParams {
                allowed_tools: allowed_tools_iter.next().unwrap_or(None),
                additional_tools: additional_tools_iter.next().unwrap_or(None),
                tool_choice: tool_choice_iter.next().unwrap_or(None),
                parallel_tool_calls: parallel_tool_calls_iter.next().unwrap_or(None),
            });
        }
        Ok(all_dynamic_tool_params)
    }
}

impl From<ToolCallConfigDatabaseInsert> for ToolCallConfig {
    fn from(db_insert: ToolCallConfigDatabaseInsert) -> Self {
        Self {
            tools_available: db_insert
                .tools_available
                .into_iter()
                .map(|tool| {
                    ToolConfig::Dynamic(DynamicToolConfig {
                        description: tool.description,
                        parameters: DynamicJSONSchema::new(tool.parameters),
                        name: tool.name,
                        strict: tool.strict,
                    })
                })
                .collect(),
            tool_choice: db_insert.tool_choice,
            parallel_tool_calls: db_insert.parallel_tool_calls,
        }
    }
}

/// For use in initializing JSON functions
/// Creates a ToolCallConfig with a single implicit tool that takes the schema as arguments
pub fn create_implicit_tool_call_config(schema: JSONSchemaFromPath) -> ToolCallConfig {
    let implicit_tool = ToolConfig::Implicit(ImplicitToolConfig { parameters: schema });
    ToolCallConfig {
        tools_available: vec![implicit_tool],
        tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
        parallel_tool_calls: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;
    use serde_json::json;
    use tracing_test::traced_test;

    lazy_static! {
        static ref TOOLS: HashMap<String, Arc<StaticToolConfig>> = {
            let mut map = HashMap::new();
            map.insert(
                "get_temperature".to_string(),
                Arc::new(StaticToolConfig {
                    name: "get_temperature".to_string(),
                    description: "Get the current temperature in a given location".to_string(),
                    #[allow(clippy::expect_used)]
                    parameters: JSONSchemaFromPath::from_value(&json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                        "required": ["location"]
                    }))
                    .expect("Failed to create schema for get_temperature"),
                    strict: true,
                }),
            );
            map.insert(
                "query_articles".to_string(),
                Arc::new(StaticToolConfig {
                    name: "query_articles".to_string(),
                    description: "Query articles from a database based on given criteria"
                        .to_string(),
                    #[allow(clippy::expect_used)]
                    parameters: JSONSchemaFromPath::from_value(&json!({
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                        },
                        "required": ["keyword"]
                    }))
                    .expect("Failed to create schema for query_articles"),
                    strict: false,
                }),
            );
            map
        };
        static ref EMPTY_TOOLS: HashMap<String, Arc<StaticToolConfig>> = HashMap::new();
        static ref EMPTY_FUNCTION_TOOLS: Vec<String> = vec![];
        static ref ALL_FUNCTION_TOOLS: Vec<String> =
            vec!["get_temperature".to_string(), "query_articles".to_string()];
        static ref AUTO_TOOL_CHOICE: ToolChoice = ToolChoice::Auto;
        static ref WEATHER_TOOL_CHOICE: ToolChoice =
            ToolChoice::Specific("get_temperature".to_string());
    }

    #[tokio::test]
    async fn test_tool_call_config_new() {
        // Empty tools in function, no dynamic tools, tools are configured in the config
        // This should return no tools because the function does not specify any tools
        let tool_call_config = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap();
        assert!(tool_call_config.is_none());

        // All tools available, no dynamic tools, tools are configured in the config
        // This should return all tools because the function specifies all tools
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        assert_eq!(tool_call_config.tool_choice, ToolChoice::Auto);
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        assert!(tool_call_config.tools_available[0].strict());
        assert!(!tool_call_config.tools_available[1].strict());

        // Empty tools in function and config but we specify an allowed tool (should fail)
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            ..Default::default()
        };
        let err = ToolCallConfig::new(
            &EMPTY_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &EMPTY_TOOLS,
            dynamic_tool_params,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "get_temperature".to_string()
            }
            .into()
        );

        // Dynamic tool config specifies a particular tool to call and it's in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("get_temperature".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("get_temperature".to_string())
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));

        // Dynamic tool config specifies a particular tool to call and it's not in the function tools list
        let dynamic_tool_params = DynamicToolParams {
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let err = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::ToolNotFound {
                name: "establish_campground".to_string()
            }
            .into()
        );

        // We pass an empty list of allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 1);
        let first_tool = tool_call_config.tools_available.first().unwrap();
        assert_eq!(first_tool.name(), "establish_campground");
        assert!(!first_tool.strict());

        // We pass a list of a single allowed tool and then configure a new tool
        // This should remove the other configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec!["get_temperature".to_string()]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 2);
        // The following code depends on an implementation detail for this ordering,
        // might break if we change the order
        assert_eq!(
            tool_call_config.tools_available[0].name(),
            "get_temperature"
        );
        assert_eq!(
            tool_call_config.tools_available[1].name(),
            "establish_campground"
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(false));

        // We pass a list of no allowed tools and then configure a new tool
        // This should remove all configured tools and add the new tool
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: Some(vec![]),
            additional_tools: Some(vec![Tool {
                name: "establish_campground".to_string(),
                description: "Establish a campground".to_string(),
                parameters: json!({}),
                strict: false,
            }]),
            tool_choice: Some(ToolChoice::Specific("establish_campground".to_string())),
            ..Default::default()
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            dynamic_tool_params,
        )
        .unwrap()
        .unwrap();
        assert_eq!(tool_call_config.tools_available.len(), 1);
        assert_eq!(
            tool_call_config.tools_available[0].name(),
            "establish_campground"
        );
        assert_eq!(tool_call_config.parallel_tool_calls, Some(true));
        assert_eq!(
            tool_call_config.tool_choice,
            ToolChoice::Specific("establish_campground".to_string())
        );
        assert!(!tool_call_config.tools_available[0].strict());
    }

    #[tokio::test]
    async fn test_tool_call_output_new() {
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "123".to_string(),
        };
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams::default(),
        )
        .unwrap()
        .unwrap();
        // Tool call is valid, so we should get a valid ToolCallOutput
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.raw_name, "get_temperature");
        assert_eq!(
            tool_call_output.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(tool_call_output.id, "123");
        assert_eq!(tool_call_output.name, Some("get_temperature".to_string()));
        assert_eq!(
            tool_call_output.arguments,
            Some(json!({
                "location": "San Francisco",
                "unit": "celsius"
            }))
        );

        // Bad arguments, but valid name (parsed_name is set but parsed_arguments is not)
        let tool_call = ToolCall {
            name: "get_temperature".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.name, Some("get_temperature".to_string()));
        assert_eq!(tool_call_output.arguments, None);
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(tool_call_output.raw_name, "get_temperature");
        assert_eq!(
            tool_call_output.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"kelvin\"}"
        );

        // Bad name, good arguments (both not set since the name is invalid and we can't be sure what tool this goes to)
        let tool_call = ToolCall {
            name: "not_get_weather".to_string(),
            arguments: "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.name, None);
        assert_eq!(tool_call_output.arguments, None);
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(tool_call_output.raw_name, "not_get_weather");
        assert_eq!(
            tool_call_output.raw_arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );

        // Make sure validation works with dynamic tools
        let tool_call_config = ToolCallConfig::new(
            &ALL_FUNCTION_TOOLS,
            &AUTO_TOOL_CHOICE,
            Some(true),
            &TOOLS,
            DynamicToolParams {
                additional_tools: Some(vec![Tool {
                    name: "establish_campground".to_string(),
                    description: "Establish a campground".to_string(),
                    parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                    strict: false,
                }]),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        let tool_call = ToolCall {
            name: "establish_campground".to_string(),
            arguments: "{\"location\": \"Lucky Dog\"}".to_string(),
            id: "321".to_string(),
        };
        let tool_call_output = ToolCallOutput::new(tool_call, Some(&tool_call_config)).await;
        assert_eq!(tool_call_output.raw_name, "establish_campground");
        assert_eq!(
            tool_call_output.raw_arguments,
            "{\"location\": \"Lucky Dog\"}"
        );
        assert_eq!(tool_call_output.id, "321");
        assert_eq!(
            tool_call_output.name,
            Some("establish_campground".to_string())
        );
        assert_eq!(
            tool_call_output.arguments,
            Some(json!({"location": "Lucky Dog"}))
        );
    }

    #[test]
    fn test_tool_call_deserialize_plain_raw() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "raw_name": "should have ignored raw name",
            "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}",
            "raw_arguments": "should have ignored raw arguments",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(
            tool_call.arguments,
            "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
        );
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_raw_only() {
        let tool_call = serde_json::json!({
            "raw_name": "get_temperature",
            "raw_arguments": "my raw arguments",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "my raw arguments");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_object() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": {"my": "arguments"},
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\":\"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_arguments_string() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        let tool_call: ToolCall = serde_json::from_value(tool_call).unwrap();
        assert_eq!(tool_call.name, "get_temperature");
        assert_eq!(tool_call.arguments, "{\"my\": \"arguments\"}");
        assert_eq!(tool_call.id, "123");
    }

    #[test]
    fn test_tool_call_deserialize_missing_name() {
        let tool_call = serde_json::json!({
            "arguments": "{\"my\": \"arguments\"}",
            "id": "123"
        });
        let err_msg = serde_json::from_value::<ToolCall>(tool_call)
            .unwrap_err()
            .to_string();
        assert_eq!(err_msg, "ToolCall must have `name` or `raw_name` set");
    }

    #[test]
    fn test_tool_call_deserialize_missing_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123"
        });
        let err_msg = serde_json::from_value::<ToolCall>(tool_call)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err_msg,
            "ToolCall must have `arguments` or `raw_arguments` set"
        );
    }

    #[test]
    #[traced_test]
    fn test_tool_call_deserialize_deprecated_arguments() {
        let tool_call = serde_json::json!({
            "name": "get_temperature",
            "id": "123",
            "arguments": "My string arguments"
        });
        serde_json::from_value::<ToolCall>(tool_call).unwrap();
        assert!(logs_contain("Deprecation Warning: Treating string 'ToolCall.arguments' as a serialized JSON object."))
    }
}
