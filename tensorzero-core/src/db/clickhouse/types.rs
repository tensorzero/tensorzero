use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, deserialize_from_pyobj, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::Text;
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedInput},
    serde_util::{deserialize_defaulted_string_or_parsed_json, deserialize_string_or_parsed_json},
    tool::ToolCallConfigDatabaseInsert,
    variant::{chat_completion::prepare_model_input, VariantConfig},
};
#[cfg(feature = "pyo3")]
use pyo3::types::{PyAny, PyList};
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*, IntoPyObjectExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Represents an stored inference to be used for optimization.
/// These are retrieved from the database in this format.
/// NOTE / TODO: As an incremental step we are deserializing this enum from Python.
/// in the final version we should instead make this a native PyO3 class and
/// avoid deserialization entirely unless given a dict.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub enum StoredInference {
    Chat(StoredChatInference),
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInference {
    #[new]
    #[expect(clippy::too_many_arguments)]
    pub fn new<'py>(
        py: Python<'py>,
        r#type: String,
        function_name: String,
        variant_name: String,
        input: Bound<'py, PyAny>,
        output: Bound<'py, PyAny>,
        episode_id: Bound<'py, PyAny>,
        inference_id: Bound<'py, PyAny>,
        tool_params: Option<Bound<'py, PyAny>>,
        output_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let input: ResolvedInput = deserialize_from_pyobj(py, &input)?;
        let episode_id: Uuid = deserialize_from_pyobj(py, &episode_id)?;
        let inference_id: Uuid = deserialize_from_pyobj(py, &inference_id)?;
        match r#type.as_str() {
            "chat" => {
                let output: Vec<ContentBlockChatOutput> = deserialize_from_pyobj(py, &output)?;
                let Some(tool_params) = tool_params.map(|x| deserialize_from_pyobj(py, &x)) else {
                    return Err(PyValueError::new_err(
                        "tool_params is required for chat inferences",
                    ));
                };
                let tool_params: ToolCallConfigDatabaseInsert = tool_params?;
                Ok(Self::Chat(StoredChatInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    episode_id,
                    inference_id,
                    tool_params,
                }))
            }
            "json" => {
                let output: JsonInferenceOutput = deserialize_from_pyobj(py, &output)?;
                let Some(output_schema) = output_schema.map(|x| deserialize_from_pyobj(py, &x))
                else {
                    return Err(PyValueError::new_err(
                        "output_schema is required for json inferences",
                    ));
                };
                let output_schema: Value = output_schema?;
                Ok(Self::Json(StoredJsonInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    episode_id,
                    inference_id,
                    output_schema,
                }))
            }
            _ => Err(PyValueError::new_err(format!("Invalid type: {type}"))),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_function_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.function_name.clone(),
            StoredInference::Json(example) => example.function_name.clone(),
        }
    }

    #[getter]
    pub fn get_variant_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.variant_name.clone(),
            StoredInference::Json(example) => example.variant_name.clone(),
        }
    }

    #[getter]
    pub fn get_input(&self) -> ResolvedInput {
        match self {
            StoredInference::Chat(example) => example.input.clone(),
            StoredInference::Json(example) => example.input.clone(),
        }
    }

    /// Returns the output of the inference as PyO3 classes.
    /// This is actually a List of ContentBlockChatOutputs for StoredChatInference
    /// and a JsonInferenceOutput for StoredJsonInference.
    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => example
                .output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?
                .into_py_any(py)?
                .into_bound(py),
            StoredInference::Json(example) => {
                example.output.clone().into_py_any(py)?.into_bound(py)
            }
        })
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.episode_id),
            StoredInference::Json(example) => uuid_to_python(py, example.episode_id),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.inference_id),
            StoredInference::Json(example) => uuid_to_python(py, example.inference_id),
        }
    }

    #[getter]
    pub fn get_tool_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => {
                example.tool_params.clone().into_py_any(py)?.into_bound(py)
            }
            // Json inferences don't have tool params
            StoredInference::Json(_) => py.None().into_bound(py),
        })
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(_) => py.None().into_bound(py),
            StoredInference::Json(example) => {
                serialize_to_dict(py, example.output_schema.clone())?.into_bound(py)
            }
        })
    }

    #[getter]
    pub fn get_type(&self) -> String {
        match self {
            StoredInference::Chat(_) => "chat".to_string(),
            StoredInference::Json(_) => "json".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output: Vec<ContentBlockChatOutput>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(deserialize_with = "deserialize_defaulted_string_or_parsed_json")]
    pub tool_params: ToolCallConfigDatabaseInsert,
}

impl std::fmt::Display for StoredChatInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredChatInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output: JsonInferenceOutput,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: Value,
}

impl std::fmt::Display for StoredJsonInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredJsonInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl StoredInference {
    pub fn input_mut(&mut self) -> &mut ResolvedInput {
        match self {
            StoredInference::Chat(example) => &mut example.input,
            StoredInference::Json(example) => &mut example.input,
        }
    }
    pub fn input(&self) -> &ResolvedInput {
        match self {
            StoredInference::Chat(example) => &example.input,
            StoredInference::Json(example) => &example.input,
        }
    }

    pub fn function_name(&self) -> &str {
        match self {
            StoredInference::Chat(example) => &example.function_name,
            StoredInference::Json(example) => &example.function_name,
        }
    }
}

/// Represents an inference that has been prepared for fine-tuning.
/// This is constructed by rendering a StoredInference with a variant for messages
/// and by resolving all network resources (e.g. images).
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[derive(Debug, PartialEq, Serialize)]
pub struct RenderedStoredInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: ModelInput,
    pub output: Vec<ContentBlockChatOutput>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RenderedStoredInference {
    #[getter]
    pub fn get_function_name(&self) -> &str {
        &self.function_name
    }

    #[getter]
    pub fn get_variant_name(&self) -> &str {
        &self.variant_name
    }

    #[getter]
    pub fn get_input(&self) -> ModelInput {
        self.input.clone()
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let output = self
            .output
            .iter()
            .map(|x| content_block_chat_output_to_python(py, x.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, output).map(|list| list.into_any())
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.episode_id)
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.inference_id)
    }

    #[getter]
    pub fn get_tool_params(&self) -> Option<ToolCallConfigDatabaseInsert> {
        self.tool_params.clone()
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, self.output_schema.clone()).map(|x| x.into_bound(py))
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for RenderedStoredInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Serialize the rendered inference to pretty-printed JSON
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Convert a StoredInference's input to a ModelInput.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the stored inference is being rendered.
/// This does not handle resolving network resources (e.g. images).
fn render_model_input(
    inference_example: &StoredInference,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<ModelInput, Error> {
    let variant_name = variants
        .get(inference_example.function_name())
        .ok_or_else(|| {
            Error::new(ErrorDetails::MissingFunctionInVariants {
                function_name: inference_example.function_name().to_string(),
            })
        })?;
    let function_config = config.get_function(inference_example.function_name())?;
    let variant_config = function_config
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownVariant {
                name: variant_name.clone(),
            })
        })?;
    let VariantConfig::ChatCompletion(chat_completion_config) = &variant_config.inner else {
        return Err(Error::new(ErrorDetails::InvalidVariantForOptimization {
            function_name: inference_example.function_name().to_string(),
            variant_name: variant_name.clone(),
        }));
    };
    let system_template_name = chat_completion_config
        .system_template
        .as_ref()
        .map(|x| {
            x.path
                .to_str()
                .ok_or_else(|| Error::new(ErrorDetails::InvalidTemplatePath))
        })
        .transpose()?;
    let user_template_name = chat_completion_config
        .user_template
        .as_ref()
        .map(|x| {
            x.path
                .to_str()
                .ok_or_else(|| Error::new(ErrorDetails::InvalidTemplatePath))
        })
        .transpose()?;
    let assistant_template_name = chat_completion_config
        .assistant_template
        .as_ref()
        .map(|x| {
            x.path
                .to_str()
                .ok_or_else(|| Error::new(ErrorDetails::InvalidTemplatePath))
        })
        .transpose()?;
    prepare_model_input(
        inference_example.input().system.as_ref(),
        &inference_example.input().messages,
        &config.templates,
        system_template_name,
        user_template_name,
        assistant_template_name,
        function_config.template_schema_info(),
    )
}

/// Render a StoredInference to a RenderedStoredInference.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the inference example is being rendered.
///
/// This does not handle resolving network resources (e.g. images).
pub fn render_stored_inference(
    inference_example: StoredInference,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<RenderedStoredInference, Error> {
    let model_input = render_model_input(&inference_example, config, variants)?;
    match inference_example {
        StoredInference::Chat(example) => Ok(RenderedStoredInference {
            function_name: example.function_name,
            variant_name: example.variant_name,
            input: model_input,
            output: example.output,
            episode_id: example.episode_id,
            inference_id: example.inference_id,
            tool_params: Some(example.tool_params),
            output_schema: None,
        }),
        StoredInference::Json(example) => {
            let output: Vec<ContentBlockChatOutput> = match example.output.raw {
                Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
                None => vec![],
            };
            Ok(RenderedStoredInference {
                function_name: example.function_name,
                variant_name: example.variant_name,
                input: model_input,
                output,
                episode_id: example.episode_id,
                inference_id: example.inference_id,
                tool_params: None,
                output_schema: Some(example.output_schema),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stored_inference_deserialization_chat() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "chat",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "[{\"type\": \"text\", \"text\": \"Hello! How can I help you today?\"}]",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tool_params": ""
            }
        "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );

        // Test the Python version (singly serialized)
        let json = r#"
        {
            "type": "chat",
            "function_name": "test_function",
            "variant_name": "test_variant",
            "input": {"system": "you are a helpful assistant", "messages": []},
            "output": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "episode_id": "123e4567-e89b-12d3-a456-426614174000",
            "inference_id": "123e4567-e89b-12d3-a456-426614174000",
            "tool_params": ""
        }
    "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );
    }

    #[test]
    fn test_stored_inference_deserialization_json() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "json",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "{\"raw\":\"{\\\"answer\\\":\\\"Goodbye\\\"}\",\"parsed\":{\"answer\":\"Goodbye\"}}",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "output_schema": "{\"type\": \"object\", \"properties\": {\"output\": {\"type\": \"string\"}}}"
            }
        "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );

        // Test the Python version (singly serialized)
        let json = r#"
         {
             "type": "json",
             "function_name": "test_function",
             "variant_name": "test_variant",
             "input": {"system": "you are a helpful assistant", "messages": []},
             "output": {"raw":"{\"answer\":\"Goodbye\"}","parsed":{"answer":"Goodbye"}},
             "episode_id": "123e4567-e89b-12d3-a456-426614174000",
             "inference_id": "123e4567-e89b-12d3-a456-426614174000",
             "output_schema": {"type": "object", "properties": {"output": {"type": "string"}}}
         }
     "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );
    }
}
