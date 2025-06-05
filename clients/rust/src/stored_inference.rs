use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyAny, PyList};
use serde::Deserialize;
use serde_json::Value;
#[cfg(feature = "pyo3")]
use tensorzero_internal::inference::types::pyo3_helpers::{
    content_block_output_to_python, serialize_to_dict, uuid_to_python,
};
use tensorzero_internal::{
    config_parser::Config,
    error::{Error, ErrorDetails},
    inference::types::{
        ContentBlockChatOutput, ContentBlockOutput, JsonInferenceOutput, ModelInput, ResolvedInput,
    },
    tool::ToolCallConfigDatabaseInsert,
    variant::{chat_completion::prepare_model_input, VariantConfig},
};
use uuid::Uuid;

/// Represents an stored inference to be used for optimization.
/// These are retrieved from the database in this format.
/// NOTE / TODO: As an incremental step we are deserializing this enum from Python.
/// in the final version we should instead make this a native PyO3 class and
/// avoid deserialization entirely unless given a dict.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredInference {
    Chat(StoredChatInference),
    Json(StoredJsonInference),
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: ResolvedInput,
    pub output: Vec<ContentBlockChatOutput>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub tool_params: ToolCallConfigDatabaseInsert,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: ResolvedInput,
    pub output: JsonInferenceOutput,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub output_schema: Value,
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
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, PartialEq)]
pub struct RenderedStoredInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: ModelInput,
    pub output: Vec<ContentBlockOutput>,
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
            .map(|x| content_block_output_to_python(py, x))
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
    let VariantConfig::ChatCompletion(chat_completion_config) = variant_config else {
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
            output: example.output.into_iter().map(|x| x.into()).collect(),
            episode_id: example.episode_id,
            inference_id: example.inference_id,
            tool_params: Some(example.tool_params),
            output_schema: None,
        }),
        StoredInference::Json(example) => {
            let output: Vec<ContentBlockOutput> = match example.output.raw {
                Some(raw) => vec![raw.into()],
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
