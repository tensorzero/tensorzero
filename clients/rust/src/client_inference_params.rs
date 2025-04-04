use std::{collections::HashMap, ops::Deref};

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_internal::{
    cache::CacheParamsOptions,
    endpoints::inference::{InferenceParams, Params},
    error::Error,
    inference::types::{
        extra_body::UnfilteredInferenceExtraBody, Input, InputMessage, InputMessageContent,
    },
    tool::DynamicToolParams,
};
use uuid::Uuid;

use crate::client_input::{test_client_input_to_input, ClientInput};

// This is a copy-paste of the `Params` struct from `tensorzero_internal::endpoints::inference::Params`.
// with just the `credentials` field adjusted to allow serialization.
/// The expected payload is a JSON object with the following fields:
#[derive(Clone, Debug, Serialize, Default)]
pub struct ClientInferenceParams {
    // The function name. Exactly one of `function_name` or `model_name` must be provided.
    pub function_name: Option<String>,
    // The model name to run using a default function. Exactly one of `function_name` or `model_name` must be provided.
    pub model_name: Option<String>,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    pub episode_id: Option<Uuid>,
    // the input for the inference
    pub input: ClientInput,
    // default False
    pub stream: Option<bool>,
    // Inference-time overrides for variant types (use with caution)
    pub params: InferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // if true, the inference will not be stored
    pub dryrun: Option<bool>,
    // if true, the inference will be internal and validation of tags will be skipped
    pub internal: bool,
    // the tags to add to the inference
    pub tags: HashMap<String, String>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<String>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Tool>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<ToolChoice>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<bool>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    pub output_schema: Option<Value>,
    pub credentials: HashMap<String, ClientSecretString>,
    pub cache_options: CacheParamsOptions,
    /// If `true`, add an `original_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `original_response`
    /// if the fuser/judge model failed
    #[serde(default)]
    pub include_original_response: bool,
    #[serde(default)]
    pub extra_body: UnfilteredInferenceExtraBody,
}

impl TryFrom<ClientInferenceParams> for Params {
    type Error = Error;
    fn try_from(this: ClientInferenceParams) -> Result<Self, Error> {
        let mut messages = Vec::with_capacity(this.input.messages.len());
        for message in this.input.messages {
            let mut content = Vec::with_capacity(message.content.len());
            for input_content in message.content {
                content.push(InputMessageContent::try_from(input_content)?);
            }
            messages.push(InputMessage {
                role: message.role,
                content,
            });
        }
        Ok(Params {
            function_name: this.function_name,
            model_name: this.model_name,
            episode_id: this.episode_id,
            input: Input {
                system: this.input.system,
                messages,
            },
            stream: this.stream,
            params: this.params,
            variant_name: this.variant_name,
            dryrun: this.dryrun,
            tags: this.tags,
            internal: this.internal,
            dynamic_tool_params: this.dynamic_tool_params,
            output_schema: this.output_schema,
            // TODO - can we avoid reconstructing the hashmap here?
            credentials: this
                .credentials
                .into_iter()
                .map(|(k, v)| (k, v.0))
                .collect(),
            cache_options: this.cache_options,
            include_original_response: this.include_original_response,
            extra_body: this.extra_body,
        })
    }
}

// This asserts that the fields in `ClientInferenceParams` match the fields in `Params`,
// by explicitly naming all of the fields in both structs.
// This will stop compiling if the fields don't match.
#[allow(unused)]
fn assert_params_match(client_params: ClientInferenceParams) {
    let ClientInferenceParams {
        function_name,
        model_name,
        episode_id,
        input,
        stream,
        params,
        variant_name,
        dryrun,
        tags,
        internal,
        dynamic_tool_params,
        output_schema,
        credentials,
        cache_options,
        include_original_response,
        extra_body,
    } = client_params;
    let _ = Params {
        function_name,
        model_name,
        episode_id,
        input: test_client_input_to_input(input),
        stream,
        params,
        variant_name,
        dryrun,
        tags,
        internal,
        dynamic_tool_params,
        output_schema,
        credentials: credentials.into_iter().map(|(k, v)| (k, v.0)).collect(),
        cache_options,
        include_original_response,
        extra_body,
    };
}

#[derive(Clone, Debug, Deserialize)]
/// A `SecretString` wrapper that implements `Serialize`, allowing it to be used in
/// the client request input.
pub struct ClientSecretString(pub SecretString);

impl Deref for ClientSecretString {
    type Target = SecretString;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Serialize for ClientSecretString {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.expose_secret().serialize(serializer)
    }
}

// The orphan rule requires us to write some impls in this crate, instead of in the `python-pyo3` wrapper crate.
#[cfg(feature = "pyo3")]
mod pyo3_impls {
    use super::*;
    use pyo3::prelude::*;

    impl<'py> FromPyObject<'py> for ClientSecretString {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let secret: String = ob.extract()?;
            Ok(ClientSecretString(SecretString::new(secret.into())))
        }
    }
}
