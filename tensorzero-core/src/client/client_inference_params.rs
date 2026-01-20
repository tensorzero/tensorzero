use std::{collections::HashMap, ops::Deref};

use crate::{
    cache::CacheParamsOptions,
    config::UninitializedVariantInfo,
    endpoints::inference::{InferenceParams, Params},
    error::Error,
    inference::types::{
        Input, extra_body::UnfilteredInferenceExtraBody,
        extra_headers::UnfilteredInferenceExtraHeaders,
    },
    tool::DynamicToolParams,
};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// This is a copy-paste of the `Params` struct from `tensorzero_core::endpoints::inference::Params`.
// with just the `credentials` field adjusted to allow serialization.
/// The expected payload is a JSON object with the following fields:
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct ClientInferenceParams {
    // The function name. Exactly one of `function_name` or `model_name` must be provided.
    pub function_name: Option<String>,
    // The model name to run using a default function. Exactly one of `function_name` or `model_name` must be provided.
    pub model_name: Option<String>,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    pub episode_id: Option<Uuid>,
    // the input for the inference
    pub input: Input,
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
    #[cfg_attr(feature = "ts-bindings", ts(type = "Map<string, string>"))]
    pub credentials: HashMap<String, ClientSecretString>,
    pub cache_options: CacheParamsOptions,
    /// DEPRECATED (#5697 / 2026.4+): Use `include_raw_response` instead.
    /// If `true`, add an `original_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `original_response`
    /// if the fuser/judge model failed.
    #[serde(default)]
    pub include_original_response: bool,
    /// If `true`, add a `raw_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `raw_response`
    /// if the fuser/judge model failed.
    #[serde(default)]
    pub include_raw_response: bool,
    /// If `true`, include `raw_usage` in the response's `usage` field, containing the raw usage data from each model inference.
    #[serde(default)]
    pub include_raw_usage: bool,
    // NOTE: Currently, ts_rs does not handle #[serde(transparent)] correctly,
    // so we disable the type generation for the extra_body and extra_headers fields.
    // I tried doing a direct #[ts(type = "InferenceExtraBody[]")] and
    // a #[ts(as = "Vec<InferenceExtraBody>")] and these would generate the types but then
    // type checking would fail because the ClientInferenceParams struct would not be
    // generated with the correct import.
    //
    // Not sure if this is solvable with the existing crate.
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub extra_body: UnfilteredInferenceExtraBody,
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub internal_dynamic_variant_config: Option<UninitializedVariantInfo>,

    /// OTLP trace headers to attach to the HTTP request to the TensorZero Gateway.
    /// These headers will be prefixed with `tensorzero-otlp-traces-extra-header-` and
    /// forwarded to the OTLP exporter. This field is not serialized into the request body.
    #[serde(skip)]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub otlp_traces_extra_headers: HashMap<String, String>,

    #[serde(skip)]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub otlp_traces_extra_attributes: HashMap<String, String>,

    #[serde(skip)]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub otlp_traces_extra_resources: HashMap<String, String>,

    /// Tensorzero API key to set in the `Authorization` header when making the HTTP request to the TensorZero Gateway.
    /// This field is not serialized into the request body.
    #[serde(skip)]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub api_key: Option<SecretString>,
}

impl TryFrom<ClientInferenceParams> for Params {
    type Error = Error;
    fn try_from(this: ClientInferenceParams) -> Result<Self, Error> {
        Ok(Params {
            function_name: this.function_name,
            model_name: this.model_name,
            episode_id: this.episode_id,
            input: this.input,
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
            include_raw_response: this.include_raw_response,
            include_raw_usage: this.include_raw_usage,
            extra_body: this.extra_body,
            extra_headers: this.extra_headers,
            internal_dynamic_variant_config: this.internal_dynamic_variant_config,
        })
    }
}

// This asserts that the fields in `ClientInferenceParams` match the fields in `Params`,
// by explicitly naming all of the fields in both structs.
// This will stop compiling if the fields don't match.
#[expect(unused)]
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
        include_raw_response,
        include_raw_usage,
        extra_body,
        extra_headers,
        internal_dynamic_variant_config,
        otlp_traces_extra_headers: _,
        otlp_traces_extra_attributes: _,
        otlp_traces_extra_resources: _,
        api_key: _,
    } = client_params;
    let _ = Params {
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
        credentials: credentials.into_iter().map(|(k, v)| (k, v.0)).collect(),
        cache_options,
        include_original_response,
        include_raw_response,
        include_raw_usage,
        extra_body,
        extra_headers,
        internal_dynamic_variant_config,
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

// The orphan rule requires us to write some impls in this crate, instead of in the `python` wrapper crate.
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
