use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use itertools::izip;
use metrics::counter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::iter::repeat;
use tracing::instrument;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::function::sample_variant;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::batch::{
    BatchEpisodeIds, BatchEpisodeIdsWithSize, BatchInferenceDatabaseInsertMetadata,
    BatchInferenceParams, BatchInferenceParamsWithSize, BatchModelInferenceRow,
    BatchOutputSchemasWithSize, BatchRequestRow, BatchStatus,
};
use crate::inference::types::RequestMessage;
use crate::inference::types::{batch::BatchModelInferenceWithMetadata, Input};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::tool::{
    BatchDynamicToolParams, BatchDynamicToolParamsWithSize, DynamicToolParams, ToolCallConfig,
    ToolCallConfigDatabaseInsert,
};
use crate::variant::{BatchInferenceConfig, Variant};

use super::inference::{InferenceClients, InferenceCredentials, InferenceModels, InferenceParams};

/// The expected payload to the `/start_batch_inference` endpoint.
/// It will be a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the function name
    pub function_name: String,
    // the episode IDs for each inference (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    #[serde(default)]
    pub episode_ids: Option<BatchEpisodeIdInput>,
    // the inputs for the inferences
    pub inputs: Vec<Input>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    pub params: BatchInferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // the tags to add to the inference
    #[serde(default)]
    pub tags: Option<BatchTags>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: BatchDynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<Option<Vec<String>>>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Option<Vec<Tool>>>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<Vec<Option<ToolChoice>>>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<Vec<Option<bool>>>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    #[serde(default)]
    pub output_schemas: Option<BatchOutputSchemas>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
}

pub type BatchEpisodeIdInput = Vec<Option<Uuid>>;
pub type BatchTags = Vec<Option<HashMap<String, String>>>;
pub type BatchOutputSchemas = Vec<Option<Value>>;

/// This handler starts a batch inference request for a particular function.
/// The entire batch must use the same function and variant.
/// It will fail if we fail to kick off the batch request for any reason.
/// However, the batch request might still fail for other reasons after it has been started.
#[instrument(
    name="start_batch_inference",
    skip_all,
    fields(
        function_name = %params.function_name,
        variant_name = ?params.variant_name,
    )
)]
#[debug_handler(state = AppStateData)]
pub async fn start_batch_inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;
    let num_inferences = params.inputs.len();
    if num_inferences == 0 {
        return Err(ErrorDetails::InvalidRequest {
            message: "No inputs provided".to_string(),
        }
        .into());
    }
    // Collect the tool params and output schemas into vectors of the same length as the batch
    let batch_dynamic_tool_params: Vec<DynamicToolParams> =
        BatchDynamicToolParamsWithSize(params.dynamic_tool_params, num_inferences).try_into()?;
    let batch_dynamic_output_schemas: Vec<Option<DynamicJSONSchema>> =
        BatchOutputSchemasWithSize(params.output_schemas, num_inferences).try_into()?;

    let tool_configs = batch_dynamic_tool_params
        .into_iter()
        .map(|dynamic_tool_params| function.prepare_tool_config(dynamic_tool_params, &config.tools))
        .collect::<Result<Vec<_>, _>>()?;
    // Collect the function variant names as a Vec<&str>
    let mut candidate_variant_names: Vec<&str> =
        function.variants().keys().map(AsRef::as_ref).collect();

    // If the function has no variants, return an error
    if candidate_variant_names.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        }
        .into());
    }

    // Validate the input
    params
        .inputs
        .iter()
        .enumerate()
        .try_for_each(|(i, input)| {
            function.validate_input(input).map_err(|e| {
                Error::new(ErrorDetails::BatchInputValidation {
                    index: i,
                    message: e.to_string(),
                })
            })
        })?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variant_names.retain(|k| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variant_names.is_empty() {
            return Err(ErrorDetails::UnknownVariant {
                name: variant_name.to_string(),
            }
            .into());
        }
    }

    // Retrieve or generate the episode IDs and validate them (in the impl)
    let episode_ids: BatchEpisodeIds =
        BatchEpisodeIdsWithSize(params.episode_ids, num_inferences).try_into()?;

    // Increment the request count
    counter!(
        "request_count",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(1);
    counter!(
        "inference_count",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(num_inferences as u64);

    // Keep track of which variants failed
    let mut variant_errors = std::collections::HashMap::new();
    let inference_config = BatchInferenceConfig::new(
        &config.templates,
        tool_configs,
        batch_dynamic_output_schemas,
        &params.function_name,
        params.variant_name.as_deref(),
    );

    let inference_clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse_connection_info,
        credentials: &params.credentials,
    };

    let inference_models = InferenceModels {
        models: &config.models,
        embedding_models: &config.embedding_models,
    };
    let inference_params: Vec<InferenceParams> =
        BatchInferenceParamsWithSize(params.params, num_inferences).try_into()?;

    // Keep sampling variants until one succeeds
    // We already guarantee there is at least one inference
    let first_episode_id = episode_ids
        .first()
        .ok_or_else(|| Error::new(ErrorDetails::Inference {
            message: "batch episode_ids unexpectedly empty. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
        }))?;
    let cloned_config = inference_config.clone();
    let inference_configs = cloned_config.inference_configs();
    while !candidate_variant_names.is_empty() {
        // We sample the same variant for the whole batch
        let (variant_name, variant) = sample_variant(
            &mut candidate_variant_names,
            function.variants(),
            &params.function_name,
            first_episode_id,
        )?;
        // Will be edited by the variant as part of making the request so we must clone here
        // This could potentially be improved by decoupling the variant name from the rest of the inference params
        let variant_inference_params = inference_params.clone();

        let result = variant
            .start_batch_inference(
                &params.inputs,
                &inference_models,
                function,
                &inference_configs,
                &inference_clients,
                variant_inference_params,
            )
            .await;

        let result = match result {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                        "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                        function_name = params.function_name,
                        variant_name = variant_name,
                    );
                variant_errors.insert(variant_name.to_string(), e);
                continue;
            }
        };

        // Write to ClickHouse (don't spawn a thread for this because it's required and we should fail loudly)
        let write_metadata = BatchInferenceDatabaseInsertMetadata {
            function_name: params.function_name.as_str(),
            variant_name,
            episode_ids: &episode_ids,
            tags: params.tags,
        };

        let (batch_id, inference_ids) = write_batch_inference(
            &clickhouse_connection_info,
            params.inputs,
            result,
            write_metadata,
            inference_config,
        )
        .await?;

        return Ok(Json(PrepareBatchInferenceOutput {
            batch_id,
            inference_ids,
            episode_ids,
        })
        .into_response());
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(ErrorDetails::AllVariantsFailed {
        errors: variant_errors,
    }
    .into())
}

// Determines the return type of the `/start_batch_inference` endpoint upon success
#[derive(Debug, Serialize)]
struct PrepareBatchInferenceOutput {
    batch_id: Uuid,
    inference_ids: Vec<Uuid>,
    episode_ids: Vec<Uuid>,
}

// Helper struct for writing to the `BatchModelInference` table in ClickHouse
// This is only used to help with iteration in the `write_batch_inference` function
struct BatchInferenceRow<'a> {
    inference_id: &'a Uuid,
    input: Input,
    input_messages: Vec<RequestMessage>,
    system: Option<&'a str>,
    tool_config: Option<ToolCallConfig>,
    inference_params: &'a InferenceParams,
    output_schema: Option<&'a Value>,
    raw_request: &'a str,
    tags: Option<HashMap<String, String>>,
}

async fn write_batch_inference<'a>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inputs: Vec<Input>,
    result: BatchModelInferenceWithMetadata<'a>,
    metadata: BatchInferenceDatabaseInsertMetadata<'a>,
    inference_config: BatchInferenceConfig<'a>,
) -> Result<(Uuid, Vec<Uuid>), Error> {
    let batch_id = result.batch_id.to_string();

    // Collect all the data into BatchInferenceRow structs
    let inference_rows = izip!(
        result.inference_ids.iter(),
        inputs,
        result.input_messages.into_iter(),
        result.systems.iter(),
        inference_config.tool_configs,
        result.inference_params.iter(),
        result.output_schemas.into_iter(),
        result.raw_requests.iter(),
        metadata
            .tags
            .unwrap_or_default()
            .into_iter()
            .chain(repeat(None)),
    )
    .map(
        |(
            inference_id,
            input,
            input_messages,
            system,
            tool_config,
            inference_params,
            output_schema,
            raw_request,
            tags,
        )| {
            BatchInferenceRow {
                inference_id,
                input,
                input_messages,
                system: system.as_deref(),
                tool_config,
                inference_params,
                output_schema,
                raw_request,
                tags,
            }
        },
    );
    let mut rows: Vec<BatchModelInferenceRow<'_>> = vec![];

    // Process each row by serializing the stuff that needs to be serialized twice
    for row in inference_rows {
        let tool_params: Option<ToolCallConfigDatabaseInsert> = row.tool_config.map(|t| t.into());

        rows.push(BatchModelInferenceRow {
            inference_id: *row.inference_id,
            batch_id: &batch_id,
            function_name: metadata.function_name,
            variant_name: metadata.variant_name,
            episode_id: metadata.episode_ids[rows.len()],
            input: row.input,
            input_messages: row.input_messages,
            system: row.system,
            tool_params,
            inference_params: row.inference_params,
            output_schema: row.output_schema,
            raw_request: row.raw_request,
            model_name: result.model_name,
            model_provider_name: result.model_provider_name,
            tags: row.tags.unwrap_or_default(),
        });
    }

    clickhouse_connection_info
        .write(&rows, "BatchModelInference")
        .await?;

    let batch_request_insert = BatchRequestRow::new(
        &batch_id,
        result.batch_params,
        result.model_name,
        result.model_provider_name,
        BatchStatus::Pending,
        None, // There should not be errors for the batch request on initialization
    );
    clickhouse_connection_info
        .write(&[batch_request_insert], "BatchRequest")
        .await?;

    Ok((result.batch_id, result.inference_ids))
}
