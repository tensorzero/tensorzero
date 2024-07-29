use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use uuid::Uuid;

use crate::api_util::{AppState, AppStateData, StructuredJson};
use crate::error::Error;
use crate::function::sample_variant;
use crate::inference::types::InputMessage;
use crate::variant::Variant;

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the function name
    function_name: String,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    episode_id: Option<Uuid>,
    // the input for the inference
    input: Vec<InputMessage>,
    // the maximum number of tokens to generate (if not provided, the default value will be used)
    #[allow(unused)] // TODO: remove
    max_tokens: Option<u32>,
    // default False
    stream: Option<bool>,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. DEBUGGING).
    variant_name: Option<String>,
    // if true, the inference will not be stored
    dryrun: Option<bool>,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        ..
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;

    // Clone the function variants so we can modify the collection as we sample them
    let mut variants = function.variants().clone();

    // If the function has no variants, return an error
    if variants.is_empty() {
        return Err(Error::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        });
    }

    // Validate the input
    function.validate_input(&params.input)?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        variants.retain(|k, _| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if variants.is_empty() {
            return Err(Error::UnknownVariant {
                name: params.variant_name.unwrap(),
            });
        }
    }

    // Retrieve or generate the episode ID
    // TODO: validate that the episode ID is a UUIDv7
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());

    // Should we store the results?
    #[allow(unused)] // TODO: remove
    let dryrun = params.dryrun.unwrap_or(false);

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    let mut models_tried = std::collections::HashSet::new();

    // Keep sampling variants until one succeeds
    while !variants.is_empty() {
        let (variant_name, variant) =
            sample_variant(&mut variants, &params.function_name, &episode_id)?;
        // `insert` returns false if the value was already in the set
        if !models_tried.insert(variant_name.clone()) {
            continue;
        }
        if stream {
            #[allow(unused)] // TODO: remove
            let stream = variant
                .infer_stream(
                    &params.input,
                    &config.models,
                    function.output_schema(),
                    &http_client,
                )
                .await?;
            // TODO: handle streaming inference
        } else {
            variant
                .infer(
                    &params.input,
                    &config.models,
                    function.output_schema(),
                    &http_client,
                )
                .await?;
        }
        // TODO: figure out storage, return types, streaming, streaming retries,

        todo!("Run inference and store results");
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    // TODO: make this a nicer error that contains all the errors from below.
    Err(Error::Inference {
        message: "Inference failed for every variant in this function".to_string(),
    })
}
