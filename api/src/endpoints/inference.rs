use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

use crate::api_util::{AppState, AppStateData, StructuredJson};
use crate::error::Error;
use crate::function::{InputMessage, VariantConfig};

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the function name
    function_name: String,
    // the episode ID (if not provided, a new one will be generated)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU
    episode_id: Option<Uuid>,
    // the input for the inference (must match the function's input_schema)
    #[allow(unused)] // TODO: remove
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
    State(AppStateData { config }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;

    // Clone the function variants so we can modify the collection as we sample them
    let mut variants = function.variants.clone();

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

    // If no episode ID is provided, generate a new one
    let episode_id = params.episode_id.unwrap_or_else(Uuid::now_v7);

    // Should we store the results?
    #[allow(unused)] // TODO: remove
    let dryrun = params.dryrun.unwrap_or(false);

    // Should we stream the inference?
    #[allow(unused)] // TODO: remove
    let stream = params.stream.unwrap_or(false);

    // Keep sampling variants until one succeeds
    while !variants.is_empty() {
        #[allow(unused)] // TODO: remove
        let (variant_name, variant) =
            sample_variant(&function.variants, &params.function_name, &episode_id)?;

        todo!("Run inference and store results");
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(Error::Inference {
        message: "Inference failed for every variant in this function".to_string(),
    })
}

/// Sample a variant from the function based on variant weights (uniform random selection)
fn sample_variant<'a>(
    variants: &'a HashMap<String, VariantConfig>,
    function_name: &'a str,
    episode_id: &Uuid,
) -> Result<(&'a String, &'a VariantConfig), Error> {
    // Compute the total weight of all variants
    let total_weight = variants.values().map(|variant| variant.weight).sum::<f64>();

    // If the total weight is non-positive, return an error
    // NOTE: We enforce non-negative weights at the config parsing stage, but it's good to be extra
    //       safe here to ensure that we catch any regressions we might introduce in the future.
    if total_weight <= 0. {
        return Err(Error::InvalidFunctionVariants {
            message: format!("Function `{function_name}` variants have non-positive total weight"),
        });
    }

    // Sample a random threshold between 0 and the total weight
    let random_threshold = get_uniform_value(function_name, episode_id) * total_weight;

    // Iterate over the variants to find the one that corresponds to the sampled threshold
    let mut cumulative_weight = 0.;
    for (variant_name, variant) in variants.iter() {
        cumulative_weight += variant.weight;
        if cumulative_weight > random_threshold {
            return Ok((variant_name, variant));
        }
    }

    // Return the last variant as a fallback (this should only happen due to rare numerical precision issues)
    variants
        .iter()
        .last()
        .ok_or_else(|| Error::InvalidFunctionVariants {
            message: format!("Function `{function_name}` has no variants"),
        })
}

/// Implements a uniform distribution over the interval [0, 1) using a hash function.
/// This function is deterministic but should have good statistical properties.
fn get_uniform_value(function_name: &str, episode_id: &Uuid) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(function_name.as_bytes());
    hasher.update(episode_id.as_bytes());
    let hash_value = hasher.finalize();
    let truncated_hash =
        u32::from_be_bytes([hash_value[0], hash_value[1], hash_value[2], hash_value[3]]);
    truncated_hash as f64 / u32::MAX as f64
}
