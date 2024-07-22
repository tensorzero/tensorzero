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
    State(AppStateData { config }): AppState,
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
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());

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
            sample_variant(&variants, &params.function_name, &episode_id)?;

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
    let total_weight = variants
        .values()
        .map(|variant| variant.weight())
        .sum::<f64>();

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
        cumulative_weight += variant.weight();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::ChatCompletionConfig;
    use std::collections::HashMap;

    #[test]
    fn test_get_uniform_value() {
        // Test with function name and episode ID
        let episode_id = Uuid::now_v7();
        let value1 = get_uniform_value("test_function", &episode_id);
        let value2 = get_uniform_value("test_function", &episode_id);

        // Values should be the same due to deterministic input
        assert_eq!(value1, value2);
        assert!((0.0..1.0).contains(&value1));
        assert!((0.0..1.0).contains(&value2));

        // Test with different function names
        let value3 = get_uniform_value("another_function", &episode_id);
        assert_ne!(value1, value3);
        assert!((0.0..1.0).contains(&value3));

        // Test with different episode IDs
        let value4 = get_uniform_value("test_function", &Uuid::now_v7());
        assert_ne!(value1, value4);
        assert_ne!(value3, value4);
        assert!((0.0..1.0).contains(&value4));
    }

    /// Tests the `sample_variant` function with a variety of test cases through Monte Carlo simulations.
    ///
    /// NOTE: If this test fails, it might be due to sampling. Please run it again to check if the
    ///       issue persists.
    #[test]
    fn test_sample_variant() {
        // Helper function to create a HashMap of variant names to their weights
        fn create_variants(variant_weights: &[(&str, f64)]) -> HashMap<String, VariantConfig> {
            variant_weights
                .iter()
                .map(|&(name, weight)| {
                    (
                        name.to_string(),
                        VariantConfig::ChatCompletion(ChatCompletionConfig {
                            weight,
                            model: "model-name".to_string(),
                            system_template: None,
                            user_template: None,
                            assistant_template: None,
                        }),
                    )
                })
                .collect()
        }

        // Helper function to test the distribution of variant weights by sampling them many times
        // and checking if the observed distribution is close to the expected distribution
        fn test_variant_distribution(
            variants: &HashMap<String, VariantConfig>,
            sample_size: usize,
            tolerance: f64,
        ) {
            let total_weight: f64 = variants.values().map(|v| v.weight()).sum();
            let mut counts: HashMap<String, usize> = HashMap::new();

            for _ in 0..sample_size {
                let (variant_name, _) =
                    sample_variant(variants, "test_function", &Uuid::now_v7()).unwrap();
                *counts.entry(variant_name.clone()).or_insert(0) += 1;
            }

            for (variant_name, variant) in variants {
                let expected_prob = variant.weight() / total_weight;
                let actual_prob =
                    *counts.get(variant_name).unwrap_or(&0) as f64 / sample_size as f64;
                let diff = (expected_prob - actual_prob).abs();

                println!(
                    "Variant {}: Expected probability = {}, Actual probability = {}",
                    variant_name, expected_prob, actual_prob
                );

                assert!(
                    diff <= tolerance,
                    "Probability for variant {} is outside the acceptable range",
                    variant_name
                );
            }
        }

        // Test case 1: Equal weights
        let variants = create_variants(&[("A", 1.0), ("B", 1.0), ("C", 1.0)]);
        test_variant_distribution(&variants, 10_000, 0.02);

        // Test case 2: Unequal weights
        let variants = create_variants(&[("X", 1.0), ("Y", 2.0), ("Z", 3.0)]);
        test_variant_distribution(&variants, 10_000, 0.02);

        // Test case 3: Extreme weights
        let variants = create_variants(&[("Rare", 0.01), ("Common", 0.99)]);
        test_variant_distribution(&variants, 10_000, 0.005);

        // Test case 4: Single weights
        let variants = create_variants(&[("Solo", 1.0)]);
        test_variant_distribution(&variants, 10_000, 0.0);
    }
}
