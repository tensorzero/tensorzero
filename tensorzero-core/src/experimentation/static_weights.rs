use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use uuid::Uuid;

use crate::{
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    variant::VariantInfo,
};

use super::VariantSampler;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StaticWeightsConfig {
    // Map from variant name to weight. We enforce that weights are positive at construction time.
    candidate_variants: BTreeMap<String, f64>,
    // list of fallback variants (we will uniformly sample from these at inference time)
    fallback_variants: Vec<String>,
}

impl StaticWeightsConfig {
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        // TODO: produce a candidate variants map
        // and a list of fallback variants
        let mut candidate_variants = BTreeMap::new();
        let mut fallback_variants = Vec::new();

        for (name, variant) in variants {
            if let Some(weight) = variant.inner.weight() {
                if weight > 0.0 {
                    candidate_variants.insert(name.clone(), weight);
                }
                // If the weight is 0 then it is explicitly disabled and we don't include it
            } else {
                fallback_variants.push(name.clone());
            }
        }

        Self {
            candidate_variants,
            fallback_variants,
        }
    }
}

impl VariantSampler for StaticWeightsConfig {
    async fn setup(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Sample a variant from the function based on variant weights (a categorical distribution)
    /// This function pops the sampled variant from the candidate variants map.
    /// NOTE: We use a BTreeMap to ensure that the variants are sorted by their names and the
    /// sampling choices are deterministic given an episode ID.
    async fn inner_sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        // Compute the total weight of variants present in variant_names
        let total_weight = active_variants
            .keys()
            .map(|variant_name| self.candidate_variants.get(variant_name).unwrap_or(&0.0))
            .sum::<f64>();
        if total_weight <= 0.0 {
            // Assume that there are no active variants in the candidate set so
            // we will try the fallback variants
            // First, we take the intersection of active_variants and fallback_variants
            let intersection = active_variants
                .keys()
                .filter(|variant_name| self.fallback_variants.contains(variant_name))
                .collect::<Vec<_>>();
            if intersection.is_empty() {
                Err(ErrorDetails::NoFallbackVariantsRemaining.into())
            } else {
                // Randomly select a variant from the intersection
                // (we don't need to use the hashing trick since this is a fallback)
                // we should use the rand crate instead and pop it from active variants
                let mut rng = rand::rng();
                let selected_variant = intersection.iter().choose(&mut rng).ok_or_else(|| Error::new(ErrorDetails::Inference {
                    message: format!("Failed to sample variant from nonempty intersection. {IMPOSSIBLE_ERROR_MESSAGE}")
                }))?.to_string();
                let variant_data = active_variants.remove(&selected_variant).ok_or_else(|| Error::new(ErrorDetails::Inference {
                    message: format!("Failed to remove variant from active variants. {IMPOSSIBLE_ERROR_MESSAGE}")
                }))?;
                Ok((selected_variant, variant_data))
            }
        } else {
            // Sample a variant from the candidate variants
            // later
            todo!()
        }
    }
}
