use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;

mod static_weights;

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type")]
pub enum ExperimentationConfig {
    StaticWeights(static_weights::StaticWeightsConfig),
    #[default]
    Uniform,
}

pub trait VariantSampler {
    // TODO, when we add bandits: pass CH and PG clients here (but use opaque trait types)
    async fn setup(&self) -> Result<(), Error>;
    async fn inner_sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error>;

    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        match active_variants.len() {
            0 => Err(Error::new(ErrorDetails::Inference {
                message: format!("VariantSampler::sample called with no active variants. {IMPOSSIBLE_ERROR_MESSAGE}")
            })),
            1 => {
                let Some((variant_name, variant)) = active_variants.pop_first() else {
                    return Err(ErrorDetails::Inference {
                        message: format!("`pop_first` returned None in the 1 case in sampling. {IMPOSSIBLE_ERROR_MESSAGE}")
                    }.into());
                };
                Ok((variant_name, variant))
            }
            _ => self.inner_sample(function_name, episode_id, active_variants).await
        }
    }
}

impl ExperimentationConfig {
    /// Note: in the future when we deprecate variant.weight we can simply use #[serde(default)]
    /// and default to ExperimentationConfig::Uniform
    ///
    /// For now, we call this function from the
    pub fn legacy_from_variants_map(variants: &HashMap<String, Arc<VariantInfo>>) -> Self {
        // We loop over the variants map and if any of them have weights we output a StaticWeightsConfig
        // otherwise, we output a Uniform config
        for variant in variants.values() {
            if variant.inner.weight().is_some() {
                return Self::StaticWeights(
                    static_weights::StaticWeightsConfig::legacy_from_variants_map(variants),
                );
            }
        }
        Self::Uniform
    }
}

impl VariantSampler for ExperimentationConfig {
    async fn setup(&self) -> Result<(), Error> {
        match self {
            Self::StaticWeights(config) => config.setup().await,
            Self::Uniform => Ok(()),
        }
    }

    async fn inner_sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        match self {
            Self::StaticWeights(config) => {
                config
                    .inner_sample(function_name, episode_id, active_variants)
                    .await
            }
            Self::Uniform => sample_uniform(function_name, &episode_id, active_variants),
        }
    }
}

fn sample_uniform(
    function_name: &str,
    episode_id: &Uuid,
    active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
) -> Result<(String, Arc<VariantInfo>), Error> {
    let random_index = (get_uniform_value(function_name, episode_id) * active_variants.len() as f64)
        .floor() as usize;
    let Some(sampled_variant_name) = active_variants.keys().nth(random_index).cloned() else {
        return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "Invalid index {random_index} for function `{function_name}` with {} variants. {IMPOSSIBLE_ERROR_MESSAGE}",
                active_variants.len()
            ),
        }));
    };
    active_variants.remove_entry(&sampled_variant_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "Function `{function_name}` has no variant for the sampled variant `{sampled_variant_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
            )
        })
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
