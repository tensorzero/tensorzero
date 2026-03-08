//! Generic adaptive experimentation types.
//!
//! This module defines the algorithm-agnostic wrapper types for adaptive experimentation.
//! The `algorithm` field selects the concrete strategy (currently only Track-and-Stop),
//! while the inner config holds the algorithm-specific parameters.
//!
//! When new adaptive algorithms are added, they should be added as variants to
//! `AdaptiveExperimentationAlgorithm`, and the `load()` method should dispatch to the
//! appropriate algorithm-specific config.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::MetricConfig,
    db::{feedback::FeedbackQueries, postgres::PostgresConnectionInfo},
    error::Error,
    variant::VariantInfo,
};

use super::VariantSampler;
use super::track_and_stop::{TrackAndStopConfig, UninitializedTrackAndStopExperimentationConfig};

/// Algorithm used for adaptive experimentation.
/// Currently only `TrackAndStop` is supported.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveExperimentationAlgorithm {
    #[default]
    TrackAndStop,
}

/// Uninitialized adaptive experimentation config.
/// Wraps a track-and-stop config (the only algorithm currently supported) with
/// an additional `algorithm` field.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct UninitializedAdaptiveExperimentationConfig {
    #[serde(default)]
    pub algorithm: AdaptiveExperimentationAlgorithm,
    // All track-and-stop fields are flattened in (since this is the only option at the moment)
    #[serde(flatten)]
    pub inner: UninitializedTrackAndStopExperimentationConfig,
}

/// Loaded adaptive experimentation config.
/// Wraps a loaded `TrackAndStopConfig` with `algorithm` metadata.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AdaptiveExperimentationConfig {
    pub algorithm: AdaptiveExperimentationAlgorithm,
    #[serde(flatten)]
    pub inner: TrackAndStopConfig,
}

impl UninitializedAdaptiveExperimentationConfig {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, MetricConfig>,
        namespace: Option<String>,
    ) -> Result<AdaptiveExperimentationConfig, Error> {
        let inner = self.inner.load(variants, metrics, namespace)?;
        Ok(AdaptiveExperimentationConfig {
            algorithm: self.algorithm,
            inner,
        })
    }
}

impl VariantSampler for AdaptiveExperimentationConfig {
    async fn setup(
        &self,
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        postgres: &PostgresConnectionInfo,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        self.inner
            .setup(db, function_name, postgres, cancel_token)
            .await
    }

    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        self.inner
            .sample(function_name, episode_id, active_variants, postgres)
            .await
    }

    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_ {
        self.inner.allowed_variants()
    }

    fn get_current_display_probabilities<'a>(
        &self,
        function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        self.inner
            .get_current_display_probabilities(function_name, active_variants, postgres)
    }
}
