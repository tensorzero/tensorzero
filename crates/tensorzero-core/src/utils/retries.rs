#[expect(clippy::disallowed_types)]
use backon::{BackoffBuilder, ExponentialBuilder, Retryable};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{future::Future, time::Duration};
use tensorzero_stored_config::StoredRetryConfig;

use crate::error::Error;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/*
 * This file implements TensorZero's custom retry logic.
 * We allow retries to be configured via the RetryConfig struct in the TOML.
 * We also handle the fact that certain errors are non-retryable.
 *
 * The implementation is based on the backon crate.
 * We ban use of the backon crate anywhere but in this file so callers must use
 * this code.
 */

#[derive(ts_rs::TS, Debug, Deserialize, Copy, Clone, PartialEq, JsonSchema, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct RetryConfig {
    #[serde(default = "default_num_retries")]
    pub num_retries: usize,
    #[serde(default = "default_max_delay_s")]
    pub max_delay_s: f32,
}

impl std::fmt::Display for RetryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            num_retries: default_num_retries(),
            max_delay_s: default_max_delay_s(),
        }
    }
}

fn default_num_retries() -> usize {
    0
}

fn default_max_delay_s() -> f32 {
    10.0
}

impl RetryConfig {
    pub fn retry<R, F: Future<Output = Result<R, Error>>>(
        &self,
        func: impl FnMut() -> F,
    ) -> impl Future<Output = Result<R, Error>> {
        let backoff = self.get_backoff();
        func.retry(backoff).when(Error::is_retryable)
    }

    /// Like `retry`, but collects intermediate errors from failed attempts.
    ///
    /// Returns `(final_result, intermediate_errors)`:
    /// - On success: `intermediate_errors` contains errors from all failed attempts before the success.
    /// - On failure: `intermediate_errors` contains errors from earlier attempts;
    ///   the final (non-retryable or last-attempt) error is in the `Err`.
    pub async fn retry_collecting_errors<R, F, Fut>(
        &self,
        mut func: F,
    ) -> (Result<R, Error>, Vec<Error>)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<R, Error>>,
    {
        let mut backoff = self.get_backoff().build();
        let mut errors = Vec::new();
        loop {
            match func().await {
                Ok(result) => return (Ok(result), errors),
                Err(err) => {
                    if !err.is_retryable() {
                        return (Err(err), errors);
                    }
                    match backoff.next() {
                        Some(delay) => {
                            errors.push(err);
                            tokio::time::sleep(delay).await;
                        }
                        None => return (Err(err), errors),
                    }
                }
            }
        }
    }

    fn get_backoff(&self) -> backon::ExponentialBuilder {
        ExponentialBuilder::default()
            .with_jitter()
            .with_max_delay(Duration::from_secs_f32(self.max_delay_s))
            .with_max_times(self.num_retries)
    }
}

impl From<StoredRetryConfig> for RetryConfig {
    fn from(stored: StoredRetryConfig) -> Self {
        RetryConfig {
            num_retries: stored.num_retries as usize,
            max_delay_s: stored.max_delay_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_retry_config_round_trip() {
        let original = RetryConfig {
            num_retries: 3,
            max_delay_s: 10.0,
        };
        let stored: StoredRetryConfig = original.into();
        let restored: RetryConfig = stored.into();
        expect_that!(restored, eq(original));
    }
}
