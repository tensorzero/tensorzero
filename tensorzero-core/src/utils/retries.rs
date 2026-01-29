#[expect(clippy::disallowed_types)]
use backon::{BackoffBuilder, ExponentialBuilder, Retryable};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{future::Future, time::Duration};

use crate::error::Error;

/// Result of a retry operation that includes both the final result and any intermediate failures.
pub struct RetryResultWithFailures<R> {
    pub result: Result<R, Error>,
    pub failed_attempts: Vec<Error>,
}

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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Copy, Clone, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

    /// Like `retry`, but also returns errors from failed retry attempts.
    /// This allows collecting raw_response data from failed attempts.
    pub async fn retry_collecting_failures<R, F: Future<Output = Result<R, Error>>>(
        &self,
        mut func: impl FnMut() -> F,
    ) -> RetryResultWithFailures<R> {
        let mut failed_attempts = Vec::new();
        let mut backoff_iter = self.get_backoff().build();

        loop {
            match func().await {
                Ok(result) => {
                    return RetryResultWithFailures {
                        result: Ok(result),
                        failed_attempts,
                    };
                }
                Err(error) => {
                    // Check if we should retry
                    if !error.is_retryable() {
                        return RetryResultWithFailures {
                            result: Err(error),
                            failed_attempts,
                        };
                    }

                    // Get next backoff delay; None means retries exhausted
                    let Some(delay) = backoff_iter.next() else {
                        return RetryResultWithFailures {
                            result: Err(error),
                            failed_attempts,
                        };
                    };

                    // Collect the failed attempt and wait before retrying
                    failed_attempts.push(error);
                    tokio::time::sleep(delay).await;
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
