#[expect(clippy::disallowed_types)]
use backon::{ExponentialBuilder, Retryable};
use serde::{Deserialize, Serialize};
use std::{future::Future, time::Duration};

use crate::error::Error;

/*
 * This file implements TensorZero's custom retry logic.
 * We allow retries to be configured via the RetryConfig struct in the TOML.
 * We also handle the fact that certain errors are non-retryable.
 *
 * The implementation is based on the backon crate.
 * We ban use of the backon crate anywhere but in this file so callers must use
 * this code.
 */

#[derive(Debug, Deserialize, Copy, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct RetryConfig {
    #[serde(default = "default_num_retries")]
    pub num_retries: usize,
    #[serde(default = "default_max_delay_s")]
    pub max_delay_s: f32,
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

    fn get_backoff(&self) -> backon::ExponentialBuilder {
        ExponentialBuilder::default()
            .with_jitter()
            .with_max_delay(Duration::from_secs_f32(self.max_delay_s))
            .with_max_times(self.num_retries)
    }
}
