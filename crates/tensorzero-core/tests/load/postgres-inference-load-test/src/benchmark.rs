use std::sync::{Arc, atomic::AtomicU64};

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Url;
use rlt::{BenchSuite, IterInfo, IterReport, Status};
use serde_json::json;
use tokio::time::Instant;

#[derive(Clone)]
pub struct InferenceBenchmark {
    pub config: Arc<BenchmarkConfig>,
    pub run_stats: Arc<RunStats>,
    pub request_counter: Arc<AtomicU64>,
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub inference_url: Url,
    pub function_name: String,
    pub max_tokens: u32,
    pub prompt_chars: usize,
    pub randomize_prompt: bool,
    pub run_id: String,
    pub load_test_case: String,
}

#[derive(Default)]
pub struct RunStats {
    pub successful_requests: AtomicU64,
    pub total_requests: AtomicU64,
}

pub struct WorkerState {
    pub client: reqwest::Client,
}

#[async_trait]
impl BenchSuite for InferenceBenchmark {
    type WorkerState = WorkerState;

    async fn state(&self, _worker_id: u32) -> Result<Self::WorkerState> {
        Ok(WorkerState {
            client: reqwest::Client::new(),
        })
    }

    async fn bench(
        &mut self,
        state: &mut Self::WorkerState,
        info: &IterInfo,
    ) -> Result<IterReport> {
        self.run_stats
            .total_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let request_id = self
            .request_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let prompt = if self.config.randomize_prompt {
            let suffix = format!(
                " [w{} r{} i{}]",
                info.worker_id, info.runner_seq, request_id
            );
            make_prompt(self.config.prompt_chars, &suffix)
        } else {
            make_prompt(self.config.prompt_chars, "")
        };

        let payload = json!({
            "function_name": self.config.function_name,
            "stream": false,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            "params": {
                "chat_completion": {
                    "max_tokens": self.config.max_tokens,
                }
            },
            "cache_options": {
                "enabled": "off",
            },
            "tags": {
                "load_test_run_id": self.config.run_id,
                "load_test_case": self.config.load_test_case,
            }
        });

        let start = Instant::now();
        let result = state
            .client
            .post(self.config.inference_url.clone())
            .json(&payload)
            .send()
            .await;
        let duration = start.elapsed();

        match result {
            Ok(response) => {
                let status_code = response.status();
                if let Err(error) = response.bytes().await {
                    tracing::warn!("Failed to read inference response body: {error}");
                }

                if status_code.is_success() {
                    self.run_stats
                        .successful_requests
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                Ok(IterReport {
                    duration,
                    status: map_http_status(status_code),
                    bytes: 0,
                    items: u64::from(status_code.is_success()),
                })
            }
            Err(error) => {
                tracing::warn!("Inference request failed: {error}");
                Ok(IterReport {
                    duration,
                    status: Status::error(599),
                    bytes: 0,
                    items: 0,
                })
            }
        }
    }
}

fn map_http_status(status_code: reqwest::StatusCode) -> Status {
    let code = i64::from(status_code.as_u16());
    if status_code.is_success() {
        Status::success(code)
    } else if status_code.is_client_error() {
        Status::client_error(code)
    } else if status_code.is_server_error() {
        Status::server_error(code)
    } else {
        Status::error(code)
    }
}

fn make_prompt(prompt_chars: usize, suffix: &str) -> String {
    let base_len = prompt_chars.max(1);
    let base = "x".repeat(base_len);
    format!("load-test:{base}{suffix}")
}
