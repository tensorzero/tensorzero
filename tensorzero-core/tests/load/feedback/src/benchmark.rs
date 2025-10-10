use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use async_trait::async_trait;
use rlt::{BenchSuite, IterInfo, IterReport, Status};
use serde_json::Value;
use tensorzero::{Client, FeedbackParams};
use tokio::time::Instant;
use uuid::Uuid;

#[derive(Clone)]
pub struct RateLimitBenchmark {
    pub client: Arc<Client>,
    pub inference_ids: Arc<Vec<Uuid>>,
    pub request_counter: Arc<AtomicU64>,
}

pub struct WorkerState {
    pub client: Arc<Client>,
    pub inference_ids: Arc<Vec<Uuid>>,
}

#[async_trait]
impl BenchSuite for RateLimitBenchmark {
    type WorkerState = WorkerState;

    async fn state(&self, _worker_id: u32) -> Result<Self::WorkerState> {
        Ok(WorkerState {
            client: self.client.clone(),
            inference_ids: self.inference_ids.clone(),
        })
    }

    async fn bench(
        &mut self,
        state: &mut Self::WorkerState,
        _info: &IterInfo,
    ) -> Result<IterReport> {
        if state.inference_ids.is_empty() {
            return Ok(IterReport {
                duration: std::time::Duration::from_nanos(0),
                status: Status::error(500),
                bytes: 0,
                items: 0,
            });
        }

        // Select an inference ID using the global counter for pseudo-randomness
        let counter = self
            .request_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let idx = (counter as usize) % state.inference_ids.len();
        let inference_id = state.inference_ids[idx];
        let random_value = (counter % 1000) as f64 / 1000.0; // pseudo-random value 0-1

        // Create feedback parameters
        let feedback_params = FeedbackParams {
            inference_id: Some(inference_id),
            episode_id: None,
            metric_name: "test".to_string(),
            value: Value::Number(serde_json::Number::from_f64(random_value).unwrap()),
            internal: false,
            tags: std::collections::HashMap::new(),
            dryrun: None,
        };

        let start = Instant::now();
        let result = state.client.feedback(feedback_params).await;
        let duration = start.elapsed();

        match result {
            Ok(_response) => Ok(IterReport {
                duration,
                status: Status::success(200),
                bytes: 0,
                items: 1,
            }),
            Err(_e) => Ok(IterReport {
                duration,
                status: Status::error(500),
                bytes: 0,
                items: 0,
            }),
        }
    }
}
