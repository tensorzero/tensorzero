use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAISFTConfig {
    batch_size: Option<usize>,
    learning_rate_multiplier: Option<f64>,
    n_epochs: Option<usize>,
}

pub struct OpenAISFTJobHandle {
    pub job_id: String,
}
