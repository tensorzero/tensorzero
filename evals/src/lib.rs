use anyhow::Result;
use tensorzero::{Client, ClientInferenceParams, FeedbackParams, InferenceOutput};
use tokio::sync::Semaphore;

pub mod dataset;
pub mod evaluators;
pub mod helpers;

pub struct ThrottledTensorZeroClient {
    client: Client,
    semaphore: Semaphore,
}

impl ThrottledTensorZeroClient {
    pub fn new(client: Client, semaphore: Semaphore) -> Self {
        Self { client, semaphore }
    }

    pub async fn inference(&self, params: ClientInferenceParams) -> Result<InferenceOutput> {
        let _permit = self.semaphore.acquire().await;
        let inference_output = self.client.inference(params).await?;
        Ok(inference_output)
    }

    pub async fn feedback(&self, params: FeedbackParams) -> Result<()> {
        let _permit = self.semaphore.acquire().await;
        self.client.feedback(params).await?;
        Ok(())
    }
}
