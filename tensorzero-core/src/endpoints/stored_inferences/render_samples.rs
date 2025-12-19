use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::endpoints::workflow_evaluation_run::validate_variant_pins;
use crate::error::{Error, ErrorDetails};
use crate::stored_inference::{RenderedSample, StoredSample, render_stored_sample};

const DEFAULT_CONCURRENCY: usize = 100;

pub async fn render_samples<T: StoredSample>(
    config: Arc<Config>,
    stored_samples: Vec<T>,
    variants: HashMap<String, String>,
    concurrency: Option<usize>,
) -> Result<Vec<RenderedSample>, Error> {
    validate_variant_pins(&variants, &config)?;

    let concurrency = concurrency.unwrap_or(DEFAULT_CONCURRENCY);
    if concurrency == 0 {
        return Err(ErrorDetails::InvalidRequest {
            message: "concurrency must be at least 1".to_string(),
        }
        .into());
    }
    let semaphore = Arc::new(Semaphore::new(concurrency));

    // Process all samples concurrently with semaphore-limited concurrency.
    // For now, we drop the errors here.
    // They are logged on construction in the task.
    // TODO: make it configurable whether to drop or error on failures.
    let futures = stored_samples.into_iter().map(|sample| {
        let semaphore = semaphore.clone();
        let config = config.clone();
        let variants = variants.clone();
        async move {
            // Acquire semaphore permit for this sample's processing
            let _permit = semaphore.acquire().await.ok()?;

            // Resolve the input
            let resolved_input = sample.input().clone().reresolve(&*config).await.ok()?;

            // Render the sample
            render_stored_sample(sample, resolved_input, &config, &variants)
                .await
                .ok()
        }
    });

    let final_rendered_examples: Vec<RenderedSample> =
        join_all(futures).await.into_iter().flatten().collect();

    Ok(final_rendered_examples)
}
