use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;

use crate::config::Config;
use crate::endpoints::workflow_evaluation_run::validate_variant_pins;
use crate::error::Error;
use crate::stored_inference::{render_stored_sample, RenderedSample, StoredSample};

pub async fn render_samples<T: StoredSample>(
    config: Arc<Config>,
    stored_samples: Vec<T>,
    variants: HashMap<String, String>,
) -> Result<Vec<RenderedSample>, Error> {
    validate_variant_pins(&variants, &config)?;
    let resolution_futures = stored_samples
        .iter()
        .map(|inference_example| inference_example.input().clone().reresolve(&*config));

    // Await all futures concurrently.
    // For now, we drop the errors here.
    // They are logged on construction in the task.
    // TODO: make it configurable whether to drop or error on failures.
    let results = join_all(resolution_futures).await;

    let final_rendered_examples: Vec<RenderedSample> = join_all(
        stored_samples
            .into_iter() // Consumes Vec<impl StoredSample>; elements are already mutated
            .zip(results.into_iter()) // Creates an iterator of (StoredInference, Result<(), Error>)
            .filter_map(|(example, resolution_result)| {
                // Filter out examples where reresolve_input_for_fine_tuning failed.
                // If resolution_result is Ok, map Some(()) to Some(example).
                // If resolution_result is Err, .ok() yields None, so filter_map drops it.
                resolution_result.ok().map(|resolved| (example, resolved))
            })
            .map(|(sample, resolved_input)| async {
                // resolved_example is a StoredInference that was successfully processed by reresolve.
                // Now, attempt to render it.
                // render_stored_inference returns Result<RenderedStoredInference, Error>.
                // .ok() converts this to Option<RenderedStoredInference>.
                // filter_map will keep Some(RenderedStoredInference) and discard None (if rendering failed).
                render_stored_sample(sample, resolved_input, &config, &variants)
                    .await
                    .ok()
            }),
    )
    .await
    .into_iter()
    .flatten()
    .collect();

    Ok(final_rendered_examples)
}
