use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;

use crate::config_parser::Config;
use crate::endpoints::dynamic_evaluation_run::validate_variant_pins;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::stored_inference::{
    render_stored_inference, reresolve_input_for_fine_tuning, RenderedStoredInference,
    StoredInference,
};

pub async fn render_inferences(
    config: Arc<Config<'static>>,
    mut inferences: Vec<StoredInference>,
    variants: HashMap<String, String>,
) -> Result<Vec<RenderedStoredInference>, Error> {
    validate_variant_pins(&variants, &config)?;
    let resolution_futures = inferences.iter_mut().map(|inference_example| {
        // Create a future for each call to reresolve_input_for_fine_tuning.
        // This function modifies inference_example.input_mut() in place.
        // `self` (the client) is passed by immutable reference.
        reresolve_input_for_fine_tuning(inference_example.input_mut(), &config)
    });

    // Await all futures concurrently.
    // For now, we drop the errors here.
    // They are logged on construction in the task.
    // TODO: make it configurable whether to drop or error on failures.
    let results = join_all(resolution_futures).await;

    // Ensure that the number of results matches the number of inference examples.
    // This should be guaranteed to be true based on the code above, but we assert it anyway.
    if inferences.len() != results.len() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Mismatch between number of inference examples and resolution results ({} != {}). {IMPOSSIBLE_ERROR_MESSAGE}",
                inferences.len(),
                results.len()
            ),
        }));
    }

    let final_rendered_examples: Vec<RenderedStoredInference> = inferences
        .into_iter() // Consumes Vec<StoredInference>; elements are already mutated
        .zip(results.into_iter()) // Creates an iterator of (StoredInference, Result<(), Error>)
        .filter_map(|(example, resolution_result)| {
            // Filter out examples where reresolve_input_for_fine_tuning failed.
            // If resolution_result is Ok, map Some(()) to Some(example).
            // If resolution_result is Err, .ok() yields None, so filter_map drops it.
            resolution_result.ok().map(|_| example)
        })
        .filter_map(|resolved_example| {
            // resolved_example is a StoredInference that was successfully processed by reresolve.
            // Now, attempt to render it.
            // render_stored_inference returns Result<RenderedStoredInference, Error>.
            // .ok() converts this to Option<RenderedStoredInference>.
            // filter_map will keep Some(RenderedStoredInference) and discard None (if rendering failed).
            render_stored_inference(resolved_example, &config, &variants).ok()
        })
        .collect();

    Ok(final_rendered_examples)
}
