use std::collections::HashMap;

use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppStateData;
use axum::routing::MethodRouter;
use std::convert::Infallible;

pub mod batch_inference;
pub mod datasets;
pub mod embeddings;
pub mod episodes;
pub mod fallback;
pub mod feedback;
pub mod functions;
pub mod inference;
pub mod internal;
pub mod object_storage;
pub mod openai_compatible;
pub mod shared_types;
pub mod status;
pub mod stored_inferences;
pub mod ui;
pub mod variant_probabilities;
pub mod workflow_evaluation_run;
pub mod workflow_evaluations;

/// Helper struct to hold the data needed for a call to `Router.route`
/// We use this to pass the route names to middleware before they register the routes
/// (since axum doesn't let you list the registered routes in a `Router`)
pub struct RouteHandlers {
    pub routes: Vec<(&'static str, MethodRouter<AppStateData, Infallible>)>,
}

pub fn validate_tags(tags: &HashMap<String, String>, internal: bool) -> Result<(), Error> {
    if internal {
        return Ok(());
    }
    for tag_name in tags.keys() {
        if tag_name.starts_with("tensorzero::") {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Tag name cannot start with 'tensorzero::': {tag_name}"),
            }));
        }
    }
    Ok(())
}

pub use tensorzero_auth::middleware::TensorzeroAuthMiddlewareState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tags() {
        let mut tags = HashMap::new();
        assert!(validate_tags(&tags, false).is_ok());
        tags.insert("tensorzero::test".to_string(), "test".to_string());
        assert!(validate_tags(&tags, false).is_err());
        // once we're in internal mode, we can have tags that start with "tensorzero::"
        assert!(validate_tags(&tags, true).is_ok());
    }
}
