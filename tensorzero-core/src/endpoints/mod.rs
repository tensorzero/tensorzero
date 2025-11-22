use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, ErrorDetails};

pub mod batch_inference;
pub mod datasets;
pub mod embeddings;
pub mod fallback;
pub mod feedback;
pub mod inference;
pub mod object_storage;
pub mod openai_compatible;
pub mod shared_types;
pub mod status;
pub mod stored_inferences;
pub mod variant_probabilities;
pub mod workflow_evaluation_run;
use crate::utils::gateway::AppStateData;
use axum::routing::MethodRouter;
use std::convert::Infallible;
use tensorzero_auth::key::TensorZeroApiKey;

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

// Our auth middleware stores this in the request extensions when auth is enabled,
// *and* the API key is valid.
// This is used to get access to the API key (e.g. for rate-limiting), *not* to require that a route is authenticated -
// the `auth` middleware itself rejects requests without API keys
#[derive(Debug, Clone)]
pub struct RequestApiKeyExtension {
    pub api_key: Arc<TensorZeroApiKey>,
}

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
