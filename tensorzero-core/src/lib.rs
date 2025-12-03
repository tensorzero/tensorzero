// This is an internal crate, so we're the only consumers of
// traits with async fns for now.
#![expect(async_fn_in_trait)]

pub mod cache;
pub mod client; // Rust client for TensorZero
pub mod config; // TensorZero config file
pub mod db;
pub mod embeddings; // embedding inference
pub mod endpoints; // API endpoints
pub mod error; // error handling
pub mod evaluations; // evaluation
pub mod experimentation;
pub mod function; // types and methods for working with TensorZero functions
pub mod howdy;
pub mod http;
pub mod inference; // model inference
pub mod jsonschema_util; // utilities for working with JSON schemas
mod minijinja_util; // utilities for working with MiniJinja templates
pub mod model; // types and methods for working with TensorZero-supported models
pub mod model_table;
pub mod observability; // utilities for observability (logs, metrics, etc.)
pub mod optimization;
pub mod providers; // providers for the inference and / or optimization services TensorZero integrates
pub mod rate_limiting; // utilities for rate limiting
pub mod serde_util; // utilities for working with serde
pub mod stored_inference; // types and methods for working with stored inferences
#[cfg(any(test, feature = "e2e_tests"))]
pub mod test_helpers; // e2e test utilities for external crates
mod testing; // unit test utilities for tensorzero-core
pub mod tool; // types and methods for working with TensorZero tools
pub mod utils;
pub mod variant; // types and methods for working with TensorZero variants

pub mod built_info {
    #![expect(clippy::allow_attributes)]
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}
