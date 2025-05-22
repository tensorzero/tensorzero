// This is an internal crate, so we're the only consumers of
// traits with async fns for now.
#![expect(async_fn_in_trait)]

pub mod cache;
pub mod clickhouse;
pub mod config_parser; // TensorZero config file
pub mod embeddings; // embedding inference
pub mod endpoints; // API endpoints
pub mod error; // error handling
pub mod evaluations; // evaluation
pub mod function; // types and methods for working with TensorZero functions
pub mod gateway_util; // utilities for gateway
pub mod inference; // model inference
pub mod jsonschema_util; // utilities for working with JSON schemas
mod minijinja_util; // utilities for working with MiniJinja templates
pub mod model; // types and methods for working with TensorZero-supported models
pub mod model_table;
pub mod observability; // utilities for observability (logs, metrics, etc.)
mod testing;
pub mod tool; // types and methods for working with TensorZero tools
mod uuid_util; // utilities for working with UUIDs
pub mod variant; // types and methods for working with TensorZero variants

pub mod built_info {
    #![expect(clippy::allow_attributes)]
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}
