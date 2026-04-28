#![recursion_limit = "256"]
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::unwrap_used
)]
mod aggregated_response;
mod best_of_n;
mod built_in;
mod cache;
mod clickhouse;
mod common;
mod config;
mod config_editing;
mod cost;
mod db;
mod db_only_boot;
mod dicl;
mod dynamic_variants;
mod endpoints;
mod experimentation;
mod fallback;
mod feedback;
mod health;
mod howdy;
mod human_feedback;
mod image_url;
mod inference;
mod inference_clickhouse;
mod inference_evaluation_human_feedback;
mod mcp;
mod mixture_of_n;
mod object_storage;
mod openai_compatible;
mod otel;
mod otel_config_headers;
mod otel_export;
mod prometheus;
mod provider_tools;
mod providers;
mod proxy;
mod rate_limiting;
mod rate_limiting_startup;
mod raw_response;
mod raw_usage;
mod render_inferences;
mod retries;
mod streaming_errors;
mod template;
mod timeouts;
mod utils;
mod workflow_evaluations;
