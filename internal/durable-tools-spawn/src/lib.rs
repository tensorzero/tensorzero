//! Lightweight durable task spawning.
//!
//! This crate provides minimal functionality for spawning durable tasks
//! by name without requiring heavy dependencies like `tensorzero` or `schemars`.
//!
//! For the full tool execution framework with workers, tool registration,
//! and inference capabilities, use the `durable-tools` crate instead.
//!
//! # Example
//!
//! ```ignore
//! use durable_tools_spawn::{SpawnClient, TaskToolParams};
//! use secrecy::SecretString;
//! use uuid::Uuid;
//!
//! let client = SpawnClient::builder()
//!     .database_url(database_url)
//!     .queue_name("tools")
//!     .build()
//!     .await?;
//!
//! let episode_id = Uuid::now_v7();
//! client.spawn_tool_by_name(
//!     "research",
//!     serde_json::json!({"topic": "rust"}),
//!     serde_json::json!(null),  // side_info (use json!(null) if not needed)
//!     episode_id,
//! ).await?;
//! ```

mod client;
mod error;
mod params;

pub use client::{SpawnClient, SpawnClientBuilder};
pub use error::{SpawnError, SpawnResult};
pub use params::TaskToolParams;

// Re-export durable types needed for spawning
pub use durable::{SpawnOptions, SpawnResult as DurableSpawnResult};

// Re-export async_trait for convenience
pub use async_trait::async_trait;
