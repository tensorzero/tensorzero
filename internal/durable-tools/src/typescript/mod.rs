//! TypeScript tool support for durable-tools.
//!
//! This module enables writing durable tools in TypeScript that have full access
//! to `ToolContext` methods via the global `ctx` object.
//!
//! # Architecture
//!
//! TypeScript tools are executed on a pool of worker threads, each owning a
//! long-lived `JsRuntime`. This architecture is necessary because:
//!
//! 1. `JsRuntime` is `!Send` - cannot be held across `.await` in `TaskTool::execute`
//! 2. `JsRuntime` creation is expensive (~50-200ms per instance)
//! 3. We need O(1) runtimes for process lifetime, not O(n) per execution
//!
//! The `JsRuntimePool` must be created at startup and stored in `ToolAppState`.
//!
//! # Example
//!
//! ```ignore
//! use durable_tools::typescript::{TypeScriptTool, TypeScriptToolInstance, JsRuntimePool};
//!
//! // Create the pool at startup
//! let pool = JsRuntimePool::new_default(tokio::runtime::Handle::current());
//!
//! // Build a TypeScript tool
//! let tool = TypeScriptTool::builder("my_tool")
//!     .description("A tool that searches and summarizes")
//!     .typescript_code(r#"
//!         export default {
//!             name: "my_tool",
//!             description: "Search and summarize",
//!             parameters_schema: {
//!                 type: "object",
//!                 properties: {
//!                     query: { type: "string" }
//!                 },
//!                 required: ["query"]
//!             },
//!             async run(params, sideInfo) {
//!                 const results = await ctx.callTool("search", { q: params.query }, {});
//!                 return { summary: JSON.stringify(results) };
//!             }
//!         };
//!     "#)
//!     .parameters_schema(serde_json::json!({
//!         "type": "object",
//!         "properties": {
//!             "query": { "type": "string" }
//!         },
//!         "required": ["query"]
//!     }))
//!     .build()
//!     .unwrap();
//!
//! // Create an instance for registration
//! let instance = TypeScriptToolInstance::new(tool);
//! ```
//!
//! # Available `ctx` Methods
//!
//! The global `ctx` object in TypeScript provides:
//!
//! - `ctx.taskId()` - Get the current task ID
//! - `ctx.episodeId()` - Get the current episode ID
//! - `ctx.callTool(name, llmParams, sideInfo)` - Call another tool and wait
//! - `ctx.spawnTool(name, llmParams, sideInfo)` - Spawn a background tool
//! - `ctx.joinTool(handle)` - Wait for a spawned tool
//! - `ctx.inference(params)` - Make an inference call
//! - `ctx.rand()` - Get a durable random number
//! - `ctx.now()` - Get the durable current time
//! - `ctx.uuid7()` - Generate a durable UUID
//! - `ctx.sleepFor(name, durationMs)` - Durable sleep
//! - `ctx.awaitEvent(name, timeoutMs?)` - Wait for an event
//! - `ctx.emitEvent(name, payload)` - Emit an event

mod error;
mod ops;
mod pool;
mod runtime;
mod tool;
mod transpile;

pub use error::TypeScriptToolError;
pub use pool::JsRuntimePool;
pub use tool::{TypeScriptTool, TypeScriptToolBuilder, TypeScriptToolInstance};
pub use transpile::transpile_typescript;
