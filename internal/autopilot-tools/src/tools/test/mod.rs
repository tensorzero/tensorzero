//! Test tools for autopilot e2e testing.
//!
//! This module contains tools used for testing the autopilot infrastructure.
//! All tools in this module are only available when the `e2e_tests` feature is enabled.

// TaskTools
mod echo;
mod failing;
mod flaky;
mod panic;
mod slow;

// SimpleTools
mod error_simple;
mod good_simple;
mod slow_simple;

// TaskTool exports
pub use echo::{EchoOutput, EchoParams, EchoTool};
pub use failing::{FailingTool, FailingToolParams};
pub use flaky::{FlakyTool, FlakyToolOutput, FlakyToolParams};
pub use panic::{PanicTool, PanicToolParams};
pub use slow::{SlowTool, SlowToolOutput, SlowToolParams};

// SimpleTool exports
pub use error_simple::{ErrorSimpleParams, ErrorSimpleTool};
pub use good_simple::{GoodSimpleOutput, GoodSimpleParams, GoodSimpleTool};
pub use slow_simple::{SlowSimpleOutput, SlowSimpleParams, SlowSimpleTool};
