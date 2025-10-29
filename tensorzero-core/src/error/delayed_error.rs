//! The API for this module is intentionally restrictive - please read this entire comment before adding anything
//! * All of the fields are private - we want to force this to be used through our public methods
//! * The only way to extract an `Error` is through `log` or `log_at_level` -
//!   this enforces the invariant that an `Error` outside of this module has already been logged
//! * The `reported` field cannot be set directly, to ensure that the only way to skip logging is
//!   to explicitly call `suppress_logging_of_error_message`
//! * We do *not* implement `Into<Error> for DelayedError` or `From<DelayedError> for Error`,
//!   to prevent using `?` on functions that return
use std::fmt;
use std::fmt::Debug;

use crate::error::{Error, ErrorDetails};

pub struct DelayedError {
    inner: Error,
    // When a `DelayedError` is dropped, we'll log the error unless:
    // * We already logged it via `log`/`log_at_level`
    // * We explicitly suppressed logging via `suppress_logging_of_error_message`
    reported: bool,
}

// The 'reported' field isn't useful to display in `Debug` impls, since
// the output can end up in user-facing error message when `debug = true` is set in the config.
impl Debug for DelayedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl DelayedError {
    pub fn new(details: ErrorDetails) -> Self {
        Self {
            inner: Error::new_without_logging(details),
            reported: false,
        }
    }

    pub fn get_details(&self) -> &ErrorDetails {
        self.inner.get_details()
    }

    pub fn log(mut self) -> Error {
        self.reported = true;
        self.inner.log();
        // Error stores an `Arc`, so it's fine to clone
        self.inner.clone()
    }

    pub fn log_at_level(mut self, prefix: &str, level: tracing::Level) -> Error {
        self.reported = true;
        self.inner.log_at_level(prefix, level);
        // Error stores an `Arc`, so it's fine to clone
        self.inner.clone()
    }

    pub fn suppress_logging_of_error_message(mut self) -> String {
        self.reported = true;
        self.inner.to_string()
    }
}

impl Drop for DelayedError {
    fn drop(&mut self) {
        if !self.reported {
            self.inner.log();
        }
    }
}
