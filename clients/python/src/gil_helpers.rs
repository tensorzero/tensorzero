use std::{ops::Deref, sync::Arc};

use pyo3::Python;
/// Runs a function inside the Tokio runtime, with the GIL released.
/// This is used when we need to drop a TensorZero client (or a type that holds it),
/// so that we can block on the ClickHouse batcher shutting down, without holding the GIL.
fn in_tokio_runtime_no_gil<F: FnOnce() + Send>(f: F) {
    Python::attach(|py| {
        py.detach(|| {
            let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
            f();
        });
    });
}

/// A wrapper type for an `InferenceStream`, which ensures that we enter the Tokio runtime
/// when dropping it (if it holds the last reference to the `Arc`, which can cause us to
/// actually drop the underlying `InferenceStream`).
///
/// We do not allow access to the underlying `Arc`, to prevent accidentally cloning
/// the `Arc` and dropping it from somewhere else within pyo3 code.
/// This is not an issue within `tensorzero-core`, since we're always in the Tokio runtime.
pub struct DropInTokio<T: Send> {
    value: Arc<T>,
    make_dummy: fn() -> T,
}

impl<T: Send> DropInTokio<T> {
    /// Constructs a new `DropInTokio` wrapper, which will drop `value`
    /// inside of the Tokio runtime with the python GIL released.
    ///
    /// The `make_dummy` function is called to produce a new value,
    /// which is needed to satisfy the borrow checker. It will *not* be
    /// dropped inside of Tokio, so it should shouldn't contain a `Client`
    /// or similar TensorZero handle.
    pub fn new(value: T, make_dummy: fn() -> T) -> Self {
        Self {
            value: Arc::new(value),
            make_dummy,
        }
    }
}

impl<T: Send> Clone for DropInTokio<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            make_dummy: self.make_dummy,
        }
    }
}

impl<T: Send> Deref for DropInTokio<T> {
    // This intentionally does not allow dereferencing the `Arc` itself,
    // since it could then be cloned and dropped from somewhere else within pyo3 code.
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Send> Drop for DropInTokio<T> {
    fn drop(&mut self) {
        let dummy = (self.make_dummy)();
        // 'self.value' may hold a reference `Client`, and thus may need to block
        // inside Tokio when it gets dropped
        if let Some(value_ref) = Arc::get_mut(&mut self.value) {
            let inner_value = std::mem::replace(value_ref, dummy);
            in_tokio_runtime_no_gil(|| {
                drop(inner_value);
            });
        }
    }
}
