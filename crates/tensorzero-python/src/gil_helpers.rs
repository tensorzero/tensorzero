use std::future::Future;

use pyo3::{Python, marker::Ungil};
/// Runs a function inside the Tokio runtime, with the GIL released.
/// This is used when we need to drop a TensorZero client (or a type that holds it),
/// so that we can block on the ClickHouse batcher shutting down, without holding the GIL.
pub fn in_tokio_runtime_no_gil(f: Box<dyn FnOnce() + Send + '_>) {
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
/// We do not allow access to the underlying value, since this makes it easy to
/// accidentally clone the underlying `T` and then drop it from somewhere else,
/// bypassing the `DropInTokio` wrapper.
///
/// The intended use-case of `DropInTokio` is to wrap 'drop handles' that don't
/// need to be accessed to implement Python methods, but perform some kind of
/// cleanup when they go out of scope.
pub struct DropInTokio<T: Send> {
    value: T,
    make_dummy: fn() -> T,
}

impl<T: Send> DropInTokio<T> {
    /// Constructs a new `DropInTokio` wrapper, which will drop `value`
    /// inside of the Tokio runtime with the python GIL released.
    ///
    /// The `make_dummy` function is called to produce a new value,
    /// which is needed to satisfy the borrow checker. It will *not* be
    /// dropped inside of Tokio, so it should not contain a `Client`
    /// or similar TensorZero handle.
    pub fn new(value: T, make_dummy: fn() -> T) -> Self {
        Self { value, make_dummy }
    }
}

impl<T: Send> Drop for DropInTokio<T> {
    fn drop(&mut self) {
        let dummy = (self.make_dummy)();
        let inner_value = std::mem::replace(&mut self.value, dummy);
        in_tokio_runtime_no_gil(Box::new(|| {
            drop(inner_value);
        }));
    }
}

/// Calls `tokio::Runtime::block_on` without holding the Python GIL.
/// This is used when we call into pure-Rust code from the synchronous `TensorZeroGateway`
/// We don't need (or want) to hold the GIL when the Rust client code is running,
/// since it doesn't need to interact with any Python objects.
/// This allows other Python threads to run while the current thread is blocked on the Rust execution.
pub fn tokio_block_on_without_gil<F: Future + Send>(py: Python<'_>, fut: F) -> F::Output
where
    F::Output: Ungil,
{
    // The Tokio runtime is managed by `pyo3_async_runtimes` - the entrypoint to
    // our crate (`python`) is the `pymodule` function, rather than
    // a `#[tokio::main]` function, so we need `pyo3_async_runtimes` to keep track of
    // a Tokio runtime for us.
    py.detach(|| pyo3_async_runtimes::tokio::get_runtime().block_on(fut))
}
