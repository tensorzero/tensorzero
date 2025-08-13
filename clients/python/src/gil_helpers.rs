use std::{ops::Deref, sync::Arc};

use pyo3::Python;
use tensorzero_rust::InferenceStream;
use tokio::sync::Mutex;

/// Runs a function inside the Tokio runtime, with the GIL released.
/// This is used when we need to drop a TensorZero client (or a type that holds it),
/// so that we can block on the ClickHouse batcher shutting down, without holding the GIL.
pub fn in_tokio_runtime_no_gil<F: FnOnce() + Send>(f: F) {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
            f();
        });
    });
}

/// Helper method to drop an `InferenceStream` inside a Tokio runtime.
/// An `InferenceStream` holds a reference to a `Client`, and thus may need to block
/// inside Tokio when it gets dropped
fn drop_stream_in_tokio(stream: &mut Arc<Mutex<InferenceStream>>) {
    if let Some(mutex) = Arc::get_mut(stream) {
        let real_stream = std::mem::replace(mutex, Mutex::new(Box::pin(futures::stream::empty())));
        in_tokio_runtime_no_gil(|| {
            drop(real_stream);
        });
    }
}

/// A wrapper type for an `InferenceStream`, which ensures that we enter the Tokio runtime
/// when dropping it (if it holds the last reference to the `Arc`, which can cause us to
/// actually drop the underlying `InferenceStream`).
///
/// We do not allow access to the underling `Arc`, to prevent accidentally cloning
/// the `Arc` and dropping it from somewhere else within pyo3 code.
/// This is not an issue within `tensorzero-core`, since we're always in the Tokio runtime.
#[derive(Clone)]
pub struct TokioInferenceStream(Arc<Mutex<InferenceStream>>);

impl TokioInferenceStream {
    pub fn new(stream: InferenceStream) -> Self {
        Self(Arc::new(Mutex::new(stream)))
    }
}

impl Deref for TokioInferenceStream {
    // This intentionally does not allow dereferencing the `Arc` itself,
    // since it could then be cloned and dropped from somewhere else within pyo3 code.
    type Target = Mutex<InferenceStream>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for TokioInferenceStream {
    fn drop(&mut self) {
        drop_stream_in_tokio(&mut self.0);
    }
}
