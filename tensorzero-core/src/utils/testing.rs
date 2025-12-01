#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::missing_panics_doc
)]
use std::{
    io,
    sync::{Mutex, MutexGuard, OnceLock},
};

use tracing_subscriber::{fmt::MakeWriter, FmtSubscriber};

// This can be tweaked as needed if we want to capture more or less output
// in tests that use `capture_logs`
const TEST_LOG_FILTER: &str = "gateway=trace,tensorzero_core=trace,warn";

static GLOBAL_BUF: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

pub fn capture_logs() -> impl Fn(&str) -> bool {
    capture_logs_with_filter(TEST_LOG_FILTER)
}

/// A replacement for the `tracing_test` crate, specialized for our needs.
/// * We install a global subscriber without any per-function filtering.
///   All of our tests run under 'cargo nextest', which runs tests in separate processes (instead of separate threads)
///   This allows us to capture all output (including from `tokio::spawn` tasks) without worrying about capturing output from unrelated tests
///   that happen to be running correctly in the same process.
///   Since we don't need the function name, this no longer needs to be a macro.
///  * We customize the filter to exclude annoying output from crates like `hyper`
pub fn capture_logs_with_filter(filter: &str) -> impl Fn(&str) -> bool {
    GLOBAL_BUF
        .set(Mutex::new(Vec::new()))
        .expect("Called `capture_logs` more than once");
    install_subscriber(MockWriter::new(GLOBAL_BUF.get().unwrap()), filter);

    move |message: &str| {
        let logs = String::from_utf8(GLOBAL_BUF.get().unwrap().lock().unwrap().to_vec()).unwrap();
        logs.split('\n').any(|line| line.contains(message))
    }
}

pub fn get_captured_logs() -> String {
    String::from_utf8(GLOBAL_BUF.get().unwrap().lock().unwrap().to_vec()).unwrap()
}

/// Clears the current global buffer of captured logs.
/// You should idelaly create a separate `#[tokio::test]` function instead of using this function,
/// as it results in confusing log output when the tests fail.
pub fn reset_capture_logs() {
    println!("Called reset_capture_logs");
    GLOBAL_BUF.get().unwrap().lock().unwrap().clear();
}

// Copied from https://github.com/dbrgn/tracing-test/blob/cf7fe8c7a90eb36f00023237ae98928e7cd768e0/tracing-test/src/subscriber.rs#L11 (MIT-licensed)

/// A fake writer that writes into a buffer (behind a mutex).
#[derive(Debug)]
pub struct MockWriter<'a> {
    buf: &'a Mutex<Vec<u8>>,
}

impl<'a> MockWriter<'a> {
    /// Create a new `MockWriter` that writes into the specified buffer (behind a mutex).
    pub fn new(buf: &'a Mutex<Vec<u8>>) -> Self {
        Self { buf }
    }

    /// Give access to the internal buffer (behind a `MutexGuard`).
    fn buf(&self) -> io::Result<MutexGuard<'a, Vec<u8>>> {
        // Note: The `lock` will block. This would be a problem in production code,
        // but is fine in tests.
        self.buf
            .lock()
            .map_err(|_| io::Error::from(io::ErrorKind::Other))
    }
}

impl<'a> io::Write for MockWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Lock target buffer
        let mut target = self.buf()?;

        // Write to stdout in order to show up in tests
        print!("{}", String::from_utf8(buf.to_vec()).unwrap());

        // Write to buffer
        target.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buf()?.flush()
    }
}

impl<'a> MakeWriter<'_> for MockWriter<'a> {
    type Writer = Self;

    fn make_writer(&self) -> Self::Writer {
        MockWriter::new(self.buf)
    }
}

/// Installs a new subscriber that writes to the specified [`MockWriter`].
///
/// [`MockWriter`]: struct.MockWriter.html
fn install_subscriber(mock_writer: MockWriter<'static>, env_filter: &str) {
    FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(mock_writer)
        .with_level(true)
        .with_ansi(false)
        .init();
}
