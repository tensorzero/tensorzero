#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::missing_panics_doc
)]
use std::{
    io,
    sync::{Mutex, MutexGuard, OnceLock},
};

use lazy_static::lazy_static;
use serde_json::json;
use tensorzero_inference_types::{
    AllowedTools, AllowedToolsChoice, FunctionToolDef, ProviderToolCallConfig,
};
use tensorzero_types::ToolChoice;
use tracing_subscriber::{FmtSubscriber, fmt::MakeWriter};

lazy_static! {
    pub static ref WEATHER_TOOL_DEF: FunctionToolDef = FunctionToolDef {
        name: "get_temperature".to_string(),
        description: "Get the current temperature in a given location".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }),
        strict: false,
    };
    pub static ref QUERY_TOOL_DEF: FunctionToolDef = FunctionToolDef {
        name: "query_articles".to_string(),
        description: "Query articles from Wikipedia".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "year": {"type": "integer"}
            },
            "required": ["query", "year"]
        }),
        strict: true,
    };
    pub static ref WEATHER_PROVIDER_TOOL_CONFIG: ProviderToolCallConfig = ProviderToolCallConfig {
        tools: vec![WEATHER_TOOL_DEF.clone()],
        provider_tools: vec![],
        openai_custom_tools: vec![],
        tool_choice: ToolChoice::Specific("get_temperature".to_string()),
        parallel_tool_calls: None,
        allowed_tools: AllowedTools {
            tools: Vec::new(),
            choice: AllowedToolsChoice::FunctionDefault,
        },
    };
    pub static ref MULTI_PROVIDER_TOOL_CONFIG: ProviderToolCallConfig = ProviderToolCallConfig {
        tools: vec![WEATHER_TOOL_DEF.clone(), QUERY_TOOL_DEF.clone()],
        provider_tools: vec![],
        openai_custom_tools: vec![],
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: Some(true),
        allowed_tools: AllowedTools {
            tools: Vec::new(),
            choice: AllowedToolsChoice::FunctionDefault,
        },
    };
    pub static ref WEATHER_TOOL_CONFIG: ProviderToolCallConfig =
        WEATHER_PROVIDER_TOOL_CONFIG.clone();
    pub static ref MULTI_TOOL_CONFIG: ProviderToolCallConfig = MULTI_PROVIDER_TOOL_CONFIG.clone();
}

// ---- Test log capture utilities ----

const TEST_LOG_FILTER: &str = "gateway=trace,tensorzero_core=trace,tensorzero_providers=trace,warn";

static GLOBAL_BUF: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

pub fn capture_logs() -> impl Fn(&str) -> bool {
    capture_logs_with_filter(TEST_LOG_FILTER)
}

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

pub fn reset_capture_logs() {
    println!("Called reset_capture_logs");
    GLOBAL_BUF.get().unwrap().lock().unwrap().clear();
}

#[derive(Debug)]
pub struct MockWriter<'a> {
    buf: &'a Mutex<Vec<u8>>,
}

impl<'a> MockWriter<'a> {
    pub fn new(buf: &'a Mutex<Vec<u8>>) -> Self {
        Self { buf }
    }

    fn buf(&self) -> io::Result<MutexGuard<'a, Vec<u8>>> {
        self.buf
            .lock()
            .map_err(|_| io::Error::from(io::ErrorKind::Other))
    }
}

impl<'a> io::Write for MockWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut target = self.buf()?;
        print!("{}", String::from_utf8(buf.to_vec()).unwrap());
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

fn install_subscriber(mock_writer: MockWriter<'static>, env_filter: &str) {
    FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(mock_writer)
        .with_level(true)
        .with_ansi(false)
        .init();
}
