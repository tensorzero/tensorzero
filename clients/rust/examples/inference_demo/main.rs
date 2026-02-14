#![expect(clippy::expect_used, clippy::print_stdout)]

use std::{io::Write, path::PathBuf};

use tensorzero::{
    ClientBuilder, ClientBuilderMode, ClientInferenceParams, ContentBlockChunk, InferenceOutput,
    InferenceResponseChunk, Input, InputMessage, InputMessageContent, PostgresConfig, Role,
};
use tensorzero_core::inference::types::Template;
use tokio_stream::StreamExt;

use clap::Parser;
use url::Url;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to tensorzero.toml. This runs the client in embedded gateway mode.
    #[arg(short, long)]
    config_file: Option<PathBuf>,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs the client in HTTP gateway mode.
    #[arg(short, long)]
    gateway_url: Option<Url>,

    /// Whether or not to print streaming output
    #[arg(short, long, default_value_t = false)]
    streaming: bool,

    /// Name of the tensorzero function to call
    #[arg(short, long)]
    function_name: String,

    /// Input to the function
    input: String,
}

#[tokio::main]
async fn main() {
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to initialize tracing");

    let args = Args::parse();

    let client = match (args.gateway_url, args.config_file) {
        (Some(gateway_url), None) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        (None, Some(config_file)) => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(config_file),
            clickhouse_url: std::env::var("TENSORZERO_CLICKHOUSE_URL").ok(),
            postgres_config: std::env::var("TENSORZERO_POSTGRES_URL")
                .ok()
                .map(PostgresConfig::Url),
            valkey_url: std::env::var("TENSORZERO_VALKEY_URL").ok(),
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: false,
        }),
        (Some(_), Some(_)) => {
            tracing::error!("Cannot specify both gateway URL and config path");
            #[expect(clippy::disallowed_methods)]
            std::process::exit(1);
        }
        (None, None) => {
            tracing::error!("Gateway URL or config path is required");
            #[expect(clippy::disallowed_methods)]
            std::process::exit(1);
        }
    }
    .build()
    .await
    .expect("Failed to build client");

    let input = serde_json::from_str(&args.input).expect("Failed to parse input");

    let res = client
        .inference(ClientInferenceParams {
            function_name: Some(args.function_name),
            stream: Some(args.streaming),
            input: Input {
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: input,
                    })],
                }],
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .expect("Failed to run inference");
    match res {
        InferenceOutput::NonStreaming(data) => {
            tracing::info!("Inference output: {:?}", data);
        }
        InferenceOutput::Streaming(mut stream) => {
            let mut stdout = std::io::stdout().lock();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => match chunk {
                        InferenceResponseChunk::Chat(c) => {
                            for content in c.content {
                                if let ContentBlockChunk::Text(t) = content {
                                    write!(stdout, "{}", t.text)
                                        .expect("Failed to write to stdout");
                                    stdout.flush().expect("Failed to flush stdout");
                                }
                            }
                        }
                        InferenceResponseChunk::Json(c) => {
                            write!(stdout, "{}", c.raw).expect("Failed to write to stdout");
                            stdout.flush().expect("Failed to flush stdout");
                        }
                    },
                    Err(e) => {
                        tracing::error!("Error when reading streaming chunk: {:?}", e);
                    }
                }
            }
            println!();
        }
    }
}
