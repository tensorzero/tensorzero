#![expect(clippy::expect_used, clippy::print_stdout)]

use std::{io::Write, path::PathBuf, sync::Arc, time::Duration};

use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, ClientInput,
    ClientInputMessage, ClientInputMessageContent, ContentBlockChunk, InferenceOutput,
    InferenceResponseChunk, Role,
};
use tensorzero_internal::inference::types::TextKind;
use tokio::{runtime::Runtime, sync::Semaphore, task::JoinSet};
use tokio_stream::StreamExt;

use clap::Parser;
use url::Url;
use uuid::Uuid;

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

    /// Name of the variant to call
    #[arg(short, long)]
    variant_name: Option<String>,

    /// Input to the function
    input: String,

    #[arg(long)]
    count: usize,

    #[arg(long)]
    max_inflight: usize,
}

async fn run_inference(client: &Client, args: &Args) {
    let input = &args.input;

    let res: InferenceOutput = client
        .inference(ClientInferenceParams {
            function_name: Some(args.function_name.clone()),
            variant_name: args.variant_name.clone(),
            stream: Some(args.streaming),
            input: ClientInput {
                system: Some(serde_json::json!({
                    "assistant_name": "TensorZeroBot",
                })),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: format!("{input} : Random {}", Uuid::now_v7()),
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
            //tracing::info!("Inference output: {:?}", data);
        }
        InferenceOutput::Streaming(mut stream) => {
            //let mut stdout = std::io::stdout().lock();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => match chunk {
                        InferenceResponseChunk::Chat(c) => {
                            /*for content in c.content {
                                if let ContentBlockChunk::Text(t) = content {
                                    write!(stdout, "{}", t.text)
                                        .expect("Failed to write to stdout");
                                    stdout.flush().expect("Failed to flush stdout");
                                }
                            }*/
                        }
                        InferenceResponseChunk::Json(c) => {
                            //write!(stdout, "{}", c.raw).expect("Failed to write to stdout");
                            //stdout.flush().expect("Failed to flush stdout");
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

async fn main_inner() {
    console_subscriber::init();
    //let subscriber = tracing_subscriber::FmtSubscriber::new();
   // tracing::subscriber::set_global_default(subscriber).expect("Failed to initialize tracing");

    let args = Arc::new(Args::parse());

    let client = match (&args.gateway_url, &args.config_file) {
        (Some(gateway_url), None) => ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: gateway_url.clone(),
        }),
        (None, Some(config_file)) => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(config_file.clone()),
            clickhouse_url: std::env::var("TENSORZERO_CLICKHOUSE_URL").ok(),
            timeout: None,
        }),
        (Some(_), Some(_)) => {
            std::process::exit(1);
        }
        (None, None) => {
            tracing::error!("Gateway URL or config path is required");
            std::process::exit(1);
        }
    }
    .build()
    .await
    .expect("Failed to build client");

    let client = Arc::new(client);

    let mut pbar = tqdm::pbar(Some(args.count));
    let max_inflight = args.max_inflight;
    let semaphore = Arc::new(Semaphore::new(max_inflight));
    let handle = tokio::runtime::Handle::current();
    for i in 0..args.count {
        let args = args.clone();
        let client = client.clone();
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        tokio::spawn(async move {
            let permit = permit;
            run_inference(&client, &args).await;
            drop(permit);
        });
        pbar.update(1).unwrap();
        /*if i % 20000 == 0 {
            tracing::info!("Running tasks: {}", handle.metrics().num_alive_tasks());
        }*/
    }
    pbar.close().unwrap();
}

fn main() {
    let runtime = Runtime::new().expect("Failed to create runtime");
    runtime.block_on(main_inner());
    loop {
        let active_tasks = runtime.metrics().num_alive_tasks();
        println!("Active tasks: {}", active_tasks);
        if runtime.metrics().num_alive_tasks() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    println!("Done!");
}
