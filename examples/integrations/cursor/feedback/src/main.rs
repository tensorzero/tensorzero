//! This file is intended to run as a post-commit hook for the cursorzero application.
//!
//! Steps performed in `main`:
//! 1. Parse command-line arguments (repository path and gateway URL).
//! 2. Discover the Git repository at the given path.
//! 3. Retrieve the latest commit and its parent's timestamp interval.
//! 4. Generate diffs for each file in the commit.
//! 5. For each diff hunk, parse its content into a syntax tree.
//! 6. Compute tree-edit-distance metrics and collect inference data.
//! 7. Send collected inferences to an external service via HTTP gateway.
//!
//! By running automatically after each commit, this hook enables continuous
//! code-change analysis and integration with TensorZero.

use anyhow::Result;
use clap::Parser;
use cursorzero::{
    git::{get_commit_timestamp_and_parent_timestamp, get_diff_by_file, get_last_commit_from_repo},
    process_and_send_feedback, process_diffs, process_inferences,
};
use git2::Repository;
use tensorzero::{ClientBuilder, ClientBuilderMode};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use url::Url;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
    #[clap(long, default_value = "http://localhost:13000")]
    gateway_url: Url,
    #[clap(long)]
    user: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing();
    let args = Cli::parse();
    let repo = Repository::discover(args.path)?;
    let commit = get_last_commit_from_repo(&repo)?;
    let commit_interval = get_commit_timestamp_and_parent_timestamp(&commit)?;
    let diffs = get_diff_by_file(&repo, &commit)?;
    let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: args.gateway_url,
    })
    .build()
    .await?;

    let diff_trees = process_diffs(diffs)?;
    let normalized_inference_trees = process_inferences(&repo, commit_interval, args.user).await?;

    let (num_feedbacks_sent, inferences_with_feedback) =
        process_and_send_feedback(normalized_inference_trees, &diff_trees, &client).await?;

    #[expect(clippy::print_stdout)]
    {
        println!("Number of feedbacks sent: {num_feedbacks_sent}");
        println!("Number of inferences with feedback: {inferences_with_feedback}");
    }
    Ok(())
}

fn setup_tracing() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .finish();
    #[expect(clippy::expect_used)]
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global default subscriber");
}
