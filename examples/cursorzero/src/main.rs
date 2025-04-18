use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use clap::Parser;
use cursorzero::{
    clickhouse::get_inferences_in_time_range,
    git::{get_commit_timestamp_and_parent_timestamp, get_diff_by_file, get_last_commit_from_repo},
    parsing::parse_hunk,
};
use git2::Repository;
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(short, long, default_value = ".")]
    path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let repo = Repository::discover(args.path)?;
    let commit = get_last_commit_from_repo(&repo)?;
    let commit_interval = get_commit_timestamp_and_parent_timestamp(&commit)?;
    println!(
        "Found last commit with message: {}",
        commit.message().unwrap()
    );
    let diffs = get_diff_by_file(&repo, &commit)?;

    for (file, diffs) in diffs {
        println!("File: {}", file);
        for diff in diffs {
            println!("  {}", diff.content);
            let tree = parse_hunk(&diff.content, &file.split('.').last().unwrap())?;
            println!("  {:?}", tree);
        }
    }
    let clickhouse_url = std::env::var("CURSORZERO_CLICKHOUSE_URL")?;
    let clickhouse = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let inferences = get_inferences_in_time_range(&clickhouse, commit_interval).await?;
    println!("Found {} inferences", inferences.len());
    Ok(())
}
