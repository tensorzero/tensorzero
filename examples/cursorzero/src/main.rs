use anyhow::Result;
use clap::Parser;
use cursorzero::get_last_commit_from_repo;
use git2::Repository;
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
    println!(
        "Found last commit with message: {}",
        commit.message().unwrap()
    );

    Ok(())
}
