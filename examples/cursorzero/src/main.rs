use anyhow::Result;
use clap::Parser;
use cursorzero::{
    git::{get_diff_by_file, get_last_commit_from_repo},
    parsing::parse_hunk,
};
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
    let diffs = get_diff_by_file(&repo, &commit)?;

    for (file, diffs) in diffs {
        println!("File: {}", file);
        for diff in diffs {
            println!("  {}", diff.content);
            let tree = parse_hunk(&diff.content, &file.split('.').last().unwrap())?;
            println!("  {:?}", tree);
        }
    }

    Ok(())
}
