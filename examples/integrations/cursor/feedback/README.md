# Feedback via post-commit hook

This is very early-stage work and should be considered experimental.

In this directory we include a Rust project that uses tree-sitter to parse diff hunks into syntax trees and the Zhang–Shasha algorithm to compute tree-edit-distance metrics:

1. Discovers the Git repository at a given path
2. Retrieves the latest commit and its parent’s timestamp interval
3. Generates diffs for each file in the commit
4. Parses each diff hunk into a tree-sitter syntax tree
5. Computes Zhang–Shasha tree-edit-distance between code changes and AI-generated inferences
6. Sends these metrics to TensorZero as feedback, helping evaluate how closely AI suggestions match actual code changes

We also include an example Git hook in `post-commit.example` (pointing at the `build` directory).
This approach works best with frequent commits, giving more granular insights into which TensorZero variants lead to merged code.

To install it:

- run `cargo install --path .` from this directory.
- copy `post-commit.example` to your root `.git/hooks` for the repo you'd like to install this in.
- set environment variable `CURSORZERO_CLICKHOUSE_URL` to point at the ClickHouse database you're using to store inferences from the docker compose in the parent directory.

There’s plenty of room for improvement both in implementation details and the overall strategy.
This example is under active development -- please expect changes!
