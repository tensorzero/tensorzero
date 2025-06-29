use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use git2::{Commit, DiffDelta, DiffOptions, Repository};

/// Gets the last commit from a repository
pub fn get_last_commit_from_repo(repo: &Repository) -> Result<Commit<'_>> {
    let head_ref = repo.head()?.resolve()?;
    let oid = head_ref
        .target()
        .ok_or(anyhow!("HEAD is not pointing to a commit"))?;
    let commit = repo.find_commit(oid)?;
    Ok(commit)
}

#[derive(Debug)]
pub struct DiffAddition {
    /// 1-based line number in the **new** file where the addition starts
    pub start_line: usize,
    /// 1-based line number in the **new** file where the addition ends
    pub end_line: usize,
    /// The content that was added
    pub content: String,
}

/// Given a commit, get the diff by file
pub fn get_diff_by_file(
    repo: &Repository,
    commit: &Commit,
) -> Result<HashMap<PathBuf, Vec<DiffAddition>>> {
    // Use the commit tree and its parent tree if there is one to get the diff
    let tree = commit.tree()?;
    let parent_tree = commit.parent(0).ok().map(|p| p.tree()).transpose()?;
    // Set up diff options
    // There are many other flags here (https://docs.rs/git2/latest/git2/struct.DiffOptions.html)
    // We may want to use some of them in the future
    let mut diff_options = DiffOptions::new();
    diff_options.ignore_submodules(true);
    let diff =
        repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut diff_options))?;

    struct DiffWalkState {
        current_file: Option<PathBuf>,
        current_chunk: Option<Chunk>,
        result: HashMap<PathBuf, Vec<DiffAddition>>,
    }
    let state = Rc::new(RefCell::new(DiffWalkState {
        current_file: None,
        current_chunk: None,
        result: HashMap::new(),
    }));
    let state_file = Rc::clone(&state);
    let state_chunk = Rc::clone(&state);

    diff.foreach(
        &mut move |delta: DiffDelta, _| {
            let mut s = state_file.borrow_mut();
            // When a new file starts, flush the previous file's last chunk.
            if let (Some(file), Some(chunk)) = (s.current_file.take(), s.current_chunk.take()) {
                s.result.entry(file).or_default().push(DiffAddition {
                    start_line: chunk.start,
                    end_line: chunk.end,
                    content: chunk.buf,
                });
            }

            // `delta.new_file().path()` is *None* for pure deletions,
            // so skip those straight away.
            s.current_file = delta.new_file().path().map(PathBuf::from);
            // return `true` to keep iterating
            true
        },
        None,
        None,
        Some(&mut |_delta, _hunk, line| {
            let mut s = state_chunk.borrow_mut();
            let s = &mut *s;
            if line.origin() != '+' {
                // A non‑'+' line ends any run of additions
                if let (Some(file), Some(chunk)) = (s.current_file.as_ref(), s.current_chunk.take())
                {
                    s.result
                        .entry(file.clone())
                        .or_default()
                        .push(DiffAddition {
                            start_line: chunk.start,
                            end_line: chunk.end,
                            content: chunk.buf,
                        });
                }
                return true; // keep going
            }

            let new_lineno = line.new_lineno().unwrap_or(0) as usize; // libgit2 uses 1‑based
            let content = std::str::from_utf8(line.content()).unwrap_or_default();

            match &mut s.current_chunk {
                Some(chunk) if new_lineno == chunk.last_seen_lineno + 1 => {
                    // immediately‑following addition: extend current chunk
                    chunk.end = new_lineno;
                    chunk.last_seen_lineno = new_lineno;
                    chunk.buf.push_str(content);
                }
                _ => {
                    // gap → flush the old chunk (if any) and start new one
                    if let (Some(file), Some(chunk)) =
                        (s.current_file.as_ref(), s.current_chunk.take())
                    {
                        s.result
                            .entry(file.clone())
                            .or_default()
                            .push(DiffAddition {
                                start_line: chunk.start,
                                end_line: chunk.end,
                                content: chunk.buf,
                            });
                    }
                    s.current_chunk = Some(Chunk {
                        start: new_lineno,
                        end: new_lineno,
                        last_seen_lineno: new_lineno,
                        buf: content.to_owned(),
                    });
                }
            }
            true
        }),
    )?;
    drop(state_chunk);
    let Ok(state) = Rc::try_unwrap(state) else {
        return Err(anyhow!("Failed to unwrap state"));
    };
    let DiffWalkState {
        mut result,
        mut current_file,
        mut current_chunk,
    } = state.into_inner();
    // Flush the very last chunk
    if let (Some(file), Some(chunk)) = (current_file.take(), current_chunk.take()) {
        result.entry(file).or_default().push(DiffAddition {
            start_line: chunk.start,
            end_line: chunk.end,
            content: chunk.buf,
        });
    }

    Ok(result)
}

struct Chunk {
    start: usize,
    end: usize,
    buf: String,
    last_seen_lineno: usize,
}

pub struct CommitInterval {
    pub commit_timestamp: DateTime<Utc>,
    pub parent_timestamp: Option<DateTime<Utc>>,
}

pub fn get_commit_timestamp_and_parent_timestamp(commit: &Commit) -> Result<CommitInterval> {
    let commit_time_unix = commit.time().seconds();
    let commit_timestamp = Utc
        .timestamp_opt(commit_time_unix, 0)
        .single()
        .ok_or_else(|| anyhow!("Failed to convert commit timestamp"))?;
    let parent_timestamp = commit.parent(0).ok().and_then(|p| {
        let parent_time_unix = p.time().seconds();
        Utc.timestamp_opt(parent_time_unix, 0).single()
    });
    Ok(CommitInterval {
        commit_timestamp,
        parent_timestamp,
    })
}

/// Resolves a VSCode workpace relative path to a vector of git-relative paths.
pub fn find_paths_in_repo<P: AsRef<Path>>(repo: &Repository, path: &P) -> Result<Vec<PathBuf>> {
    // Search the repo for the path that suffixes with the given path
    let mut paths = Vec::new();
    let index = repo.index()?;
    let workdir = repo
        .workdir()
        .with_context(|| "Failed to get git repository workdir")?;
    for entry in index.iter() {
        let tracked = Path::new(std::str::from_utf8(&entry.path)?);
        // Prepend the workdir when comparing - if we're in a repository named 'foo',
        // then the 'tracked' will be path like 'my/src/file.rs'.
        // If Cursor is run from the root of the repository, then Cursor path will look like
        // 'foo/my/src/file.rs'.
        if workdir.join(tracked).ends_with(path) {
            paths.push(tracked.to_path_buf());
        }
    }
    Ok(paths)
}
