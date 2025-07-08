use std::collections::HashMap;

use crate::TensorZeroError;
use git2::Repository;

#[derive(Debug)]
pub struct GitInfo {
    pub commit_hash: String,
    pub commit_message: Option<String>,
    pub branch: Option<String>,
    pub origin: Option<String>,
    pub untracked_files: bool,
    pub modified_files: bool,
    pub author: Option<String>,
    pub author_email: Option<String>,
}

impl GitInfo {
    /// Grabs the relevant git info for the current repo.
    /// If we can't find the repository or most recent commit, we'll error.
    /// If any other operations fail, we'll return None for that field.
    pub fn new() -> Result<Self, TensorZeroError> {
        // Discover the current repository
        let repo = Repository::discover(".").map_err(|e| TensorZeroError::Git { source: e })?;
        let head = repo
            .head()
            .map_err(|e| TensorZeroError::Git { source: e })?;
        let commit = head
            .peel_to_commit()
            .map_err(|e| TensorZeroError::Git { source: e })?;
        let commit_hash = commit.id().to_string();
        let commit_message = commit.message().map(|s| s.to_string());
        let branch = head.name().map(|s| s.to_string());
        let origin = repo
            .config()
            .ok()
            .and_then(|c| c.get_string("remote.origin.url").ok())
            .map(|s| s.to_string());

        // Get author and email
        let author = commit.author().name().map(|s| s.to_string());
        let author_email = commit.author().email().map(|s| s.to_string());

        // Check for untracked and modified files
        let statuses = repo.statuses(Some(
            git2::StatusOptions::new()
                .include_untracked(true)
                .include_ignored(false)
                .include_unmodified(false),
        ));

        let (untracked_files, modified_files) = if let Ok(statuses) = statuses {
            let mut has_untracked = false;
            let mut has_modified = false;

            for status in statuses.iter() {
                let status_bits = status.status();

                if status_bits.is_wt_new() {
                    has_untracked = true;
                }

                if status_bits.is_wt_modified() || status_bits.is_index_modified() {
                    has_modified = true;
                }

                // If we've found both, we can break early
                if has_untracked && has_modified {
                    break;
                }
            }

            (has_untracked, has_modified)
        } else {
            (false, false)
        };

        Ok(Self {
            commit_hash,
            commit_message,
            branch,
            origin,
            untracked_files,
            modified_files,
            author,
            author_email,
        })
    }

    pub fn into_tags(self) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert("tensorzero::git_commit_hash".to_string(), self.commit_hash);
        if let Some(commit_message) = self.commit_message {
            tags.insert("tensorzero::git_commit_message".to_string(), commit_message);
        }
        if let Some(branch) = self.branch {
            tags.insert("tensorzero::git_branch".to_string(), branch);
        }
        if let Some(origin) = self.origin {
            tags.insert("tensorzero::git_origin".to_string(), origin);
        }
        if let Some(author) = self.author {
            tags.insert("tensorzero::git_author".to_string(), author);
        }
        if let Some(author_email) = self.author_email {
            tags.insert("tensorzero::git_author_email".to_string(), author_email);
        }
        tags.insert(
            "tensorzero::git_untracked_files".to_string(),
            self.untracked_files.to_string(),
        );
        tags.insert(
            "tensorzero::git_modified_files".to_string(),
            self.modified_files.to_string(),
        );
        tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Since we are running in a git repo, we should be able to get the git info
    #[test]
    fn test_git_info() {
        let git_info = GitInfo::new().unwrap();
        assert!(!git_info.commit_hash.is_empty());
        assert!(git_info.commit_message.is_some());
        assert!(git_info.branch.is_some());
        assert!(git_info.origin.is_some());

        let tags = git_info.into_tags();
        assert!(!tags.is_empty());
        assert!(tags.contains_key("tensorzero::git_commit_hash"));
        assert!(tags.contains_key("tensorzero::git_commit_message"));
        assert!(tags.contains_key("tensorzero::git_branch"));
        assert!(tags.contains_key("tensorzero::git_origin"));
        assert!(tags.contains_key("tensorzero::git_author"));
        assert!(tags.contains_key("tensorzero::git_author_email"));
        assert!(tags.contains_key("tensorzero::git_untracked_files"));
        assert!(tags.contains_key("tensorzero::git_modified_files"));
    }
}
