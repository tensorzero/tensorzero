/*
This file handles the outputs of inferences from Cursor.
According to their system prompt, the output format for code blocks in Cursor is:
```language:path/to/file
// ... existing code ...
{{ edit_1 }}
// ... existing code ...
{{ edit_2 }}
// ... existing code ...
```

which might be inside of a larger string that contains other text as well.
In this file, we take a particular Cursor output (as a &str)
and return a Vec<CursorCodeBlock>.

Since the cursor root and the git root might be different, we also normalize the paths to the git root.

The system message for Cursor also contains as a substring:

<user_info>
The user's OS version is darwin 24.3.0. The absolute path of the user's workspace is /Users/viraj/tensorzero/tensorzero/examples/cursorzero. The user's shell is /bin/zsh.
</user_info>

We should grab the workspace path from this string and use it to normalize the paths of the code blocks.
*/
use anyhow::Result;
use regex::Regex;
use std::convert::AsRef;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct CursorCodeBlock {
    pub language_extension: String,
    pub path: PathBuf, // normalized to the git root
    pub content: String,
}

pub fn parse_cursor_output<P: AsRef<Path>>(
    system: &str,
    output: &str,
    git_root_path: P,
) -> Result<Vec<CursorCodeBlock>> {
    // 1. Extract the workspace path from the system string.
    //    e.g. "...workspace is /Users/viraj/.../examples/cursorzero."
    let workspace_path = {
        let re = Regex::new(r"absolute path of the user's workspace is (?P<ws>\S+)\.")?;
        if let Some(cap) = re.captures(system) {
            PathBuf::from(&cap["ws"])
        } else {
            // Fallback: assume paths in code blocks are relative to git root
            tracing::warn!("No workspace path found in system message, using git root");
            git_root_path.as_ref().to_path_buf()
        }
    };

    // 2. Find all ```lang:rel/path/to/file\n...``` blocks in the output.
    //    We use (?s) so `.` matches newlines, and make the match non‑greedy.
    let block_re =
        Regex::new(r"(?s)```(?P<lang>[^:\n]+):(?P<file>[^\n]+)\r?\n(?P<content>.*?)```")?;

    let mut blocks = Vec::new();
    for cap in block_re.captures_iter(output) {
        let language_extension = cap["lang"].to_string();
        let file_ref = &cap["file"];
        let content = cap["content"].to_string();

        // 3. Turn the code‑block file reference into an absolute path.
        let abs_path = {
            let p = Path::new(file_ref);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                workspace_path.join(p)
            }
        };

        // 4. Strip the git_root_path prefix to get a repo‑relative path.
        let normalized = match abs_path.strip_prefix(git_root_path.as_ref()) {
            Ok(relative) => relative.to_path_buf(),
            Err(_) => abs_path.clone(),
        };

        blocks.push(CursorCodeBlock {
            language_extension,
            path: normalized,
            content,
        });
    }

    Ok(blocks)
}
