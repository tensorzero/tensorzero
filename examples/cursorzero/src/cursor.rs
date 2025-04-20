use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::convert::AsRef;
use std::path::{Path, PathBuf};
use tensorzero_internal::inference::types::{
    ContentBlockChatOutput, ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent,
};
/*
This file handles the outputs of inferences from Cursor. We handle two cases:
Cursor Ask (from the sidebar where you ask a question and there are also code completions included).
Cursor Edit (Cmd-K from the code directly).
*/

#[derive(Debug)]
pub struct CursorCodeBlock {
    pub language_extension: String,
    pub path: PathBuf, // normalized to the git root
    pub content: String,
}

/// Route the output to the appropriate parser based on the system message.
/// This could be avoided if we could hardcoded different TensorZero functions for different tasks.
/// Instead, we use the content of the system message to determine which parser to use.
/// This will be brittle to Cursor changing their system prompt.
pub fn parse_cursor_output<P: AsRef<Path>>(
    input: &ResolvedInput,
    output: &Vec<ContentBlockChatOutput>,
    git_root_path: P,
) -> Result<Vec<CursorCodeBlock>> {
    let Some(Value::String(system)) = &input.system else {
        return Err(anyhow::anyhow!("No system message found"));
    };
    let output_text = match output.as_slice() {
        [ContentBlockChatOutput::Text(t)] => &t.text,
        _ => {
            return Err(anyhow::anyhow!("Output is not a single text block"));
        }
    };
    if system.contains("rewrite a piece of code") {
        return parse_cursor_edit_output(&input.messages, output_text, git_root_path);
    }
    if system.contains("pair programming with a USER to solve their coding task") {
        return parse_cursor_ask_output(system, output_text, git_root_path);
    }
    Err(anyhow::anyhow!(
        "System message doesn't fit our expected format"
    ))
}

/// Parse Cursor Ask output.
///
/// According to their system prompt, the output format for code blocks in Cursor is:
/// ```language:path/to/file
/// // ... existing code ...
/// {{ edit_1 }}
/// // ... existing code ...
/// {{ edit_2 }}
/// // ... existing code ...
/// ```
/// which might be inside of a larger string that contains other text as well.
/// In this file, we take a particular Cursor output (as a &str)
/// and return a Vec<CursorCodeBlock>.
///
/// Since the cursor root and the git root might be different, we also normalize the paths to the git root.
///
/// The system message for Cursor also contains as a substring:
///
/// <user_info>
/// The user's OS version is darwin 24.3.0. The absolute path of the user's workspace is /Users/viraj/tensorzero/tensorzero/examples/cursorzero. The user's shell is /bin/zsh.
/// </user_info>
///
/// We should grab the workspace path from this string and use it to normalize the paths of the code blocks.
fn parse_cursor_ask_output<P: AsRef<Path>>(
    system: &str,
    output_text: &str,
    git_root_path: P,
) -> Result<Vec<CursorCodeBlock>> {
    // 1. Extract the workspace path from the system string.
    //    e.g. "...workspace is /Users/viraj/.../examples/cursorzero."
    let workspace_path = extract_workspace_path(system)
        .ok_or_else(|| anyhow::anyhow!("No workspace path found in system message"))?;

    // 2. Find all ```lang:rel/path/to/file\n...``` blocks in the output.
    //    We use (?s) so `.` matches newlines, and make the match non‑greedy.
    let block_re =
        Regex::new(r"(?s)```(?P<lang>[^:\n]+):(?P<file>[^\n]+)\r?\n(?P<content>.*?)```")?;

    let mut blocks = Vec::new();
    for cap in block_re.captures_iter(output_text) {
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

/// Parse Cursor Edit output.
///
/// To get the file name: The second user message seems to consistently contain
/// the header ## Selection to Rewrite and then a code block on the next line.
///
/// For example:
/// ...
// ## Selection to Rewrite
// ```src/cursor.rs
//     // Start of Selection
//     let system = match system {
//         Some(Value::String(s)) => s,
//         _ => return Err(anyhow::any
/// ...
///
/// Therefore, we use a regex to extract the file name from the second user message.
///
/// To get the code generated:
/// The generated output seems to always look like this:
/// ```
///     // Start of Selection
///     let Some(Value::String(system)) = system else {
///         return Err(anyhow::anyhow!("No system message found"));
///     };
///     // End of Selection
/// ```
/// where the code block is the code generated by Cursor.
/// Therefore, we can simply extract the code block from the output and remove the comments.
/// This is equivalent to taking the whole string except for the first two lines and the last 2 lines.
fn parse_cursor_edit_output<P: AsRef<Path>>(
    messages: &Vec<ResolvedInputMessage>,
    output_text: &str,
    git_root_path: P,
) -> Result<Vec<CursorCodeBlock>> {
    // 1. Extract the workspace path from the first message string.
    //    e.g. "...workspace is /Users/viraj/.../examples/cursorzero."
    let first_message = &messages[0];
    let first_message_text = match first_message.content.as_slice() {
        [ResolvedInputMessageContent::Text { value: text }] => {
            let Value::String(text) = text else {
                return Err(anyhow::anyhow!("Expected text in first user message"));
            };
            text
        }
        _ => {
            return Err(anyhow::anyhow!("Expected text in first user message"));
        }
    };
    let workspace_path = extract_workspace_path(first_message_text)
        .ok_or_else(|| anyhow::anyhow!("No workspace path found in system message"))?;
    // There should be 2 messages in the input:
    if messages.len() != 2 {
        return Err(anyhow::anyhow!(
            "Expected 2 messages in input, got {}",
            messages.len()
        ));
    }
    let second_message = &messages[1];
    let second_message_text = match second_message.content.as_slice() {
        [ResolvedInputMessageContent::Text { value: text }] => {
            let Value::String(text) = text else {
                return Err(anyhow::anyhow!("Expected text in second user message"));
            };
            text
        }
        _ => {
            return Err(anyhow::anyhow!("Expected text in second user message"));
        }
    };
    // 2. Extract the relative path from the second user message.
    let file_re = Regex::new(r"## Selection to Rewrite\r?\n```(?P<file>[^\n]+)")?;
    let file_ref = file_re
        .captures(second_message_text)
        .and_then(|cap| cap.name("file"))
        .map(|m| m.as_str())
        .ok_or_else(|| anyhow::anyhow!("Couldn't find file reference in edit selection"))?;
    // 3. Extract the generated code from the output.
    //    We simply strip the first two lines and the last two lines of the output.
    let lines: Vec<&str> = output_text.lines().collect();

    // Check if there are enough lines to remove the first two and last two.
    if lines.len() < 4 {
        // If there are fewer than 4 lines, removing the first 2 and last 2 is not possible
        // or results in an empty/negative range. Return an error or an empty block?
        // Let's return an error for clarity, as the expected format isn't met.
        return Err(anyhow::anyhow!(
            "Output text has fewer than 4 lines ({} lines), cannot remove first 2 and last 2 lines. Output: '{}'",
            lines.len(),
            output_text
        ));
    }
    // Extract the code content by skipping the first two and last two lines.
    let code_content_lines = &lines[2..lines.len() - 2];
    let content = code_content_lines.join("\n");
    // 4. Get the git root relative path from the workspace path.
    let abs_path = workspace_path.join(file_ref);
    let relative_path = match abs_path.strip_prefix(git_root_path.as_ref()) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => abs_path.clone(),
    };
    let language_extension = relative_path
        .extension()
        .map(|ext| ext.to_string_lossy())
        .ok_or_else(|| anyhow::anyhow!("No extension found in file reference"))?
        .to_string();

    let code_block = CursorCodeBlock {
        language_extension,
        // Using canonicalize might be better if file_ref could be non-canonical
        // path: git_root_path.as_ref().join(relative_path).canonicalize()?,
        // For now, just store the relative path as extracted.
        path: relative_path,
        content,
    };

    // Return a Vec containing this single block
    Ok(vec![code_block])
}

/// Extracts the absolute workspace path from the first <user_info> block.
///
/// Returns `Some(path)` if found, or `None` otherwise.
fn extract_workspace_path(system: &str) -> Option<PathBuf> {
    // (?s) makes `.` match newlines; we non‑greedily skip up to our marker,
    // then capture everything up to the next literal period.
    let re =
        Regex::new(r"(?s)<user_info>.*?The absolute path of the user's workspace is\s+([^\.]+)\.")
            .unwrap();

    re.captures(system)
        .and_then(|caps| caps.get(1))
        .map(|m| PathBuf::from(m.as_str()))
}
