//! Harvest block-level TOML comments at file-mode parse time and convert
//! them to per-object metadata.
//!
//! Block-level only — comments attached to a `[functions.foo]` header or
//! to a key-value line directly inside that table are captured and become
//! `metadata.notes` on the corresponding per-object row. Trailing comments
//! after a value (`temperature = 0.7  # tuned`) and mid-array comments
//! are intentionally dropped: capturing them adds parser complexity for
//! marginal value, and the UI's Notes textarea is the right place for
//! commentary that doesn't belong inline with config values.
//!
//! See the `tests` module for the precise capture rules.

use std::collections::BTreeMap;

use tensorzero_stored_config::PerObjectMetadata;

/// Parse `toml_input` and produce per-object metadata containing the
/// harvested block-level comments at `metadata.notes`.
///
/// Returns `Ok(PerObjectMetadata)` even when `toml_input` has no comments
/// — the result is just empty in that case, simplifying caller code.
/// Returns `Err` only when the TOML doesn't parse; the caller is
/// expected to surface that to the user in the same error path it would
/// for any other unparseable config.
pub fn harvest_block_comments(toml_input: &str) -> Result<PerObjectMetadata, toml_edit::TomlError> {
    let doc: toml_edit::DocumentMut = toml_input.parse()?;

    let mut out: PerObjectMetadata = BTreeMap::new();
    walk_table(doc.as_table(), Path::root(), &mut out);
    Ok(out)
}

/// A canonical-path builder. Mirrors the slash-separated paths used in
/// `PerObjectMetadata` keys (e.g. `"functions/foo"`,
/// `"functions/foo/variants/bar"`, `"models/gpt-4o"`,
/// `"gateway"` for singletons).
#[derive(Clone, Debug, Default)]
struct Path {
    segments: Vec<String>,
}

impl Path {
    fn root() -> Self {
        Self::default()
    }

    fn push(&self, segment: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(segment.to_string());
        Self { segments }
    }

    fn to_canonical_string(&self) -> String {
        self.segments.join("/")
    }

    fn is_object_path(&self) -> bool {
        // Object paths are non-empty; an empty path means "the root
        // document" which doesn't correspond to any per-object row.
        !self.segments.is_empty()
    }
}

/// Recursively walk a `toml_edit::Table`, attaching its leading
/// (decor-prefix) comments to the metadata map under the table's
/// canonical path.
fn walk_table(table: &toml_edit::Table, path: Path, out: &mut PerObjectMetadata) {
    if path.is_object_path()
        && let Some(notes) = leading_comments(table.decor().prefix())
    {
        attach_notes(out, &path.to_canonical_string(), notes);
    }

    for (key, item) in table {
        let child_path = path.push(key);
        match item {
            toml_edit::Item::Table(t) => walk_table(t, child_path, out),
            toml_edit::Item::ArrayOfTables(arr) => {
                // For `[[rate_limiting.rules]]` style: each table
                // element gets the same path as the array, with a `[N]`
                // suffix. Keeps the harvester simple — the caller can
                // route to whichever subtree it needs.
                for (idx, t) in arr.iter().enumerate() {
                    let elem_path = child_path.push(&format!("[{idx}]"));
                    walk_table(t, elem_path, out);
                }
            }
            // Scalars and inline tables are leaves: their leading
            // comments (if any) are intentionally not associated with a
            // per-object row, because they don't define a row themselves.
            // The block-comments rule means only `[section]` headers and
            // their immediate descendants count.
            _ => {}
        }
    }
}

/// Extract block-level (`#`-prefixed) lines from a `toml_edit::RawString`
/// and concatenate them into a single newline-joined notes string.
///
/// `RawString::as_str()` returns the raw whitespace-and-comments span
/// preceding the item; we grep for `#`-prefixed lines and strip the
/// leading `#` and at most one space so the resulting note doesn't
/// carry TOML syntax artifacts.
fn leading_comments(prefix: Option<&toml_edit::RawString>) -> Option<String> {
    let raw = prefix?.as_str()?;
    let mut notes: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim_start();
        if let Some(comment) = trimmed.strip_prefix('#') {
            // Strip a single leading space so `# notes` becomes `notes`,
            // but `#double-space  notes` becomes ` notes` (preserving
            // intentional indentation).
            let cleaned = comment.strip_prefix(' ').unwrap_or(comment);
            notes.push(cleaned.to_string());
        }
    }
    if notes.is_empty() {
        None
    } else {
        Some(notes.join("\n"))
    }
}

/// Attach `notes` to the metadata entry for `path`, creating the entry
/// if needed and overwriting any prior `notes` value.
fn attach_notes(out: &mut PerObjectMetadata, path: &str, notes: String) {
    let entry = out.entry(path.to_string()).or_default();
    entry.notes = Some(notes);
}

/// Set `metadata.created_by` and `metadata.source_file` on every entry
/// in `out` to the same value. Used by the file-mode loader so every
/// harvested comment also carries provenance.
pub fn stamp_provenance(
    out: &mut PerObjectMetadata,
    created_by: Option<&str>,
    source_file: Option<&str>,
) {
    for value in out.values_mut() {
        if let Some(creator) = created_by {
            value.created_by = Some(creator.to_string());
        }
        if let Some(file) = source_file {
            value.source_file = Some(file.to_string());
        }
    }
}

/// Ensure that every `path` in `paths` has at least an entry in `out`,
/// even if no block-comment was captured for it. Used so callers that
/// stamp provenance per object via `stamp_provenance` can also stamp
/// objects that had no comments.
pub fn ensure_entries<'a, I>(out: &mut PerObjectMetadata, paths: I)
where
    I: IntoIterator<Item = &'a str>,
{
    for path in paths {
        out.entry(path.to_string()).or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harvest(toml_input: &str) -> PerObjectMetadata {
        harvest_block_comments(toml_input).expect("fixture should parse")
    }

    #[test]
    fn captures_block_comment_above_section_header() {
        let toml_input = r#"
# This is the chat function for the assistant.
# It handles conversation flow.
[functions.assistant]
type = "chat"
"#;
        let m = harvest(toml_input);
        let entry = m
            .get("functions/assistant")
            .expect("assistant entry should exist");
        assert_eq!(
            entry.notes.as_deref(),
            Some("This is the chat function for the assistant.\nIt handles conversation flow.")
        );
    }

    #[test]
    fn captures_block_comment_above_nested_section() {
        let toml_input = r#"
[functions.assistant]
type = "chat"

# baseline variant — uses gpt-4o
[functions.assistant.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4o"
"#;
        let m = harvest(toml_input);
        let entry = m
            .get("functions/assistant/variants/baseline")
            .expect("variant entry should exist");
        assert_eq!(
            entry.notes.as_deref(),
            Some("baseline variant — uses gpt-4o")
        );
    }

    #[test]
    fn skips_trailing_comments_after_values() {
        // The comment after `temperature = 0.7` is intentionally NOT
        // captured — block-level only. The variant section has no
        // leading comments either, so the variant entry should not
        // appear in the map at all.
        let toml_input = r#"
[functions.assistant.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4o"
temperature = 0.7  # tuned
"#;
        let m = harvest(toml_input);
        assert!(!m.contains_key("functions/assistant/variants/baseline"));
    }

    #[test]
    fn handles_no_comments() {
        let toml_input = r#"
[functions.assistant]
type = "chat"

[models.gpt-4o]
routing = ["openai"]
"#;
        let m = harvest(toml_input);
        assert!(m.is_empty(), "no block comments should produce empty map");
    }

    #[test]
    fn array_of_tables_indexes_each_element() {
        let toml_input = r#"
[rate_limiting]

# strict tier
[[rate_limiting.rules]]
window = "1m"
limit = 100

# burst tier
[[rate_limiting.rules]]
window = "1s"
limit = 10
"#;
        let m = harvest(toml_input);
        assert_eq!(
            m.get("rate_limiting/rules/[0]")
                .and_then(|e| e.notes.as_deref()),
            Some("strict tier"),
        );
        assert_eq!(
            m.get("rate_limiting/rules/[1]")
                .and_then(|e| e.notes.as_deref()),
            Some("burst tier"),
        );
    }

    #[test]
    fn invalid_toml_returns_error_rather_than_empty_map() {
        // A naked equals sign is not valid TOML — caller can't tell
        // "no comments" from "broken file" if we silently swallow.
        let result = harvest_block_comments("= broken\n");
        assert!(result.is_err());
    }

    #[test]
    fn stamp_provenance_fills_creator_and_source_for_every_entry() {
        let toml_input = r#"
# function notes
[functions.f]
type = "chat"

# model notes
[models.m]
routing = ["openai"]
"#;
        let mut m = harvest(toml_input);
        stamp_provenance(&mut m, Some("file-import"), Some("/tmp/foo.toml"));

        let f = m.get("functions/f").expect("f");
        assert_eq!(f.created_by.as_deref(), Some("file-import"));
        assert_eq!(f.source_file.as_deref(), Some("/tmp/foo.toml"));
        assert_eq!(f.notes.as_deref(), Some("function notes"));

        let model = m.get("models/m").expect("m");
        assert_eq!(model.created_by.as_deref(), Some("file-import"));
        assert_eq!(model.source_file.as_deref(), Some("/tmp/foo.toml"));
    }

    #[test]
    fn ensure_entries_creates_default_metadata_for_paths_without_comments() {
        let mut m = PerObjectMetadata::new();
        ensure_entries(&mut m, ["functions/a", "models/b"]);
        assert!(m.contains_key("functions/a"));
        assert!(m.contains_key("models/b"));
        assert!(
            m.get("functions/a").unwrap().is_empty(),
            "default entry should be empty",
        );
    }
}
