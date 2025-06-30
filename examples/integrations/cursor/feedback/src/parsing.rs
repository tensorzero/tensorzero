use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};
use tree_sitter::{Language, Parser, Tree};

static LANGUAGES: OnceLock<RwLock<HashMap<String, Language>>> = OnceLock::new();

fn get_language_for_extension(ext: &str) -> Result<Language> {
    let lock = LANGUAGES.get_or_init(|| RwLock::new(HashMap::new()));

    // Fast path: shared read-lock
    if let Ok(guard) = lock.read() {
        if let Some(lang) = guard.get(ext) {
            return Ok(lang.clone());
        }
    }

    // Slow path: upgrade to exclusive write-lock
    let mut w = lock
        .write()
        .map_err(|_| anyhow::anyhow!("Failed to lock languages"))?;
    if let Some(lang) = w.get(ext) {
        return Ok(lang.clone());
    }

    let lang = match ext {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "toml" => tree_sitter_toml_ng::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "md" => tree_sitter_md::LANGUAGE.into(),
        _ => return Err(anyhow::anyhow!("Unsupported file extension: {}", ext)),
    };
    w.insert(ext.to_string(), lang);
    if let Some(lang) = w.get(ext) {
        Ok(lang.clone())
    } else {
        Err(anyhow::anyhow!(
            "Failed to insert language for extension: {}",
            ext
        ))
    }
}

pub fn parse_hunk(hunk: &str, hunk_file_extension: &str) -> Result<Tree> {
    let language = get_language_for_extension(hunk_file_extension)?;
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let tree = parser
        .parse(hunk, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse hunk: {}", hunk))?;
    Ok(tree)
}
