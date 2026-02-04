//! Shared definitions for config path patterns used by tensorzero-core (config loading)
//! and config-writer (config serialization).

/// A component in a path pattern for matching config keys.
#[derive(Debug, Copy, Clone)]
pub enum PathComponent {
    /// Matches a specific key name exactly.
    Literal(&'static str),
    /// Matches any key name (used for user-defined names like function names).
    Wildcard,
}

/// This stores patterns for every possible file path that can be stored in a TensorZero config file.
/// For example, `functions.my_function.system_schema = "some/relative/schema_path.json"` is matched by
/// `&[PathComponent::Literal("functions"), PathComponent::Wildcard, PathComponent::Literal("system_schema")]`
///
/// Any config struct that stores a `ResolvedTomlPath` needs a corresponding entry in this array.
/// If an entry is missing, then deserializing the struct will fail, as the `ResolvedTomlPath` deserializer
/// expects a table produced by `resolve_paths`.
///
/// During config loading, we pre-process the `toml::de::DeTable`, and convert all entries located at
/// `TARGET_PATH_COMPONENTS` (which should be strings) into absolute paths, using the source TOML file
/// as the base path. For example, `functions.my_function.system_schema = "some/relative/schema_path.json"
/// will become `functions.my_function.system_schema = { __tensorzero_remapped_path = "base/directory/some/relative/schema_path.json" }`
///
/// This allows us to abstract over config file globbing, and allow almost all of the codebase to work with
/// absolute paths, without needing to know which TOML file a particular path was originally written in.
///
/// You should avoid declaring a `PathBuf` inside any TensorZero config structs, unless you're certain
/// that the path should not be relative to the TOML file that it's written in.
///
/// One alternative we considered was use `Spanned<PathBuf>` in our config structs, and deserialize from
/// a `toml::de::DeTable`. Unfortunately, this breaks whenever serde has an internal 'boundary'
/// (internally-tagged enums, `#[serde(flatten)]`, and possible other attributes). In these cases, serde
/// will deserialize into its own custom type (consuming the original `Deserializer`), and continue
/// deserializing with the internal serde `Deserializer`. This causes any extra information to get
/// lost (including the span information held by the `toml::de::DeTable` deserializer).
/// While it might be possible to work around this (similar to what we do for error messages in
/// the `TensorZeroDeserialize` macro), this is a load-bearing part of the codebase, and implicitly
/// depends on internal Serde implementation details.
pub static TARGET_PATH_COMPONENTS: &[&[PathComponent]] = &[
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("output_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("output_schema"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("schemas"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_instructions"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("templates"),
        PathComponent::Wildcard,
        PathComponent::Literal("path"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("system_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("user_template"),
    ],
    &[
        PathComponent::Literal("evaluations"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluators"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("assistant_template"),
    ],
    &[
        PathComponent::Literal("tools"),
        PathComponent::Wildcard,
        PathComponent::Literal("parameters"),
    ],
    &[
        PathComponent::Literal("gateway"),
        PathComponent::Literal("template_filesystem_access"),
        PathComponent::Literal("base_path"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("fuser"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("user"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("assistant"),
    ],
    &[
        PathComponent::Literal("functions"),
        PathComponent::Wildcard,
        PathComponent::Literal("variants"),
        PathComponent::Wildcard,
        PathComponent::Literal("evaluator"),
        PathComponent::Literal("input_wrappers"),
        PathComponent::Literal("system"),
    ],
];

/// Returns the expected file extension for a given config key.
///
/// This is used by config-writer to determine what file extension to use when
/// writing file content to disk.
///
/// # Returns
/// - `Some(".minijinja")` for keys ending in `_template` or wrapper keys (`user`, `system`, `assistant`)
/// - `Some(".json")` for keys ending in `_schema` or `parameters`
/// - `Some(".txt")` for `system_instructions`
/// - `None` for `path` (preserve original extension) or `base_path` (directory, no extension)
pub fn file_extension_for_key(key: &str) -> Option<&'static str> {
    if key.ends_with("_template") {
        Some(".minijinja")
    } else if key.ends_with("_schema") || key == "parameters" {
        Some(".json")
    } else if key == "system_instructions" {
        Some(".txt")
    } else if key == "path" || key == "base_path" {
        // `path` preserves original extension, `base_path` is a directory
        None
    } else if key == "user" || key == "system" || key == "assistant" {
        // input_wrappers keys are minijinja templates
        Some(".minijinja")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_extension_for_template_keys() {
        assert_eq!(
            file_extension_for_key("system_template"),
            Some(".minijinja"),
            "system_template should use .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("user_template"),
            Some(".minijinja"),
            "user_template should use .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("assistant_template"),
            Some(".minijinja"),
            "assistant_template should use .minijinja extension"
        );
    }

    #[test]
    fn test_file_extension_for_schema_keys() {
        assert_eq!(
            file_extension_for_key("system_schema"),
            Some(".json"),
            "system_schema should use .json extension"
        );
        assert_eq!(
            file_extension_for_key("user_schema"),
            Some(".json"),
            "user_schema should use .json extension"
        );
        assert_eq!(
            file_extension_for_key("output_schema"),
            Some(".json"),
            "output_schema should use .json extension"
        );
        assert_eq!(
            file_extension_for_key("parameters"),
            Some(".json"),
            "parameters should use .json extension"
        );
    }

    #[test]
    fn test_file_extension_for_system_instructions() {
        assert_eq!(
            file_extension_for_key("system_instructions"),
            Some(".txt"),
            "system_instructions should use .txt extension"
        );
    }

    #[test]
    fn test_file_extension_for_path_keys() {
        assert_eq!(
            file_extension_for_key("path"),
            None,
            "path should preserve original extension (return None)"
        );
        assert_eq!(
            file_extension_for_key("base_path"),
            None,
            "base_path is a directory and should return None"
        );
    }

    #[test]
    fn test_file_extension_for_input_wrapper_keys() {
        assert_eq!(
            file_extension_for_key("user"),
            Some(".minijinja"),
            "input_wrappers.user should use .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("system"),
            Some(".minijinja"),
            "input_wrappers.system should use .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("assistant"),
            Some(".minijinja"),
            "input_wrappers.assistant should use .minijinja extension"
        );
    }

    #[test]
    fn test_file_extension_for_unknown_keys() {
        assert_eq!(
            file_extension_for_key("unknown_key"),
            None,
            "unknown keys should return None"
        );
        assert_eq!(
            file_extension_for_key("model"),
            None,
            "model should return None"
        );
    }
}
