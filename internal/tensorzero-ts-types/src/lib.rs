/// A pre-computed bundle of TypeScript type declarations.
///
/// Contains all declarations needed to fully define a type and its transitive dependencies,
/// concatenated in dependency order with import statements stripped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TsTypeBundle(pub &'static str);

impl TsTypeBundle {
    /// Returns the TypeScript declarations as a string slice.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for TsTypeBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::ops::Deref for TsTypeBundle {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}

/// Bundle for the unit type `()`, used when a tool has no meaningful output.
pub const UNIT: TsTypeBundle = TsTypeBundle("type Unit = null;");

// Include the generated constants
include!(concat!(env!("OUT_DIR"), "/ts_bundles.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    // This test will only pass after `pnpm build-bindings` has been run.
    // It verifies that at least one bundle was generated and is non-empty.
    #[test]
    fn test_bundles_are_generated() {
        // Check that a known LlmParams bundle exists and contains the type name
        assert!(
            !INFERENCE_TOOL_PARAMS.is_empty(),
            "INFERENCE_TOOL_PARAMS bundle should not be empty"
        );
        assert!(
            INFERENCE_TOOL_PARAMS.contains("InferenceToolParams"),
            "INFERENCE_TOOL_PARAMS bundle should contain the type name"
        );
    }

    #[test]
    fn test_bundle_contains_dependencies() {
        // InferenceToolParams depends on Input, InferenceParams, DynamicToolParams, etc.
        // Verify that at least one dependency is included
        assert!(
            INFERENCE_TOOL_PARAMS.contains("Input"),
            "INFERENCE_TOOL_PARAMS bundle should contain the Input dependency"
        );
    }

    #[test]
    fn test_display_impl() {
        let bundle = TsTypeBundle("type Foo = string;");
        assert_eq!(format!("{bundle}"), "type Foo = string;");
    }

    #[test]
    fn test_deref_impl() {
        let bundle = TsTypeBundle("type Foo = string;");
        assert_eq!(&*bundle, "type Foo = string;");
        assert!(bundle.contains("Foo"));
    }
}
