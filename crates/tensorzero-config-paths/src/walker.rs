//! A generic walker for `TARGET_PATH_COMPONENTS` patterns over a mutable TOML-like tree.
//!
//! Both config loading (in `tensorzero-core`) and config serialization (in `config-applier`)
//! need to walk a parsed TOML tree along the patterns in [`TARGET_PATH_COMPONENTS`](super::TARGET_PATH_COMPONENTS),
//! visiting every leaf value (e.g. `functions.my_func.variants.v1.system_template`) and performing
//! some action on it. The two crates use different underlying TOML types (`toml::Value` vs
//! `toml_edit::Item`), so this module abstracts the walking logic behind the [`TomlTreeMut`] /
//! [`TomlTableMut`] traits and drives it with a [`TomlPathVisitor`].
//!
//! The walker always matches entire patterns, with `PathComponent::Wildcard` enumerating every
//! key of the current table. Callers can customise behaviour by overriding visitor methods:
//!
//! - [`TomlPathVisitor::visit_leaf`] — required. Called at terminal nodes with the full path.
//! - [`TomlPathVisitor::visit_wildcard_key`] — optional. Called before descending into each
//!   wildcard-matched key. Use this for things like path-traversal validation on user-provided names.
//! - [`TomlPathVisitor::visit_non_table`] — optional. Called when the walker expected a table but
//!   found something else. By default this returns [`WalkError::ExpectedTable`]; callers that want
//!   to silently skip malformed subtrees can override it to return `Ok(())`.

use thiserror::Error;

use crate::{PathComponent, TARGET_PATH_COMPONENTS};

/// Errors emitted directly by the walker. Consumers provide a `From<WalkError>` impl for their
/// own error type so these can be propagated through their visitor.
#[derive(Debug, Error)]
pub enum WalkError {
    #[error("`{path}`: Expected a table, found {found}")]
    ExpectedTable { path: String, found: String },
    #[error("`{path}`: Path cannot end with a wildcard")]
    WildcardAtEnd { path: String },
}

/// A mutable TOML-like value that can be asked for its table contents and its type name.
pub trait TomlTreeMut {
    type Table: TomlTableMut<Value = Self>;

    fn as_table_mut(&mut self) -> Option<&mut Self::Table>;

    /// A short human-readable name for this value's variant (e.g. `"string"`, `"integer"`).
    fn type_name(&self) -> &'static str;
}

/// A mutable TOML-like table keyed by string.
pub trait TomlTableMut {
    type Value: TomlTreeMut<Table = Self>;

    /// Snapshot of the keys in this table. Returned as owned strings so callers can hold them
    /// across mutable accesses to the table.
    fn keys(&self) -> Vec<String>;

    fn get_mut(&mut self, key: &str) -> Option<&mut Self::Value>;
}

/// Hook interface invoked while walking a pattern tree.
pub trait TomlPathVisitor<V: TomlTreeMut + ?Sized> {
    type Error: From<WalkError>;

    /// Called at the terminal node of a pattern. `path` is the full dotted path from the root
    /// (including any wildcard-resolved keys and any `initial_path` the caller supplied).
    fn visit_leaf(&mut self, value: &mut V, path: &[String]) -> Result<(), Self::Error>;

    /// Called after a wildcard key has been pushed onto `path` but before we descend into it.
    /// Override this to reject certain keys (e.g. path-traversal sequences). The default is a
    /// no-op.
    fn visit_wildcard_key(&mut self, _path: &[String]) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Called when the walker expected to descend into a table but found another variant.
    /// Returning `Ok(())` silently skips this subtree; returning `Err` aborts the walk.
    /// The default returns [`WalkError::ExpectedTable`].
    fn visit_non_table(&mut self, path: &[String], found_type: &str) -> Result<(), Self::Error> {
        Err(WalkError::ExpectedTable {
            path: path.join("."),
            found: found_type.to_string(),
        }
        .into())
    }
}

/// Walk every config target-path pattern in [`TARGET_PATH_COMPONENTS`] over `root`, calling
/// the visitor's hooks as described in [`TomlPathVisitor`].
pub fn walk_target_paths<V, Visitor>(
    root: &mut V,
    visitor: &mut Visitor,
) -> Result<(), Visitor::Error>
where
    V: TomlTreeMut,
    Visitor: TomlPathVisitor<V>,
{
    walk_patterns(root, TARGET_PATH_COMPONENTS, visitor)
}

/// Walk every config target-path pattern in [`TARGET_PATH_COMPONENTS`] whose prefix matches
/// `matched_prefix`. The matched prefix is prepended to the path reported to the visitor.
pub fn walk_target_paths_from_prefix<V, Visitor>(
    entry: &mut V,
    matched_prefix: &[&str],
    visitor: &mut Visitor,
) -> Result<(), Visitor::Error>
where
    V: TomlTreeMut,
    Visitor: TomlPathVisitor<V>,
{
    for pattern in TARGET_PATH_COMPONENTS {
        if pattern_matches_prefix(pattern, matched_prefix) && pattern.len() > matched_prefix.len() {
            let suffix = &pattern[matched_prefix.len()..];
            walk_pattern(entry, suffix, matched_prefix, visitor)?;
        }
    }
    Ok(())
}

/// Walk every pattern in `patterns` over `root`, calling the visitor's hooks as described
/// in [`TomlPathVisitor`].
pub fn walk_patterns<V, Visitor>(
    root: &mut V,
    patterns: &[&[PathComponent]],
    visitor: &mut Visitor,
) -> Result<(), Visitor::Error>
where
    V: TomlTreeMut,
    Visitor: TomlPathVisitor<V>,
{
    for pattern in patterns {
        walk_pattern(root, pattern, &[], visitor)?;
    }
    Ok(())
}

fn pattern_matches_prefix(pattern: &[PathComponent], prefix: &[&str]) -> bool {
    if pattern.len() < prefix.len() {
        return false;
    }
    pattern
        .iter()
        .zip(prefix.iter())
        .all(|(component, key)| match component {
            PathComponent::Wildcard => true,
            PathComponent::Literal(lit) => *lit == *key,
        })
}

/// Walk a single pattern starting from `entry`, with `initial_path` prepended to the path
/// reported to the visitor. This is useful when the caller has already descended to a subtree
/// of the config and wants the visitor to see the full dotted path.
pub fn walk_pattern<V, Visitor>(
    entry: &mut V,
    pattern: &[PathComponent],
    initial_path: &[&str],
    visitor: &mut Visitor,
) -> Result<(), Visitor::Error>
where
    V: TomlTreeMut,
    Visitor: TomlPathVisitor<V>,
{
    let mut path: Vec<String> = initial_path.iter().map(|s| (*s).to_string()).collect();
    walk_recursive(entry, pattern, &mut path, visitor)
}

fn walk_recursive<V, Visitor>(
    entry: &mut V,
    pattern: &[PathComponent],
    path: &mut Vec<String>,
    visitor: &mut Visitor,
) -> Result<(), Visitor::Error>
where
    V: TomlTreeMut,
    Visitor: TomlPathVisitor<V>,
{
    let Some(first) = pattern.first() else {
        return Ok(());
    };

    match first {
        PathComponent::Literal(literal) => {
            let found_type = entry.type_name();
            let Some(table) = entry.as_table_mut() else {
                return visitor.visit_non_table(path, found_type);
            };

            let Some(nested) = table.get_mut(literal) else {
                return Ok(());
            };

            path.push((*literal).to_string());
            let result = if pattern.len() == 1 {
                visitor.visit_leaf(nested, path)
            } else {
                walk_recursive(nested, &pattern[1..], path, visitor)
            };
            path.pop();
            result
        }
        PathComponent::Wildcard => {
            if pattern.len() == 1 {
                return Err(WalkError::WildcardAtEnd {
                    path: path.join("."),
                }
                .into());
            }

            let found_type = entry.type_name();
            let Some(table) = entry.as_table_mut() else {
                return visitor.visit_non_table(path, found_type);
            };

            let keys = table.keys();
            for key in keys {
                path.push(key.clone());

                let visit_result = visitor.visit_wildcard_key(path);
                if visit_result.is_err() {
                    path.pop();
                    return visit_result;
                }

                let recurse_result = if let Some(nested) = table.get_mut(&key) {
                    walk_recursive(nested, &pattern[1..], path, visitor)
                } else {
                    Ok(())
                };

                path.pop();
                recurse_result?;
            }
            Ok(())
        }
    }
}

// ============================================================================
// Impls: toml::Value / toml::Table
// ============================================================================

impl<'a> TomlTreeMut for toml::Spanned<toml::de::DeValue<'a>> {
    type Table = toml::de::DeTable<'a>;

    fn as_table_mut(&mut self) -> Option<&mut <Self as TomlTreeMut>::Table> {
        match self.get_mut() {
            toml::de::DeValue::Table(table) => Some(table),
            _ => None,
        }
    }

    fn type_name(&self) -> &'static str {
        self.get_ref().type_str()
    }
}

impl<'a> TomlTableMut for toml::de::DeTable<'a> {
    type Value = toml::Spanned<toml::de::DeValue<'a>>;

    fn keys(&self) -> Vec<String> {
        self.keys().map(|key| key.get_ref().to_string()).collect()
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Self::Value> {
        toml::de::DeTable::get_mut(self, key)
    }
}

impl TomlTreeMut for toml::Value {
    type Table = toml::Table;

    fn as_table_mut(&mut self) -> Option<&mut <Self as TomlTreeMut>::Table> {
        toml::Value::as_table_mut(self)
    }

    fn type_name(&self) -> &'static str {
        toml::Value::type_str(self)
    }
}

impl TomlTableMut for toml::Table {
    type Value = toml::Value;

    fn keys(&self) -> Vec<String> {
        self.keys().cloned().collect()
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Self::Value> {
        toml::Table::get_mut(self, key)
    }
}

// ============================================================================
// Impls: toml_edit::Item / toml_edit::Table
// ============================================================================

impl TomlTreeMut for toml_edit::Item {
    type Table = toml_edit::Table;

    fn as_table_mut(&mut self) -> Option<&mut <Self as TomlTreeMut>::Table> {
        toml_edit::Item::as_table_mut(self)
    }

    fn type_name(&self) -> &'static str {
        toml_edit::Item::type_name(self)
    }
}

impl TomlTableMut for toml_edit::Table {
    type Value = toml_edit::Item;

    fn keys(&self) -> Vec<String> {
        self.iter().map(|(k, _)| k.to_string()).collect()
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Self::Value> {
        toml_edit::Table::get_mut(self, key)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use googletest::prelude::*;
    // Re-shadow the `Result` alias from `googletest::prelude` with the standard one, so
    // the visitor trait impls below use `std::result::Result<_, TestError>`.
    use std::result::Result;

    use super::*;

    // ========================================================================
    // pattern_matches_prefix tests
    // ========================================================================

    #[test]
    fn test_pattern_matches_prefix_exact_match() {
        let pattern = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("variants"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let prefix = &["functions", "my_func", "variants", "v1"];
        assert!(
            pattern_matches_prefix(pattern, prefix),
            "pattern should match prefix with wildcard substitution"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_literal_mismatch() {
        let pattern = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("variants"),
        ];
        let prefix = &["evaluations", "my_eval"];
        assert!(
            !pattern_matches_prefix(pattern, prefix),
            "pattern should not match when literal doesn't match"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_too_short() {
        let pattern = &[PathComponent::Literal("functions"), PathComponent::Wildcard];
        let prefix = &["functions", "my_func", "variants"];
        assert!(
            !pattern_matches_prefix(pattern, prefix),
            "pattern should not match when it's shorter than prefix"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_empty_prefix() {
        let pattern = &[PathComponent::Literal("functions"), PathComponent::Wildcard];
        let prefix: &[&str] = &[];
        assert!(
            pattern_matches_prefix(pattern, prefix),
            "empty prefix should match any pattern"
        );
    }

    /// Test error type with `From<WalkError>` so visitors can propagate walker errors.
    #[derive(Debug)]
    enum TestError {
        Walk(WalkError),
        Visitor(String),
    }

    impl From<WalkError> for TestError {
        fn from(err: WalkError) -> Self {
            TestError::Walk(err)
        }
    }

    // A visitor that records every leaf it visits, as (path, leaf-string).
    #[derive(Default)]
    struct RecordingVisitor {
        leaves: Vec<(Vec<String>, String)>,
    }

    impl TomlPathVisitor<toml::Value> for RecordingVisitor {
        type Error = TestError;

        fn visit_leaf(
            &mut self,
            value: &mut toml::Value,
            path: &[String],
        ) -> Result<(), Self::Error> {
            let Some(s) = value.as_str() else {
                return Err(TestError::Visitor(format!(
                    "expected string leaf at `{}`",
                    path.join(".")
                )));
            };
            self.leaves.push((path.to_vec(), s.to_string()));
            Ok(())
        }
    }

    #[gtest]
    fn walks_literal_only_pattern() {
        let mut tree = toml::Value::Table(toml::toml! {
            [gateway]
            mode = "live"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("gateway"),
            PathComponent::Literal("mode"),
        ];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern], &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves.len(), eq(1));
        expect_that!(
            visitor.leaves[0].0,
            elements_are![eq("gateway"), eq("mode")]
        );
        expect_that!(visitor.leaves[0].1, eq("live"));
    }

    #[gtest]
    fn walks_wildcard_pattern_enumerates_all_keys() {
        let mut tree = toml::Value::Table(toml::toml! {
            [functions.foo]
            system_template = "foo-template"

            [functions.bar]
            system_template = "bar-template"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern], &mut visitor),
            ok(eq(&()))
        );

        expect_that!(visitor.leaves.len(), eq(2));
        expect_that!(
            visitor.leaves,
            unordered_elements_are![
                eq(&(
                    vec![
                        "functions".to_string(),
                        "bar".to_string(),
                        "system_template".to_string()
                    ],
                    "bar-template".to_string()
                )),
                eq(&(
                    vec![
                        "functions".to_string(),
                        "foo".to_string(),
                        "system_template".to_string()
                    ],
                    "foo-template".to_string()
                ))
            ]
        );
    }

    #[gtest]
    fn walks_multiple_patterns() {
        let mut tree = toml::Value::Table(toml::toml! {
            [functions.foo]
            system_template = "sys"
            user_template = "user"
        });
        let pattern_sys: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let pattern_user: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("user_template"),
        ];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern_sys, pattern_user], &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves.len(), eq(2));
        expect_that!(visitor.leaves[0].1, eq("sys"));
        expect_that!(visitor.leaves[1].1, eq("user"));
    }

    #[gtest]
    fn missing_key_is_not_an_error() {
        // The `functions` key doesn't contain `system_template`, so the walker should simply
        // skip it without reporting an error.
        let mut tree = toml::Value::Table(toml::toml! {
            [functions.foo]
            user_template = "u"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern], &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves, is_empty());
    }

    #[gtest]
    fn missing_literal_prefix_is_not_an_error() {
        // Root has no `functions` entry at all.
        let mut tree = toml::Value::Table(toml::toml! {
            other = "x"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern], &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves, is_empty());
    }

    #[gtest]
    fn non_table_mid_walk_defaults_to_error() {
        // `functions.foo` is a leaf instead of a table, so the walker can't descend into it.
        let mut tree = toml::Value::Table(toml::toml! {
            [functions]
            foo = "not-a-table"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = RecordingVisitor::default();

        let result = walk_patterns(&mut tree, &[pattern], &mut visitor);
        expect_that!(
            result,
            err(matches_pattern!(TestError::Walk(pat!(
                WalkError::ExpectedTable {
                    path: contains_substring("functions.foo"),
                    found: eq("string")
                }
            ))))
        );
    }

    /// Visitor that tolerates non-table mid-walk (mirrors `config-applier`'s behaviour).
    #[derive(Default)]
    struct LenientVisitor {
        leaves: usize,
    }

    impl TomlPathVisitor<toml::Value> for LenientVisitor {
        type Error = TestError;

        fn visit_leaf(
            &mut self,
            _value: &mut toml::Value,
            _path: &[String],
        ) -> Result<(), Self::Error> {
            self.leaves += 1;
            Ok(())
        }

        fn visit_non_table(
            &mut self,
            _path: &[String],
            _found_type: &str,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[gtest]
    fn non_table_mid_walk_can_be_tolerated_by_visitor() {
        let mut tree = toml::Value::Table(toml::toml! {
            [functions]
            foo = "not-a-table"

            [functions.bar]
            system_template = "bar-template"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = LenientVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pattern], &mut visitor),
            ok(eq(&()))
        );
        // Only the well-formed `bar` subtree produced a leaf.
        expect_that!(visitor.leaves, eq(1));
    }

    #[gtest]
    fn wildcard_at_end_of_pattern_errors() {
        let mut tree = toml::Value::Table(toml::toml! {
            [functions]
            foo = "x"
        });
        let pattern: &[PathComponent] =
            &[PathComponent::Literal("functions"), PathComponent::Wildcard];
        let mut visitor = RecordingVisitor::default();

        let result = walk_patterns(&mut tree, &[pattern], &mut visitor);
        expect_that!(
            result,
            err(matches_pattern!(TestError::Walk(pat!(
                WalkError::WildcardAtEnd { .. }
            ))))
        );
    }

    /// Visitor that rejects any wildcard key containing `..`.
    #[derive(Default)]
    struct ValidatingVisitor {
        visited_keys: Vec<String>,
    }

    impl TomlPathVisitor<toml::Value> for ValidatingVisitor {
        type Error = TestError;

        fn visit_leaf(
            &mut self,
            _value: &mut toml::Value,
            _path: &[String],
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn visit_wildcard_key(&mut self, path: &[String]) -> Result<(), Self::Error> {
            let key = path
                .last()
                .expect("wildcard visitor should receive a non-empty path");
            self.visited_keys.push(key.clone());
            if key.contains("..") {
                return Err(TestError::Visitor(format!("rejected key `{key}`")));
            }
            Ok(())
        }
    }

    #[gtest]
    fn wildcard_visitor_can_reject_keys() {
        let mut tree = toml::Value::Table(toml::toml! {
            [functions."../evil"]
            system_template = "x"

            [functions.good]
            system_template = "y"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let mut visitor = ValidatingVisitor::default();

        let result = walk_patterns(&mut tree, &[pattern], &mut visitor);
        expect_that!(
            result,
            err(matches_pattern!(TestError::Visitor(contains_substring(
                "rejected key"
            ))))
        );
        // BTreeMap iterates sorted, so `../evil` is visited first and causes an early abort.
        expect_that!(visitor.visited_keys, elements_are![eq("../evil")]);
    }

    #[gtest]
    fn walk_pattern_with_initial_path_prefixes_reported_path() {
        // Caller has already descended into `functions.foo.variants.v1`; walker only needs
        // to traverse `system_template` from here, but the visitor should see the full path.
        let mut subtree = toml::Value::Table(toml::toml! {
            system_template = "content"
        });
        let suffix: &[PathComponent] = &[PathComponent::Literal("system_template")];
        let initial = ["functions", "foo", "variants", "v1"];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_pattern(&mut subtree, suffix, &initial, &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves.len(), eq(1));
        expect_that!(
            visitor.leaves[0].0,
            elements_are![
                eq("functions"),
                eq("foo"),
                eq("variants"),
                eq("v1"),
                eq("system_template")
            ]
        );
    }

    #[gtest]
    fn visitor_leaf_error_is_propagated() {
        // `RecordingVisitor::visit_leaf` returns an error when the value isn't a leaf.
        // Here the terminal node is a table, so the visitor should error out.
        let mut tree = toml::Value::Table(toml::toml! {
            [gateway.mode]
            inner = "x"
        });
        let pattern: &[PathComponent] = &[
            PathComponent::Literal("gateway"),
            PathComponent::Literal("mode"),
        ];
        let mut visitor = RecordingVisitor::default();

        let result = walk_patterns(&mut tree, &[pattern], &mut visitor);
        expect_that!(
            result,
            err(matches_pattern!(TestError::Visitor(contains_substring(
                "gateway.mode"
            ))))
        );
    }

    #[gtest]
    fn path_is_restored_after_recursion() {
        // After the walker finishes, it should have left no path state behind. We check
        // this indirectly: visiting two independent patterns should produce full paths for
        // each, without the second path accumulating state from the first.
        let mut tree = toml::Value::Table(toml::toml! {
            [a]
            leaf1 = "one"

            [b]
            leaf2 = "two"
        });
        let pat_a: &[PathComponent] =
            &[PathComponent::Literal("a"), PathComponent::Literal("leaf1")];
        let pat_b: &[PathComponent] =
            &[PathComponent::Literal("b"), PathComponent::Literal("leaf2")];
        let mut visitor = RecordingVisitor::default();

        expect_that!(
            walk_patterns(&mut tree, &[pat_a, pat_b], &mut visitor),
            ok(eq(&()))
        );
        expect_that!(visitor.leaves[0].0, elements_are![eq("a"), eq("leaf1")]);
        expect_that!(visitor.leaves[1].0, elements_are![eq("b"), eq("leaf2")]);
    }
}
