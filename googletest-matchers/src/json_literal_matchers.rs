use googletest::{
    description::Description,
    matcher::{Matcher, MatcherBase, MatcherResult},
};
use serde_json::{Map, Value};

pub use serde_json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralMatchMode {
    Exact,
    PartialObject,
}

/// Matches a JSON value against a literal JSON value.
///
/// By default this performs an exact match (`actual == expected`).
pub fn matches_json_literal(expected: Value) -> MatchesJsonLiteralMatcher {
    MatchesJsonLiteralMatcher {
        expected,
        mode: LiteralMatchMode::Exact,
    }
}

/// Converts a literal matcher into partial-object mode.
///
/// In partial-object mode, object keys present in the expected literal must
/// exist and match in the actual value, but extra keys in the actual object
/// are ignored (recursively for nested objects). Arrays still require exact
/// length and positional matches.
pub fn partially(matcher: MatchesJsonLiteralMatcher) -> MatchesJsonLiteralMatcher {
    matcher.partially()
}

/// A convenience macro for inlining a JSON literal matcher.
#[macro_export]
macro_rules! matches_json_literal {
    ($($json:tt)+) => {
        $crate::json_literal_matchers::matches_json_literal(
            $crate::json_literal_matchers::serde_json::json!($($json)+)
        )
    };
}

#[derive(Debug, MatcherBase)]
pub struct MatchesJsonLiteralMatcher {
    expected: Value,
    mode: LiteralMatchMode,
}

impl MatchesJsonLiteralMatcher {
    fn partially(mut self) -> Self {
        self.mode = LiteralMatchMode::PartialObject;
        self
    }
}

impl Matcher<&Value> for MatchesJsonLiteralMatcher {
    fn matches(&self, actual: &Value) -> MatcherResult {
        find_literal_mismatch(actual, &self.expected, "$", self.mode)
            .is_none()
            .into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        let mode_text = match self.mode {
            LiteralMatchMode::Exact => "exactly matches",
            LiteralMatchMode::PartialObject => "partially matches",
        };
        match matcher_result {
            MatcherResult::Match => format!(
                "{mode_text} JSON literal {expected:#}",
                expected = self.expected
            )
            .into(),
            MatcherResult::NoMatch => format!(
                "does not {mode_text} JSON literal {expected:#}",
                expected = self.expected
            )
            .into(),
        }
    }

    fn explain_match(&self, actual: &Value) -> Description {
        match find_literal_mismatch(actual, &self.expected, "$", self.mode) {
            Some(message) => message.into(),
            None => "which matches".into(),
        }
    }
}

fn find_literal_mismatch(
    actual: &Value,
    expected: &Value,
    path: &str,
    mode: LiteralMatchMode,
) -> Option<String> {
    match (actual, expected) {
        (Value::Object(actual_object), Value::Object(expected_object)) => {
            find_object_mismatch(actual_object, expected_object, path, mode)
        }
        (Value::Array(actual_array), Value::Array(expected_array)) => {
            if actual_array.len() != expected_array.len() {
                return Some(format!(
                    "which differs at {path}: expected array length {expected_len}, got {actual_len}",
                    expected_len = expected_array.len(),
                    actual_len = actual_array.len()
                ));
            }

            for (index, (actual_value, expected_value)) in
                actual_array.iter().zip(expected_array).enumerate()
            {
                let nested_path = format!("{path}[{index}]");
                if let Some(mismatch) =
                    find_literal_mismatch(actual_value, expected_value, &nested_path, mode)
                {
                    return Some(mismatch);
                }
            }
            None
        }
        _ => {
            if actual == expected {
                None
            } else {
                Some(format!(
                    "which differs at {path}: expected {expected:#}, got {actual:#}"
                ))
            }
        }
    }
}

fn find_object_mismatch(
    actual: &Map<String, Value>,
    expected: &Map<String, Value>,
    path: &str,
    mode: LiteralMatchMode,
) -> Option<String> {
    for (key, expected_value) in expected {
        let Some(actual_value) = actual.get(key) else {
            return Some(format!("which is missing key `{key}` at {path}"));
        };

        let nested_path = format!("{path}.{key}");
        if let Some(mismatch) =
            find_literal_mismatch(actual_value, expected_value, &nested_path, mode)
        {
            return Some(mismatch);
        }
    }

    if mode == LiteralMatchMode::Exact {
        for key in actual.keys() {
            if !expected.contains_key(key) {
                return Some(format!("which has unexpected key `{key}` at {path}"));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use serde_json::json;

    #[gtest]
    fn matches_json_literal_exact_match_passes() {
        let actual = json!({
            "assistant_name": "AskJeeves",
            "count": 2
        });

        expect_that!(
            actual,
            matches_json_literal(json!({
                "assistant_name": "AskJeeves",
                "count": 2
            }))
        );
    }

    #[gtest]
    fn matches_json_literal_exact_match_fails_on_extra_keys() {
        let actual = json!({
            "assistant_name": "AskJeeves",
            "count": 2,
            "extra": true
        });

        expect_that!(
            actual,
            not(matches_json_literal(json!({
                "assistant_name": "AskJeeves",
                "count": 2
            })))
        );
    }

    #[gtest]
    fn partially_matches_json_literal_ignores_extra_keys() {
        let actual = json!({
            "assistant_name": "AskJeeves",
            "meta": {
                "id": 7,
                "extra_nested": "ignored"
            },
            "extra": true
        });

        expect_that!(
            actual,
            partially(matches_json_literal(json!({
                "assistant_name": "AskJeeves",
                "meta": {
                    "id": 7
                }
            })))
        );
    }

    #[gtest]
    fn partially_matches_json_literal_still_fails_on_missing_keys() {
        let actual = json!({
            "assistant_name": "AskJeeves"
        });

        expect_that!(
            actual,
            not(partially(matches_json_literal(json!({
                "assistant_name": "AskJeeves",
                "required": 1
            }))))
        );
    }

    #[gtest]
    fn partially_matches_json_literal_still_fails_on_mismatched_values() {
        let actual = json!({
            "assistant_name": "AskJeeves",
            "count": 2
        });

        expect_that!(
            actual,
            not(partially(matches_json_literal(json!({
                "assistant_name": "AskJeeves",
                "count": 3
            }))))
        );
    }

    #[gtest]
    fn matches_json_literal_macro_allows_inline_literals() {
        let actual = json!({
            "assistant_name": "AskJeeves"
        });

        expect_that!(
            actual,
            matches_json_literal!({"assistant_name": "AskJeeves"})
        );
    }
}
