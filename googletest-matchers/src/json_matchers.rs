use googletest::{
    description::Description,
    matcher::{Matcher, MatcherBase, MatcherResult},
    matchers::{EqMatcher, eq},
};
use serde_json::Value;

pub use googletest;
pub use serde_json;

/// A `Copy` wrapper used as matcher subject for JSON values.
///
/// This adapter intentionally implements `PartialEq` for common scalar types
/// and `IntoIterator` for arrays so existing googletest matchers like `eq(...)`
/// and `contains(...)` can be reused directly.
#[derive(Clone, Copy)]
pub struct JsonValueRef<'a>(&'a Value);

impl std::fmt::Debug for JsonValueRef<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq<&str> for JsonValueRef<'_> {
    fn eq(&self, other: &&str) -> bool {
        self.0.as_str() == Some(*other)
    }
}

impl PartialEq<String> for JsonValueRef<'_> {
    fn eq(&self, other: &String) -> bool {
        self.0.as_str() == Some(other.as_str())
    }
}

impl PartialEq<bool> for JsonValueRef<'_> {
    fn eq(&self, other: &bool) -> bool {
        self.0.as_bool() == Some(*other)
    }
}

macro_rules! impl_json_value_partial_eq_signed {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PartialEq<$type> for JsonValueRef<'_> {
                fn eq(&self, other: &$type) -> bool {
                    self.0.as_i64() == Some(i64::from(*other))
                }
            }
        )+
    };
}

macro_rules! impl_json_value_partial_eq_unsigned {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PartialEq<$type> for JsonValueRef<'_> {
                fn eq(&self, other: &$type) -> bool {
                    self.0.as_u64() == Some(u64::from(*other))
                }
            }
        )+
    };
}

impl_json_value_partial_eq_signed!(i8, i16, i32, i64);
impl_json_value_partial_eq_unsigned!(u8, u16, u32, u64);

impl PartialEq<f32> for JsonValueRef<'_> {
    fn eq(&self, other: &f32) -> bool {
        self.0.as_f64() == Some(f64::from(*other))
    }
}

impl PartialEq<f64> for JsonValueRef<'_> {
    fn eq(&self, other: &f64) -> bool {
        self.0.as_f64() == Some(*other)
    }
}

impl PartialEq<Value> for JsonValueRef<'_> {
    fn eq(&self, other: &Value) -> bool {
        self.0 == other
    }
}

impl PartialEq<&Value> for JsonValueRef<'_> {
    fn eq(&self, other: &&Value) -> bool {
        self.0 == *other
    }
}

/// Iterator over a JSON array as [`JsonValueRef`] items.
pub enum JsonValueIter<'a> {
    Array(std::slice::Iter<'a, Value>),
    Empty,
}

impl<'a> Iterator for JsonValueIter<'a> {
    type Item = JsonValueRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            JsonValueIter::Array(iter) => iter.next().map(JsonValueRef),
            JsonValueIter::Empty => None,
        }
    }
}

impl<'a> IntoIterator for JsonValueRef<'a> {
    type Item = JsonValueRef<'a>;
    type IntoIter = JsonValueIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            Value::Array(values) => JsonValueIter::Array(values.iter()),
            _ => JsonValueIter::Empty,
        }
    }
}

/// Converts supported values into a matcher of [`JsonValueRef`].
pub trait IntoJsonMatcher {
    type MatcherT;
    fn into_json_matcher(self) -> Self::MatcherT;
}

impl<MatcherT> IntoJsonMatcher for MatcherT
where
    for<'a> MatcherT: Matcher<JsonValueRef<'a>>,
{
    type MatcherT = MatcherT;

    fn into_json_matcher(self) -> Self::MatcherT {
        self
    }
}

impl<'a> IntoJsonMatcher for &'a str {
    type MatcherT = EqMatcher<&'a str>;

    fn into_json_matcher(self) -> Self::MatcherT {
        eq(self)
    }
}

impl IntoJsonMatcher for String {
    type MatcherT = EqMatcher<String>;

    fn into_json_matcher(self) -> Self::MatcherT {
        eq(self)
    }
}

impl IntoJsonMatcher for bool {
    type MatcherT = EqMatcher<bool>;

    fn into_json_matcher(self) -> Self::MatcherT {
        eq(self)
    }
}

macro_rules! impl_into_json_matcher_for_number {
    ($($type:ty),+ $(,)?) => {
        $(
            impl IntoJsonMatcher for $type {
                type MatcherT = EqMatcher<$type>;

                fn into_json_matcher(self) -> Self::MatcherT {
                    eq(self)
                }
            }
        )+
    };
}

impl_into_json_matcher_for_number!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl IntoJsonMatcher for Value {
    type MatcherT = EqMatcher<Value>;

    fn into_json_matcher(self) -> Self::MatcherT {
        eq(self)
    }
}

impl<'a> IntoJsonMatcher for &'a Value {
    type MatcherT = EqMatcher<&'a Value>;

    fn into_json_matcher(self) -> Self::MatcherT {
        eq(self)
    }
}

pub fn into_json_matcher<MatcherLike: IntoJsonMatcher>(
    matcher: MatcherLike,
) -> MatcherLike::MatcherT {
    matcher.into_json_matcher()
}

/// A matcher that checks whether a JSON value is an object.
#[derive(Debug, MatcherBase)]
pub struct IsJsonObjectMatcher;

pub fn is_json_object() -> JsonMatchesMatcher<IsJsonObjectMatcher> {
    matches_json(IsJsonObjectMatcher)
}

impl<'a> Matcher<JsonValueRef<'a>> for IsJsonObjectMatcher {
    fn matches(&self, actual: JsonValueRef<'a>) -> MatcherResult {
        actual.0.is_object().into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => "is a JSON object".into(),
            MatcherResult::NoMatch => "is not a JSON object".into(),
        }
    }

    fn explain_match(&self, actual: JsonValueRef<'a>) -> Description {
        if actual.0.is_object() {
            "which is a JSON object".into()
        } else {
            format!("which is {actual:?}").into()
        }
    }
}

/// A matcher that checks whether a JSON value is `null`.
#[derive(Debug, MatcherBase)]
pub struct IsNullMatcher;

pub fn is_null() -> JsonMatchesMatcher<IsNullMatcher> {
    matches_json(IsNullMatcher)
}

impl<'a> Matcher<JsonValueRef<'a>> for IsNullMatcher {
    fn matches(&self, actual: JsonValueRef<'a>) -> MatcherResult {
        actual.0.is_null().into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => "is null".into(),
            MatcherResult::NoMatch => "is not null".into(),
        }
    }

    fn explain_match(&self, actual: JsonValueRef<'a>) -> Description {
        if actual.0.is_null() {
            "which is null".into()
        } else {
            format!("which is {actual:?}").into()
        }
    }
}

/// A matcher for matching the value at a JSON object key.
#[derive(Debug, MatcherBase)]
pub struct JsonKeyMatcher<MatcherT> {
    key: String,
    matcher: MatcherT,
}

pub fn json_key<MatcherLike: IntoJsonMatcher>(
    key: impl Into<String>,
    matcher: MatcherLike,
) -> JsonKeyMatcher<MatcherLike::MatcherT> {
    JsonKeyMatcher {
        key: key.into(),
        matcher: matcher.into_json_matcher(),
    }
}

impl<'a, MatcherT> Matcher<JsonValueRef<'a>> for JsonKeyMatcher<MatcherT>
where
    MatcherT: Matcher<JsonValueRef<'a>>,
{
    fn matches(&self, actual: JsonValueRef<'a>) -> MatcherResult {
        let Some(object) = actual.0.as_object() else {
            return MatcherResult::NoMatch;
        };
        let Some(value) = object.get(&self.key) else {
            return MatcherResult::NoMatch;
        };
        self.matcher.matches(JsonValueRef(value))
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => format!(
                "has key `{}` with value which {}",
                self.key,
                self.matcher.describe(MatcherResult::Match)
            )
            .into(),
            MatcherResult::NoMatch => format!(
                "does not have key `{}` with value which {}",
                self.key,
                self.matcher.describe(MatcherResult::Match)
            )
            .into(),
        }
    }

    fn explain_match(&self, actual: JsonValueRef<'a>) -> Description {
        let Some(object) = actual.0.as_object() else {
            return format!("which is not a JSON object: {actual:?}").into();
        };
        let Some(value) = object.get(&self.key) else {
            return format!(
                "which is missing key `{}`. Available keys: {:?}",
                self.key,
                object.keys().collect::<Vec<_>>()
            )
            .into();
        };

        self.matcher.explain_match(JsonValueRef(value))
    }
}

/// A wrapper matcher that can match both `&serde_json::Value` and nested
/// [`JsonValueRef`] values.
#[derive(Debug, MatcherBase)]
pub struct JsonMatchesMatcher<MatcherT> {
    matcher: MatcherT,
}

pub fn matches_json<MatcherLike: IntoJsonMatcher>(
    matcher: MatcherLike,
) -> JsonMatchesMatcher<MatcherLike::MatcherT> {
    JsonMatchesMatcher {
        matcher: matcher.into_json_matcher(),
    }
}

impl<MatcherT> Matcher<&Value> for JsonMatchesMatcher<MatcherT>
where
    for<'a> MatcherT: Matcher<JsonValueRef<'a>>,
{
    fn matches(&self, actual: &Value) -> MatcherResult {
        self.matcher.matches(JsonValueRef(actual))
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        self.matcher.describe(matcher_result)
    }

    fn explain_match(&self, actual: &Value) -> Description {
        self.matcher.explain_match(JsonValueRef(actual))
    }
}

impl<'a, MatcherT> Matcher<JsonValueRef<'a>> for JsonMatchesMatcher<MatcherT>
where
    MatcherT: Matcher<JsonValueRef<'a>>,
{
    fn matches(&self, actual: JsonValueRef<'a>) -> MatcherResult {
        self.matcher.matches(actual)
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        self.matcher.describe(matcher_result)
    }

    fn explain_match(&self, actual: JsonValueRef<'a>) -> Description {
        self.matcher.explain_match(actual)
    }
}

/// A matcher DSL for JSON values that accepts regular googletest matchers.
///
/// Example:
/// `matches_json!({ "assistant_name": eq("AskJeeves"), "messages": contains(matches_json!({...})) })`
#[macro_export]
macro_rules! matches_json {
    ({}) => {
        $crate::json_matchers::matches_json(
            $crate::json_matchers::is_json_object()
        )
    };
    ({$($key:literal : $value:expr),+ $(,)?}) => {
        $crate::json_matchers::matches_json(
            $crate::json_matchers::googletest::matchers::all![
                $(
                    $crate::json_matchers::json_key(
                        $key,
                        $crate::json_matchers::into_json_matcher($value)
                    )
                ),+
            ]
        )
    };
    ($value_matcher:expr $(,)?) => {
        $crate::json_matchers::matches_json(
            $crate::json_matchers::into_json_matcher($value_matcher)
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use serde_json::json;

    #[gtest]
    fn matches_json_supports_eq_matchers() {
        let actual = json!({
            "assistant_name": "AskJeeves",
            "unused": true
        });

        expect_that!(
            actual,
            matches_json!({
                "assistant_name": eq("AskJeeves")
            })
        );
    }

    #[gtest]
    fn matches_json_supports_contains_with_nested_matches_json() {
        let actual = json!({
            "messages": [
                {"role": "system", "content": []},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ]
        });

        expect_that!(
            actual,
            matches_json!({
                "messages": contains(matches_json!({
                    "role": eq("user")
                }))
            })
        );
    }

    #[gtest]
    fn matches_json_supports_literal_shorthand() {
        let actual = json!({
            "assistant_name": "AskJeeves"
        });

        expect_that!(actual, matches_json!({ "assistant_name": "AskJeeves" }));
    }

    #[gtest]
    fn is_null_matches_null_values() {
        let actual = json!({
            "snapshot_hash": null
        });

        expect_that!(&actual["snapshot_hash"], is_null());
    }

    #[gtest]
    fn is_null_can_be_negated_for_non_null_values() {
        let actual = json!({
            "snapshot_hash": "abc123"
        });

        expect_that!(&actual["snapshot_hash"], not(is_null()));
    }
}
