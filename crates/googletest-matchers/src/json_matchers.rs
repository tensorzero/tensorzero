use googletest::{
    description::Description,
    matcher::{Matcher, MatcherBase, MatcherResult},
    matchers::{EqMatcher, eq},
};
use serde_json::Value;

use crate::partially::Partially;

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

macro_rules! impl_json_value_partial_ord_signed {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PartialOrd<$type> for JsonValueRef<'_> {
                fn partial_cmp(&self, other: &$type) -> Option<std::cmp::Ordering> {
                    self.0.as_i64().map(|v| v.cmp(&i64::from(*other)))
                }
            }
        )+
    };
}

macro_rules! impl_json_value_partial_ord_unsigned {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PartialOrd<$type> for JsonValueRef<'_> {
                fn partial_cmp(&self, other: &$type) -> Option<std::cmp::Ordering> {
                    self.0.as_u64().map(|v| v.cmp(&u64::from(*other)))
                }
            }
        )+
    };
}

impl_json_value_partial_eq_signed!(i8, i16, i32, i64);
impl_json_value_partial_eq_unsigned!(u8, u16, u32, u64);
impl_json_value_partial_ord_signed!(i8, i16, i32, i64);
impl_json_value_partial_ord_unsigned!(u8, u16, u32, u64);

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

impl PartialOrd<f32> for JsonValueRef<'_> {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.0
            .as_f64()
            .and_then(|v| v.partial_cmp(&f64::from(*other)))
    }
}

impl PartialOrd<f64> for JsonValueRef<'_> {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.0.as_f64().and_then(|v| v.partial_cmp(other))
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

impl<T: Partially> Partially for JsonMatchesMatcher<T> {
    fn partially(self) -> Self {
        JsonMatchesMatcher {
            matcher: self.matcher.partially(),
        }
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

/// A matcher for JSON objects that checks key-value pairs and optionally
/// enforces that no extra keys are present (exhaustive matching).
///
/// By default, matching is exhaustive: the actual object must have exactly
/// the specified keys. Use [`partially()`](crate::partially::partially)
/// to allow extra keys.
#[derive(Debug, MatcherBase)]
pub struct JsonObjectMatcher<KeyMatchers> {
    pub expected_keys: &'static [&'static str],
    pub key_matchers: KeyMatchers,
    pub exhaustive: bool,
}

impl<KM> Partially for JsonObjectMatcher<KM> {
    fn partially(mut self) -> Self {
        self.exhaustive = false;
        self
    }
}

impl<'a, KM> Matcher<JsonValueRef<'a>> for JsonObjectMatcher<KM>
where
    KM: Matcher<JsonValueRef<'a>>,
{
    fn matches(&self, actual: JsonValueRef<'a>) -> MatcherResult {
        let Some(obj) = actual.0.as_object() else {
            return MatcherResult::NoMatch;
        };

        let key_result = self.key_matchers.matches(actual);
        if key_result == MatcherResult::NoMatch {
            return MatcherResult::NoMatch;
        }

        if self.exhaustive {
            for key in obj.keys() {
                if !self.expected_keys.contains(&key.as_str()) {
                    return MatcherResult::NoMatch;
                }
            }
        }

        MatcherResult::Match
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        let mode = if self.exhaustive {
            "exactly"
        } else {
            "partially"
        };
        match matcher_result {
            MatcherResult::Match => format!(
                "{mode} matches JSON object with keys {:?}",
                self.expected_keys,
            )
            .into(),
            MatcherResult::NoMatch => format!(
                "does not {mode} match JSON object with keys {:?}",
                self.expected_keys,
            )
            .into(),
        }
    }

    fn explain_match(&self, actual: JsonValueRef<'a>) -> Description {
        let Some(obj) = actual.0.as_object() else {
            return format!("which is not a JSON object: {actual:?}").into();
        };

        // First check key matchers for better error messages
        let key_result = self.key_matchers.matches(actual);
        if key_result == MatcherResult::NoMatch {
            return self.key_matchers.explain_match(actual);
        }

        if self.exhaustive {
            let extra_keys: Vec<_> = obj
                .keys()
                .filter(|k| !self.expected_keys.contains(&k.as_str()))
                .collect();
            if !extra_keys.is_empty() {
                return format!("which has unexpected keys: {extra_keys:?}").into();
            }
        }

        "which matches".into()
    }
}

/// A matcher DSL for JSON values that accepts regular googletest matchers.
///
/// By default, object key matching is **exhaustive**: the actual object must
/// have exactly the keys specified. Use
/// [`partially()`](crate::partially::partially) to allow extra keys.
///
/// Example:
/// ```ignore
/// // Exact match — fails if `actual` has extra keys
/// matches_json!({ "name": eq("Alice"), "age": gt(0) })
///
/// // Partial match — extra keys in `actual` are ignored
/// partially(matches_json!({ "name": eq("Alice") }))
/// ```
#[macro_export]
macro_rules! matches_json {
    ({}) => {
        $crate::json_matchers::matches_json(
            $crate::json_matchers::JsonObjectMatcher {
                expected_keys: &[],
                key_matchers: $crate::json_matchers::googletest::matchers::anything(),
                exhaustive: true,
            }
        )
    };
    ({$($key:literal : $value:expr),+ $(,)?}) => {
        $crate::json_matchers::matches_json(
            $crate::json_matchers::JsonObjectMatcher {
                expected_keys: &[$($key),+],
                key_matchers: $crate::json_matchers::googletest::matchers::all![
                    $(
                        $crate::json_matchers::json_key(
                            $key,
                            $crate::json_matchers::into_json_matcher($value)
                        )
                    ),+
                ],
                exhaustive: true,
            }
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
    use crate::partially::partially;
    use googletest::prelude::*;
    use serde_json::json;

    // --- matches_json! basic usage ---

    #[gtest]
    fn matches_json_exact_match() {
        let actual = json!({"assistant_name": "AskJeeves"});
        expect_that!(actual, matches_json!({ "assistant_name": eq("AskJeeves") }));
    }

    #[gtest]
    fn matches_json_rejects_extra_keys_by_default() {
        let actual = json!({"assistant_name": "AskJeeves", "unused": true});
        expect_that!(
            actual,
            not(matches_json!({ "assistant_name": eq("AskJeeves") }))
        );
    }

    #[gtest]
    fn matches_json_partially_allows_extra_keys() {
        let actual = json!({"assistant_name": "AskJeeves", "unused": true});
        expect_that!(
            actual,
            partially(matches_json!({ "assistant_name": eq("AskJeeves") }))
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
                "messages": contains(partially(matches_json!({
                    "role": eq("user")
                })))
            })
        );
    }

    #[gtest]
    fn matches_json_empty_object_matches_empty_object() {
        let actual = json!({});
        expect_that!(actual, matches_json!({}));
    }

    #[gtest]
    fn matches_json_empty_object_rejects_non_empty_object() {
        let actual = json!({"name": "Alice"});
        expect_that!(actual, not(matches_json!({})));
    }

    #[gtest]
    fn matches_json_empty_object_partially_matches_any_object() {
        let actual = json!({"name": "Alice"});
        expect_that!(actual, partially(matches_json!({})));
    }

    #[gtest]
    fn matches_json_empty_object_rejects_non_objects() {
        let actual = json!([1, 2, 3]);
        expect_that!(actual, not(matches_json!({})));

        let actual = json!("a string");
        expect_that!(actual, not(matches_json!({})));

        let actual = json!(42);
        expect_that!(actual, not(matches_json!({})));
    }

    #[gtest]
    fn matches_json_multiple_keys() {
        let actual = json!({"a": 1, "b": "two", "c": true});
        expect_that!(
            actual,
            matches_json!({
                "a": eq(1),
                "b": eq("two"),
                "c": eq(true)
            })
        );
    }

    #[gtest]
    fn matches_json_nested_objects_exact() {
        let actual = json!({"outer": {"inner": "value"}});
        expect_that!(
            actual,
            matches_json!({
                "outer": matches_json!({ "inner": eq("value") })
            })
        );
    }

    #[gtest]
    fn matches_json_nested_objects_rejects_extra_inner_keys() {
        let actual = json!({"outer": {"inner": "value", "extra": 1}});
        expect_that!(
            actual,
            not(matches_json!({
                "outer": matches_json!({ "inner": eq("value") })
            }))
        );
    }

    #[gtest]
    fn matches_json_nested_objects_partially() {
        let actual = json!({"outer": {"inner": "value", "extra": 1}});
        expect_that!(
            actual,
            matches_json!({
                "outer": partially(matches_json!({ "inner": eq("value") }))
            })
        );
    }

    #[gtest]
    fn matches_json_with_value_matcher() {
        let actual = json!("hello");
        expect_that!(actual, matches_json!(eq("hello")));
    }

    // --- matches_json! failure cases ---

    #[gtest]
    fn matches_json_fails_on_wrong_value() {
        let actual = json!({"name": "Alice"});
        expect_that!(actual, not(matches_json!({ "name": eq("Bob") })));
    }

    #[gtest]
    fn matches_json_fails_on_missing_key() {
        let actual = json!({"name": "Alice"});
        // Missing key AND extra key "name"
        expect_that!(
            actual,
            not(matches_json!({ "missing_key": eq("anything") }))
        );
    }

    #[gtest]
    fn matches_json_fails_on_type_mismatch() {
        let actual = json!({"count": "not_a_number"});
        expect_that!(actual, not(matches_json!({ "count": eq(42i64) })));
    }

    #[gtest]
    fn matches_json_fails_on_non_object_when_expecting_keys() {
        let actual = json!("just a string");
        expect_that!(actual, not(matches_json!({ "key": eq("value") })));
    }

    // --- Numeric comparisons with PartialOrd ---

    #[gtest]
    fn matches_json_gt_unsigned() {
        let actual = json!({"tokens": 150});
        expect_that!(actual, matches_json!({ "tokens": gt(100) }));
    }

    #[gtest]
    fn matches_json_gt_signed() {
        let actual = json!({"score": -5});
        expect_that!(actual, matches_json!({ "score": gt(-10) }));
    }

    #[gtest]
    fn matches_json_gt_float() {
        let actual = json!({"temperature": 0.8});
        expect_that!(actual, matches_json!({ "temperature": gt(0.5f64) }));
    }

    #[gtest]
    fn matches_json_lt_unsigned() {
        let actual = json!({"count": 5});
        expect_that!(actual, matches_json!({ "count": lt(10) }));
    }

    #[gtest]
    fn matches_json_ge_le() {
        let actual = json!({"value": 10});
        expect_that!(actual, matches_json!({ "value": ge(10) }));
        expect_that!(actual, matches_json!({ "value": le(10) }));
    }

    #[gtest]
    fn matches_json_gt_fails_when_less() {
        let actual = json!({"tokens": 50});
        expect_that!(actual, not(matches_json!({ "tokens": gt(100) })));
    }

    #[gtest]
    fn matches_json_ord_returns_none_for_non_numeric() {
        let actual = json!({"name": "Alice"});
        // String value can't be compared with gt(numeric), so the matcher fails
        expect_that!(actual, not(matches_json!({ "name": gt(0) })));
    }

    #[gtest]
    fn matches_json_gt_float_nan_returns_none() {
        let actual = json!({"value": "not a float"});
        expect_that!(actual, not(matches_json!({ "value": gt(0.0) })));
    }

    // --- is_null ---

    #[gtest]
    fn is_null_matches_null_values() {
        let actual = json!({"snapshot_hash": null});
        expect_that!(&actual["snapshot_hash"], is_null());
    }

    #[gtest]
    fn is_null_can_be_negated_for_non_null_values() {
        let actual = json!({"snapshot_hash": "abc123"});
        expect_that!(&actual["snapshot_hash"], not(is_null()));
    }

    #[gtest]
    fn is_null_rejects_non_null_types() {
        expect_that!(json!(0), not(is_null()));
        expect_that!(json!(""), not(is_null()));
        expect_that!(json!(false), not(is_null()));
        expect_that!(json!([]), not(is_null()));
        expect_that!(json!({}), not(is_null()));
    }

    // --- json_key ---

    #[gtest]
    fn json_key_matches_nested_value() {
        let actual = json!({"usage": {"input_tokens": 150}});
        expect_that!(
            actual,
            matches_json!({
                "usage": matches_json!({ "input_tokens": gt(100) })
            })
        );
    }

    #[gtest]
    fn json_key_fails_on_missing_key() {
        let actual = json!({"a": 1});
        let matcher = json_key("b", eq(1));
        expect_that!(
            matcher.matches(JsonValueRef(&actual)),
            eq(MatcherResult::NoMatch)
        );
    }

    #[gtest]
    fn json_key_fails_on_non_object() {
        let actual = json!([1, 2, 3]);
        let matcher = json_key("key", eq(1));
        expect_that!(
            matcher.matches(JsonValueRef(&actual)),
            eq(MatcherResult::NoMatch)
        );
    }

    // --- JsonValueRef PartialEq ---

    #[gtest]
    fn json_value_ref_eq_string() {
        let v = json!("hello");
        expect_that!(JsonValueRef(&v), eq("hello"));
        expect_that!(JsonValueRef(&v), not(eq("world")));
    }

    #[gtest]
    fn json_value_ref_eq_bool() {
        let v = json!(true);
        expect_that!(JsonValueRef(&v), eq(true));
        expect_that!(JsonValueRef(&v), not(eq(false)));
    }

    #[gtest]
    fn json_value_ref_eq_unsigned() {
        let v = json!(42);
        expect_that!(JsonValueRef(&v), eq(42u64));
        expect_that!(JsonValueRef(&v), not(eq(0)));
    }

    #[gtest]
    fn json_value_ref_eq_signed() {
        let v = json!(-7);
        expect_that!(JsonValueRef(&v), eq(-7i64));
        expect_that!(JsonValueRef(&v), not(eq(7i64)));
    }

    #[gtest]
    fn json_value_ref_eq_float() {
        let v = json!(1.23);
        expect_that!(JsonValueRef(&v), eq(1.23f64));
        expect_that!(JsonValueRef(&v), not(eq(4.56f64)));
    }

    #[gtest]
    fn json_value_ref_eq_cross_type_returns_false() {
        let v = json!("not a number");
        expect_that!(JsonValueRef(&v), not(eq(42u64)));

        let v = json!(42);
        expect_that!(JsonValueRef(&v), not(eq("42")));
    }

    // --- JsonValueRef PartialOrd ---

    #[gtest]
    fn json_value_ref_ord_unsigned() {
        let v = json!(10);
        expect_that!(JsonValueRef(&v), gt(5u64));
        expect_that!(JsonValueRef(&v), lt(20));
        expect_that!(JsonValueRef(&v), ge(10));
        expect_that!(JsonValueRef(&v), le(10));
    }

    #[gtest]
    fn json_value_ref_ord_signed() {
        let v = json!(-3);
        expect_that!(JsonValueRef(&v), gt(-10));
        expect_that!(JsonValueRef(&v), lt(0));
    }

    #[gtest]
    fn json_value_ref_ord_float() {
        let v = json!(2.5);
        expect_that!(JsonValueRef(&v), gt(1.0));
        expect_that!(JsonValueRef(&v), lt(3.0));
    }

    #[gtest]
    fn json_value_ref_ord_none_for_wrong_type() {
        let v = json!("hello");
        // partial_cmp returns None, so gt/lt fail
        let ref_v = JsonValueRef(&v);
        expect_that!(ref_v.partial_cmp(&0), none());
        expect_that!(ref_v.partial_cmp(&0), none());
        expect_that!(ref_v.partial_cmp(&0.0), none());
    }

    // --- IntoIterator for arrays ---

    #[gtest]
    fn json_value_ref_iterates_arrays() {
        let actual = json!({"items": [1, 2, 3]});
        expect_that!(
            actual,
            matches_json!({
                "items": contains(eq(2u64))
            })
        );
    }

    #[gtest]
    fn json_value_ref_empty_array() {
        let actual = json!({"items": []});
        expect_that!(actual, matches_json!({ "items": is_empty() }));
    }

    #[gtest]
    fn json_value_ref_non_array_iterates_empty() {
        let v = json!("not an array");
        let ref_v = JsonValueRef(&v);
        let items: Vec<_> = ref_v.into_iter().collect();
        expect_that!(items, is_empty());
    }

    // --- describe / explain_match ---

    #[gtest]
    fn json_key_explain_match_missing_key() {
        let actual = json!({"a": 1});
        let matcher = json_key("b", eq(1));
        let explanation = matcher.explain_match(JsonValueRef(&actual));
        let desc = format!("{explanation}");
        expect_that!(desc, contains_substring("missing key"));
        expect_that!(desc, contains_substring("`b`"));
    }

    #[gtest]
    fn json_key_explain_match_not_object() {
        let actual = json!(42);
        let matcher = json_key("key", eq(1));
        let explanation = matcher.explain_match(JsonValueRef(&actual));
        let desc = format!("{explanation}");
        expect_that!(desc, contains_substring("not a JSON object"));
    }

    #[gtest]
    fn is_null_explain_match_non_null() {
        let actual = json!("hello");
        let explanation = IsNullMatcher.explain_match(JsonValueRef(&actual));
        let desc = format!("{explanation}");
        expect_that!(desc, contains_substring("hello"));
    }

    // --- JsonObjectMatcher exhaustive / partial ---

    #[gtest]
    fn json_object_matcher_explain_extra_keys() {
        let actual = json!({"name": "Alice", "extra": true});
        let matcher = matches_json!({ "name": eq("Alice") });
        let explanation = Matcher::<&Value>::explain_match(&matcher, &actual);
        let desc = format!("{explanation}");
        expect_that!(desc, contains_substring("unexpected keys"));
        expect_that!(desc, contains_substring("extra"));
    }

    #[gtest]
    fn json_object_matcher_partially_at_top_level() {
        let actual = json!({"name": "Alice", "age": 30, "extra": true});
        expect_that!(
            actual,
            partially(matches_json!({ "name": eq("Alice"), "age": eq(30) }))
        );
    }

    #[gtest]
    fn json_object_matcher_exact_match_with_all_keys() {
        let actual = json!({"name": "Alice", "age": 30});
        expect_that!(
            actual,
            matches_json!({ "name": eq("Alice"), "age": eq(30) })
        );
    }

    #[gtest]
    fn json_object_matcher_exact_fails_with_extra_keys() {
        let actual = json!({"name": "Alice", "age": 30, "extra": true});
        expect_that!(
            actual,
            not(matches_json!({ "name": eq("Alice"), "age": eq(30) }))
        );
    }

    // --- Nested objects follow the same exhaustive/partial pattern ---

    #[gtest]
    fn matches_json_deeply_nested_objects() {
        let actual = json!({"a": {"b": {"c": "deep"}}});
        expect_that!(
            actual,
            matches_json!({
                "a": matches_json!({
                    "b": matches_json!({ "c": eq("deep") })
                })
            })
        );
    }

    #[gtest]
    fn matches_json_deeply_nested_rejects_extra_keys_at_any_level() {
        let actual = json!({"a": {"b": {"c": "deep", "extra": 1}}});
        expect_that!(
            actual,
            not(matches_json!({
                "a": matches_json!({
                    "b": matches_json!({ "c": eq("deep") })
                })
            }))
        );
    }

    #[gtest]
    fn matches_json_deeply_nested_partially_allows_extra_keys_at_any_level() {
        let actual = json!({"a": {"b": {"c": "deep", "extra": 1}, "extra2": 2}, "extra3": 3});
        expect_that!(
            actual,
            partially(matches_json!({
                "a": partially(matches_json!({
                    "b": partially(matches_json!({ "c": eq("deep") }))
                }))
            }))
        );
    }

    // --- Nested arrays follow the same pattern ---

    #[gtest]
    fn matches_json_nested_array_of_objects() {
        let actual = json!({
            "items": [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"}
            ]
        });
        expect_that!(
            actual,
            matches_json!({
                "items": contains(matches_json!({ "id": eq(1), "name": eq("first") }))
            })
        );
    }

    #[gtest]
    fn matches_json_nested_array_of_objects_rejects_extra_keys() {
        let actual = json!({
            "items": [{"id": 1, "name": "first", "extra": true}]
        });
        expect_that!(
            actual,
            not(matches_json!({
                "items": contains(matches_json!({ "id": eq(1), "name": eq("first") }))
            }))
        );
    }

    #[gtest]
    fn matches_json_nested_array_of_objects_partially() {
        let actual = json!({
            "items": [{"id": 1, "name": "first", "extra": true}]
        });
        expect_that!(
            actual,
            matches_json!({
                "items": contains(partially(matches_json!({ "id": eq(1) })))
            })
        );
    }

    #[gtest]
    fn matches_json_nested_array_with_each() {
        let actual = json!({
            "scores": [10, 20, 30]
        });
        expect_that!(
            actual,
            matches_json!({
                "scores": each(gt(5))
            })
        );
    }

    #[gtest]
    fn matches_json_array_of_arrays() {
        let actual = json!({
            "matrix": [[1, 2], [3, 4]]
        });
        expect_that!(
            actual,
            matches_json!({
                "matrix": contains(contains(eq(3u64)))
            })
        );
    }

    // --- partially() does NOT propagate to nested matches_json! ---

    #[gtest]
    fn matches_json_without_partially_nested_extra_keys_fail() {
        let actual = json!({"outer": {"inner": "value", "extra": true}});
        expect_that!(
            actual,
            not(matches_json!({
                "outer": matches_json!({ "inner": eq("value") })
            }))
        );
    }

    #[gtest]
    fn matches_json_top_level_partially_does_not_propagate_to_nested() {
        // partially() on the outer matcher allows extra keys at the top level,
        // but does NOT propagate into nested matches_json! matchers
        let actual = json!({"outer": {"inner": "value", "extra": true}, "top_extra": 1});
        expect_that!(
            actual,
            not(partially(matches_json!({
                "outer": matches_json!({ "inner": eq("value") })
            })))
        );
    }

    #[gtest]
    fn matches_json_with_partially_on_nested_extra_keys_succeed() {
        // partially() must be applied at each level where extra keys should be allowed
        let actual = json!({"outer": {"inner": "value", "extra": true}});
        expect_that!(
            actual,
            matches_json!({
                "outer": partially(matches_json!({ "inner": eq("value") }))
            })
        );
    }
}
