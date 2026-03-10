/// Trait for matchers that support switching to partial-object mode.
///
/// In partial-object mode, object keys present in the expected value must
/// exist and match in the actual value, but extra keys in the actual object
/// are ignored.
pub trait Partially {
    fn partially(self) -> Self;
}

/// Converts a matcher into partial-object mode.
///
/// In partial-object mode, object keys present in the expected value must
/// exist and match in the actual value, but extra keys in the actual object
/// are ignored.
pub fn partially<T: Partially>(matcher: T) -> T {
    matcher.partially()
}
