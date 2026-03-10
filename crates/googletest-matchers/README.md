# googletest-matchers

Custom [googletest](https://crates.io/crates/googletest) matchers for working with `serde_json::Value` in tests.

All value positions in `matches_json!` accept standard googletest matchers (`eq`, `gt`, `contains`, `each`, etc.). See [googletest's available matchers](https://docs.rs/googletest/latest/googletest/index.html#available-matchers) for the full list.

## Matchers

### `matches_json!` — structured JSON object matching

Matches a `serde_json::Value` against a set of key-matcher pairs. Each value position takes a googletest matcher (e.g. `eq(...)`, `gt(...)`, `contains(...)`).

By default, matching is **exhaustive**: the actual object must have exactly the specified keys. Use `partially(...)` to allow extra keys.

```rust
use googletest::prelude::*;
use googletest_matchers::{matches_json, partially};
use serde_json::json;

let actual = json!({"name": "Alice", "age": 30});

// Exact match — fails if `actual` has extra keys
expect_that!(actual, matches_json!({ "name": eq("Alice"), "age": eq(30) }));

// Partial match — extra keys in `actual` are ignored
let actual = json!({"name": "Alice", "age": 30, "extra": true});
expect_that!(actual, partially(matches_json!({ "name": eq("Alice") })));
```

Nested objects require nested `matches_json!` calls. `partially()` does **not** propagate — it must be applied at each level:

```rust
let actual = json!({"user": {"name": "Alice", "role": "admin", "extra": true}});

// Inner extra keys are rejected without partially()
expect_that!(actual, not(matches_json!({
    "user": matches_json!({ "name": eq("Alice"), "role": eq("admin") })
})));

// Apply partially() at the inner level to allow them
expect_that!(actual, matches_json!({
    "user": partially(matches_json!({ "name": eq("Alice") }))
}));
```

Arrays are matched using standard googletest collection matchers:

```rust
let actual = json!({"scores": [10, 20, 30]});

expect_that!(actual, matches_json!({
    "scores": contains(eq(20u64))
}));

expect_that!(actual, matches_json!({
    "scores": each(gt(5))
}));
```

Empty object matching checks that the value is a JSON object (any keys allowed):

```rust
let actual = json!({"anything": "goes"});
expect_that!(actual, matches_json!({}));
```

A bare matcher can be used for non-object values:

```rust
let actual = json!("hello");
expect_that!(actual, matches_json!(eq("hello")));
```

### `matches_json_literal!` — literal JSON value matching

Matches a `serde_json::Value` against a literal JSON value. Useful when you want to assert an exact JSON structure without writing individual matchers for each field.

By default, matching is **exhaustive**. Use `partially(...)` to allow extra keys. `partially()` **propagates** to all nested objects automatically.

```rust
use googletest::prelude::*;
use googletest_matchers::{matches_json_literal, partially};
use serde_json::json;

let actual = json!({"name": "Alice", "age": 30});

// Exact match
expect_that!(actual, matches_json_literal!({"name": "Alice", "age": 30}));

// Partial match — extra keys ignored at all levels
let actual = json!({
    "user": {"name": "Alice", "extra_inner": true},
    "extra_outer": 1
});
expect_that!(actual, partially(matches_json_literal!({
    "user": {"name": "Alice"}
})));
```

Arrays are compared element-by-element (order and length must match):

```rust
let actual = json!({"items": [1, 2, 3]});
expect_that!(actual, matches_json_literal!({"items": [1, 2, 3]}));

// Different length fails
expect_that!(actual, not(matches_json_literal!({"items": [1, 2]})));

// Different order fails
expect_that!(actual, not(matches_json_literal!({"items": [3, 2, 1]})));
```

### `is_null()` — null value matching

Matches a `serde_json::Value` that is `null`.

```rust
let actual = json!({"deleted_at": null});
expect_that!(&actual["deleted_at"], is_null());
expect_that!(&actual["deleted_at"], not(is_null())); // would fail
```

### `partially()` — partial object matching

Wraps a matcher to switch it into partial-object mode. In this mode, expected keys must exist and match, but extra keys in the actual object are ignored.

Works with both `matches_json!` and `matches_json_literal!`, but propagation behavior differs:

| Matcher                 | `partially()` propagates to nested objects? |
| ----------------------- | ------------------------------------------- |
| `matches_json!`         | No — apply at each level                    |
| `matches_json_literal!` | Yes — applies to all nested objects         |
