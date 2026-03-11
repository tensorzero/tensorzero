#!/usr/bin/env python3
"""Tests for check_use_placement.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_use_placement import check_source, count_braces

# ── count_braces ──────────────────────────────────────────────────────


def test_count_braces_simple():
    assert count_braces("fn foo() {") == 1
    assert count_braces("}") == -1
    assert count_braces("let x = 5;") == 0
    assert count_braces("fn foo() {}") == 0


def test_count_braces_string_literals():
    assert count_braces('let s = "hello { world";') == 0
    assert count_braces('let s = "}{";') == 0


def test_count_braces_comments():
    assert count_braces("foo(); // { not counted") == 0
    assert count_braces("{ // }") == 1


def test_count_braces_escaped_quotes():
    assert count_braces(r'let s = "hello \"}\"{";') == 0


# ── Use after fn (should flag) ───────────────────────────────────────


def test_use_after_fn_at_module_level():
    source = """\
fn foo() {
}

use std::io;
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 4
    assert "use std::io" in violations[0][1]


def test_use_after_fn_in_test_module():
    source = """\
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() {
        assert!(true);
    }

    use std::collections::HashSet;

    #[test]
    fn test_bar() {
    }
}
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 10
    assert "HashSet" in violations[0][1]


def test_multiple_uses_after_fn():
    source = """\
mod tests {
    use super::*;

    fn helper() {}

    use std::io;
    use std::fmt;
}
"""
    violations = check_source(source)
    assert len(violations) == 2


def test_use_after_pub_fn():
    source = """\
pub fn foo() {}

use std::io;
"""
    violations = check_source(source)
    assert len(violations) == 1


def test_use_after_async_fn():
    source = """\
async fn foo() {}

use std::io;
"""
    violations = check_source(source)
    assert len(violations) == 1


# ── Correct placement (should NOT flag) ──────────────────────────────


def test_use_before_fn_at_module_level():
    source = """\
use std::io;
use std::fmt;

fn foo() {}
fn bar() {}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_use_before_fn_in_test_module():
    source = """\
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_foo() {
        assert!(true);
    }

    #[test]
    fn test_bar() {
    }
}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_pub_use_after_fn_allowed():
    """pub use re-exports should not be flagged."""
    source = """\
pub(crate) fn deprecation_warning(msg: &str) {}

pub use content::{System, Text};
pub use role::Role;
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_pub_crate_use_after_fn_allowed():
    source = """\
fn foo() {}

pub(crate) use bar::Baz;
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_use_inside_fn_body_flagged():
    """use inside a function body should be flagged — move to module level."""
    source = """\
fn foo() {
    use std::fmt::Write;
    let mut s = String::new();
}

fn bar() {
    use std::io::Read;
}
"""
    violations = check_source(source)
    assert len(violations) == 2
    assert violations[0][0] == 2
    assert violations[0][2] == "inside_fn"
    assert violations[1][0] == 7
    assert violations[1][2] == "inside_fn"


def test_impl_block_fns_dont_poison_later_mod():
    """fns inside impl blocks should not cause false positives in later mod tests."""
    source = """\
impl Foo {
    fn method_a(&self) {}
    fn method_b(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() {}
}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_nested_mod_resets_scope():
    """A mod block opens a new scope — fn in outer scope doesn't affect inner."""
    source = """\
fn outer_fn() {}

mod inner {
    use std::io;

    fn inner_fn() {}
}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_use_in_separate_mod_blocks():
    source = """\
mod a {
    fn foo() {}
}

mod b {
    use std::io;
    fn bar() {}
}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_fn_after_closing_brace_then_use():
    """fn at file scope after impl block, then use — should flag."""
    source = """\
use std::fmt;

impl Foo {
    fn method(&self) {}
}

fn standalone() {}

use std::io;
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 9


def test_empty_file():
    assert check_source("") == []


def test_only_comments():
    source = """\
// just a comment
/* block comment */
"""
    assert check_source(source) == []


def test_use_after_fn_with_braces_in_string():
    """fn body containing string with braces should not confuse brace tracking."""
    source = """\
mod tests {
    use super::*;

    fn helper() {
        let s = "some { string } with braces";
    }

    use std::io;
}
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 8


# ── Use inside function body (should flag) ───────────────────────


def test_use_inside_test_fn():
    """use inside a #[test] function body should be flagged."""
    source = """\
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() {
        use std::collections::HashMap;
        let m = HashMap::new();
    }
}
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 7
    assert violations[0][2] == "inside_fn"


def test_use_inside_async_test_fn():
    """use inside an async test function body should be flagged."""
    source = """\
#[tokio::test]
async fn test_foo() {
    use reqwest::Client;
    let c = Client::new();
}
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 3
    assert violations[0][2] == "inside_fn"


def test_use_inside_fn_multiline_body():
    """use inside fn with brace on next line should be flagged."""
    source = """\
fn foo()
{
    use std::io;
}
"""
    violations = check_source(source)
    assert len(violations) == 1
    assert violations[0][0] == 3
    assert violations[0][2] == "inside_fn"


def test_use_inside_fn_not_confused_with_mod():
    """use at top of mod block should not be flagged as inside fn body."""
    source = """\
fn standalone() {}

mod inner {
    use std::io;

    fn inner_fn() {}
}
"""
    violations = check_source(source)
    assert len(violations) == 0


def test_one_line_fn_body_does_not_poison_scope():
    """fn foo() {} on one line should not make later use at same scope a fn-body violation."""
    source = """\
fn foo() {}

mod tests {
    use std::io;
}
"""
    violations = check_source(source)
    assert len(violations) == 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
