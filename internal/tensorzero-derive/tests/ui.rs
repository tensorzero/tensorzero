// Tests various errors emitted by the derive macro.
#[ignore]
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
