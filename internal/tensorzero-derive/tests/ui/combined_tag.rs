#[derive(tensorzero_derive::TensorZeroDeserialize)]
#[serde(tag = "tag", rename_all = "snake_case")]
enum Foo {
    Bar,
    Baz,
}

fn main() {}
