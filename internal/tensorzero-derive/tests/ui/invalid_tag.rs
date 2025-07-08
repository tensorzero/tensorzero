#[derive(tensorzero_derive::TensorZeroDeserialize)]
#[serde(tag = 123)]
enum Foo {
    Bar,
    Baz,
}

fn main() {}
