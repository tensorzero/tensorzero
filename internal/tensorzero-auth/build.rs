fn main() {
    println!("cargo:rerun-if-changed=src/postgres/migrations");
}
