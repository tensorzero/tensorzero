use crate::clickhouse::migration_manager::migrations::{create_table_engine, create_replacing_table_engine};

#[test]
fn test_engine_creation() {
    let engine = create_table_engine(true, "test_cluster", "test_table");
    println!("Table engine: {}", engine);
    
    let replacing_engine = create_replacing_table_engine(true, "test_cluster", "test_table", Some("version"));
    println!("Replacing engine: {}", replacing_engine);
    
    // Test unescaping
    let unescaped = engine.replace("{{", "{").replace("}}", "}");
    println!("Unescaped: {}", unescaped);
}
