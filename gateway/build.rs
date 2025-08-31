use serde_json::{json, Value};
use std::{env, fs, path::PathBuf};

fn sanitize_yaml_bigints(yaml: &str) -> String {
    yaml
        .replace("-9223372036854776000", "-9223372036854775808") // i64::MIN
        .replace("9223372036854776000", "9223372036854775807")  // i64::MAX
}

fn normalize_nullable(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if let (Some(t), Some(n)) = (m.get("type"), m.get("nullable")) {
                if n == &Value::Bool(true) {
                    if let Some(ts) = t.as_str() {
                        m.insert("type".to_string(), json!([ts, "null"]));
                        m.remove("nullable");
                    }
                }
            }
            for x in m.values_mut() {
                normalize_nullable(x);
            }
        }
        Value::Array(arr) => {
            for x in arr {
                normalize_nullable(x);
            }
        }
        _ => {}
    }
}

fn main() {
    // Re-run build script if the URL env var changes.
    println!("cargo:rerun-if-env-changed=OPENAI_OPENAPI_URL");

    // Source URL (defaults to Stainless documented spec).
    let url = env::var("OPENAI_OPENAPI_URL").unwrap_or_else(|_| {
        "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml".to_string()
    });

    let resp = reqwest::blocking::get(&url)
        .unwrap_or_else(|e| panic!("failed to fetch OpenAI OpenAPI from {}: {}", url, e));
    let body = resp
        .text()
        .unwrap_or_else(|e| panic!("failed to read response body from {}: {}", url, e));

    // Parse into serde_json::Value (YAML or JSON).
    let mut spec: Value = if body.trim_start().starts_with('{') {
        serde_json::from_str(&body).expect("invalid JSON from OpenAI/Stainless")
    } else {
        // Treat as YAML (the documented spec is .yml); clamp big integers first.
        let sanitized = sanitize_yaml_bigints(&body);
        let y: serde_yaml::Value =
            serde_yaml::from_str(&sanitized).expect("invalid YAML from OpenAI/Stainless");
        serde_json::to_value(y).expect("yaml→json conversion failed")
    };

    // Normalize nullable fields.
    normalize_nullable(&mut spec);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let out_path = out_dir.join("openapi.json");
    fs::write(&out_path, serde_json::to_vec_pretty(&spec).unwrap())
        .expect("write OUT_DIR/openapi.json failed");
}
