#![cfg(feature = "e2e_tests")]
#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::time::Duration;

use tensorzero::{Client as TensorZeroClient, ClientExt, WriteConfigRequest};
use tensorzero_core::config::UninitializedConfig;
use uuid::Uuid;

async fn test_write_config_impl(client: TensorZeroClient) {
    let id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.client_test_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: UninitializedConfig = toml::from_str(&config_toml).unwrap();

    let mut tags = HashMap::new();
    tags.insert("env".to_string(), "test".to_string());
    tags.insert("version".to_string(), "1.0".to_string());

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let request = WriteConfigRequest {
        config: stored_config,
        extra_templates: extra_templates.clone(),
        tags: tags.clone(),
    };

    let response = client.write_config(request).await;
    assert!(
        response.is_ok(),
        "write_config should succeed for a valid config snapshot, got: {:?}",
        response.err()
    );

    let write_response = response.unwrap();
    assert!(
        !write_response.hash.is_empty(),
        "write_config should return a non-empty hash for the stored snapshot"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    let get_response = client.get_config_snapshot(Some(&write_response.hash)).await;
    assert!(
        get_response.is_ok(),
        "get_config_snapshot should succeed for the newly written hash, got: {:?}",
        get_response.err()
    );

    let config_snapshot = get_response.unwrap();
    assert_eq!(
        config_snapshot.hash, write_response.hash,
        "get_config_snapshot should return the same hash that write_config returned"
    );
    assert_eq!(
        config_snapshot.extra_templates.get("test_template"),
        Some(&"Hello {{name}}!".to_string()),
        "The persisted config snapshot should include the submitted extra template"
    );
    assert_eq!(
        config_snapshot.tags.get("env"),
        Some(&"test".to_string()),
        "The persisted config snapshot should include the `env` tag"
    );
    assert_eq!(
        config_snapshot.tags.get("version"),
        Some(&"1.0".to_string()),
        "The persisted config snapshot should include the `version` tag"
    );
}

tensorzero::make_gateway_test_functions!(test_write_config_impl);

async fn test_write_config_tag_merging_impl(client: TensorZeroClient) {
    let id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.client_tag_merge_metric_{id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: UninitializedConfig = toml::from_str(&config_toml).unwrap();

    let mut tags1 = HashMap::new();
    tags1.insert("key1".to_string(), "value1".to_string());
    tags1.insert("key2".to_string(), "original".to_string());

    let request1 = WriteConfigRequest {
        config: stored_config.clone(),
        extra_templates: HashMap::new(),
        tags: tags1,
    };

    let response1 = client.write_config(request1).await.unwrap();
    let hash = response1.hash.clone();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut tags2 = HashMap::new();
    tags2.insert("key2".to_string(), "updated".to_string());
    tags2.insert("key3".to_string(), "new".to_string());

    let request2 = WriteConfigRequest {
        config: stored_config,
        extra_templates: HashMap::new(),
        tags: tags2,
    };

    let response2 = client.write_config(request2).await.unwrap();
    assert_eq!(
        response2.hash, hash,
        "Writing the same config content should return the same hash"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    let config_snapshot = client.get_config_snapshot(Some(&hash)).await.unwrap();

    assert_eq!(
        config_snapshot.tags.get("key1"),
        Some(&"value1".to_string()),
        "Tag merge should preserve existing keys not present in the update"
    );
    assert_eq!(
        config_snapshot.tags.get("key2"),
        Some(&"updated".to_string()),
        "Tag merge should overwrite an existing key with the new value"
    );
    assert_eq!(
        config_snapshot.tags.get("key3"),
        Some(&"new".to_string()),
        "Tag merge should add new keys introduced by the second write"
    );
}

tensorzero::make_gateway_test_functions!(test_write_config_tag_merging_impl);
