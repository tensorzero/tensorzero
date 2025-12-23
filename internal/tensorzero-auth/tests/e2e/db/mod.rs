#![allow(clippy::unwrap_used, clippy::panic)]

use chrono::{Duration, Utc};
use secrecy::ExposeSecret;
use sqlx::PgPool;

use tensorzero_auth::{
    key::{TensorZeroApiKey, TensorZeroAuthError},
    postgres::{AuthResult, KeyInfo, check_key, create_key, disable_key, list_key_info},
};

#[sqlx::test]
async fn test_key_lifecycle(pool: PgPool) {
    let first_key = create_key("my_org", "my_workspace", None, &pool)
        .await
        .unwrap();
    let second_key = create_key("my_org", "my_workspace", Some("Second key"), &pool)
        .await
        .unwrap();

    assert_ne!(first_key.expose_secret(), second_key.expose_secret());

    let parsed_first_key = TensorZeroApiKey::parse(first_key.expose_secret()).unwrap();
    let first_key_info = check_key(&parsed_first_key, &pool).await.unwrap();
    let AuthResult::Success(first_key_info) = first_key_info else {
        panic!("First key should be successful");
    };
    assert_eq!(first_key_info.organization, "my_org");
    assert_eq!(first_key_info.workspace, "my_workspace");
    assert_eq!(first_key_info.description, None);
    assert_eq!(first_key_info.disabled_at, None);

    let parsed_second_key = TensorZeroApiKey::parse(second_key.expose_secret()).unwrap();
    let second_key_info = check_key(&parsed_second_key, &pool).await.unwrap();
    let AuthResult::Success(second_key_info) = second_key_info else {
        panic!("Second key should be successful: {second_key_info:?}");
    };
    assert_eq!(second_key_info.organization, "my_org");
    assert_eq!(second_key_info.workspace, "my_workspace");
    assert_eq!(second_key_info.description, Some("Second key".to_string()));
    assert_eq!(second_key_info.disabled_at, None);

    let missing_key_res = check_key(
        &TensorZeroApiKey::parse(
            "sk-t0-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .unwrap(),
        &pool,
    )
    .await
    .unwrap();

    let AuthResult::MissingKey = missing_key_res else {
        panic!("Key should be missing: {missing_key_res:?}");
    };

    let list_keys_res = list_key_info(None, None, None, &pool).await.unwrap();
    assert_eq!(
        list_keys_res,
        vec![second_key_info.clone(), first_key_info.clone()]
    );

    disable_key(&parsed_first_key.public_id, &pool)
        .await
        .unwrap();
    let now = Utc::now();

    let new_first_key_info = check_key(&parsed_first_key, &pool).await.unwrap();
    let AuthResult::Disabled(disabled_at) = new_first_key_info else {
        panic!("Key should be disabled: {new_first_key_info:?}");
    };
    assert!(
        now - disabled_at < Duration::new(1, 0).unwrap(),
        "disabled_at ({disabled_at:?}) should be within 1 second of now ({now:?})"
    );

    let new_second_key_info = check_key(&parsed_second_key, &pool).await.unwrap();
    let AuthResult::Success(key_info) = new_second_key_info else {
        panic!("Second key should still be successful: {new_second_key_info:?}");
    };
    assert_eq!(key_info.organization, "my_org");
    assert_eq!(key_info.workspace, "my_workspace");
    assert_eq!(key_info.description, Some("Second key".to_string()));
    assert_eq!(key_info.disabled_at, None);

    // Check that the first key shows up as disabled in 'list_key_info'
    let new_list_keys_res = list_key_info(None, None, None, &pool).await.unwrap();
    let disabled_first_key = KeyInfo {
        public_id: first_key_info.public_id,
        organization: first_key_info.organization,
        workspace: first_key_info.workspace,
        description: first_key_info.description,
        created_at: first_key_info.created_at,
        disabled_at: Some(disabled_at),
    };
    assert_eq!(new_list_keys_res, vec![second_key_info, disabled_first_key]);
}

fn unwrap_success(result: Result<AuthResult, TensorZeroAuthError>) -> KeyInfo {
    match result.unwrap() {
        AuthResult::Success(key_info) => key_info,
        other => panic!("Key should be successful: {other:?}"),
    }
}

#[sqlx::test]
async fn test_list_keys(pool: PgPool) {
    let first_key = create_key("my_org", "my_workspace", None, &pool)
        .await
        .unwrap();
    let second_key = create_key("my_org", "my_workspace", Some("Second key"), &pool)
        .await
        .unwrap();

    let same_org_different_workspace =
        create_key("my_org", "my_second_workspace", Some("Third key"), &pool)
            .await
            .unwrap();

    let different_org = create_key(
        "different_org",
        "different_workspace",
        Some("Fourth key"),
        &pool,
    )
    .await
    .unwrap();

    // Re-use the 'my_workspace' name in two different organizations
    let collide_workspace_name =
        create_key("different_org", "my_workspace", Some("Fifth key"), &pool)
            .await
            .unwrap();

    let first_key_info = unwrap_success(
        check_key(
            &TensorZeroApiKey::parse(first_key.expose_secret()).unwrap(),
            &pool,
        )
        .await,
    );
    let second_key_info = unwrap_success(
        check_key(
            &TensorZeroApiKey::parse(second_key.expose_secret()).unwrap(),
            &pool,
        )
        .await,
    );
    let same_org_different_workspace_info = unwrap_success(
        check_key(
            &TensorZeroApiKey::parse(same_org_different_workspace.expose_secret()).unwrap(),
            &pool,
        )
        .await,
    );
    let different_org_info = unwrap_success(
        check_key(
            &TensorZeroApiKey::parse(different_org.expose_secret()).unwrap(),
            &pool,
        )
        .await,
    );
    let collide_workspace_name_info = unwrap_success(
        check_key(
            &TensorZeroApiKey::parse(collide_workspace_name.expose_secret()).unwrap(),
            &pool,
        )
        .await,
    );

    assert_eq!(
        list_key_info(None, None, None, &pool).await.unwrap(),
        vec![
            collide_workspace_name_info.clone(),
            different_org_info.clone(),
            same_org_different_workspace_info.clone(),
            second_key_info.clone(),
            first_key_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, Some(3), None, &pool).await.unwrap(),
        vec![
            collide_workspace_name_info.clone(),
            different_org_info.clone(),
            same_org_different_workspace_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, None, Some(1), &pool).await.unwrap(),
        vec![
            different_org_info.clone(),
            same_org_different_workspace_info.clone(),
            second_key_info.clone(),
            first_key_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, Some(3), Some(1), &pool).await.unwrap(),
        vec![
            different_org_info.clone(),
            same_org_different_workspace_info.clone(),
            second_key_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("my_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![
            same_org_different_workspace_info.clone(),
            second_key_info.clone(),
            first_key_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("different_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![
            collide_workspace_name_info.clone(),
            different_org_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("missing_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![]
    );
}

#[sqlx::test]
async fn test_check_bad_key(pool: PgPool) {
    let first_key = create_key("my_org", "my_workspace", None, &pool)
        .await
        .unwrap();

    let second_key = create_key("my_org", "my_workspace", None, &pool)
        .await
        .unwrap();

    let first_parsed_key = TensorZeroApiKey::parse(first_key.expose_secret()).unwrap();
    let second_parsed_key = TensorZeroApiKey::parse(second_key.expose_secret()).unwrap();

    // Construct a key with a public id and long key hash from two different (valid) keys,
    // and verify that this is rejected by 'check_key'
    let bad_key_1 = TensorZeroApiKey::new_for_testing(
        first_parsed_key.get_public_id().to_string(),
        second_parsed_key
            .get_hashed_long_key()
            .expose_secret()
            .to_string(),
    );
    let bad_key_2 = TensorZeroApiKey::new_for_testing(
        second_parsed_key.get_public_id().to_string(),
        first_parsed_key
            .get_hashed_long_key()
            .expose_secret()
            .to_string(),
    );

    let result = check_key(&bad_key_1, &pool).await.unwrap();
    let AuthResult::MissingKey = result else {
        panic!("First bad key should be missing: {result:?}");
    };

    let result = check_key(&bad_key_2, &pool).await.unwrap();
    let AuthResult::MissingKey = result else {
        panic!("Second bad key should be missing: {result:?}");
    };
}

#[sqlx::test]
async fn test_disable_key_workflow(pool: PgPool) {
    // Create an API key and keep it for later verification
    let api_key = create_key("test_org", "test_workspace", Some("Test key"), &pool)
        .await
        .unwrap();
    let parsed_key = TensorZeroApiKey::parse(api_key.expose_secret()).unwrap();

    // Use list_key_info to get information about the key
    let key_list = list_key_info(None, None, None, &pool).await.unwrap();
    assert_eq!(key_list.len(), 1);
    let key_info = &key_list[0];
    assert_eq!(key_info.organization, "test_org");
    assert_eq!(key_info.workspace, "test_workspace");
    assert_eq!(key_info.description, Some("Test key".to_string()));
    assert_eq!(key_info.disabled_at, None);

    // Disable the key using the public_id from list_key_info
    disable_key(&key_info.public_id, &pool).await.unwrap();
    let now = Utc::now();

    // Run list_key_info again and verify the key shows as disabled
    let key_list_after_disable = list_key_info(None, None, None, &pool).await.unwrap();
    assert_eq!(key_list_after_disable.len(), 1);
    let disabled_key_info = &key_list_after_disable[0];
    assert_eq!(disabled_key_info.public_id, key_info.public_id);
    assert!(
        disabled_key_info.disabled_at.is_some(),
        "Key should have a disabled_at timestamp"
    );
    let disabled_at = disabled_key_info.disabled_at.unwrap();
    assert!(
        now - disabled_at < Duration::new(1, 0).unwrap(),
        "disabled_at ({disabled_at:?}) should be within 1 second of now ({now:?})"
    );

    // Try to check the API key from the first step and verify it returns Disabled
    let check_result = check_key(&parsed_key, &pool).await.unwrap();
    let AuthResult::Disabled(check_disabled_at) = check_result else {
        panic!("Key should be disabled after calling disable_key: {check_result:?}");
    };
    assert_eq!(
        check_disabled_at, disabled_at,
        "Timestamps should match between list_key_info and check_key"
    );
}
