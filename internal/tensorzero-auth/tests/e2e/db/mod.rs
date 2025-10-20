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
        vec![first_key_info.clone(), second_key_info.clone()]
    );

    disable_key(&parsed_first_key, &pool).await.unwrap();
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
        id: first_key_info.id,
        organization: first_key_info.organization,
        workspace: first_key_info.workspace,
        description: first_key_info.description,
        disabled_at: Some(disabled_at),
    };
    assert_eq!(new_list_keys_res, vec![disabled_first_key, second_key_info]);
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
            first_key_info.clone(),
            second_key_info.clone(),
            same_org_different_workspace_info.clone(),
            different_org_info.clone(),
            collide_workspace_name_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, Some(3), None, &pool).await.unwrap(),
        vec![
            first_key_info.clone(),
            second_key_info.clone(),
            same_org_different_workspace_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, None, Some(1), &pool).await.unwrap(),
        vec![
            second_key_info.clone(),
            same_org_different_workspace_info.clone(),
            different_org_info.clone(),
            collide_workspace_name_info.clone()
        ]
    );

    assert_eq!(
        list_key_info(None, Some(3), Some(1), &pool).await.unwrap(),
        vec![
            second_key_info.clone(),
            same_org_different_workspace_info.clone(),
            different_org_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("my_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![
            first_key_info.clone(),
            second_key_info.clone(),
            same_org_different_workspace_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("different_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![
            different_org_info.clone(),
            collide_workspace_name_info.clone(),
        ]
    );

    assert_eq!(
        list_key_info(Some("missing_org".to_string()), None, None, &pool)
            .await
            .unwrap(),
        vec![]
    );
}
