#![expect(clippy::print_stdout)]
use tensorzero_core::db::{SelectQueries, clickhouse::test_helpers::get_clickhouse};

#[tokio::test]
async fn test_clickhouse_query_episode_table() {
    let clickhouse = get_clickhouse().await;

    // Test basic pagination
    let episodes = clickhouse
        .query_episode_table(10, None, None)
        .await
        .unwrap();
    println!("First 10 episodes: {episodes:#?}");

    assert_eq!(episodes.len(), 10);
    println!("First 10 episodes: {episodes:#?}");

    // Verify episodes are in descending order by episode_id
    for i in 1..episodes.len() {
        assert!(
            episodes[i - 1].episode_id > episodes[i].episode_id,
            "Episodes should be in descending order by episode_id"
        );
    }

    // Test pagination with before (this should return 10 since there are lots of episodes)
    let episodes2 = clickhouse
        .query_episode_table(10, Some(episodes[episodes.len() - 1].episode_id), None)
        .await
        .unwrap();
    assert_eq!(episodes2.len(), 10);

    // Test pagination with after (should return 0 for most recent episodes)
    let episodes3 = clickhouse
        .query_episode_table(10, None, Some(episodes[0].episode_id))
        .await
        .unwrap();
    assert_eq!(episodes3.len(), 0);
    let episodes3 = clickhouse
        .query_episode_table(10, None, Some(episodes[4].episode_id))
        .await
        .unwrap();
    assert_eq!(episodes3.len(), 4);

    // Test that before and after together throws error
    let result = clickhouse
        .query_episode_table(
            10,
            Some(episodes[0].episode_id),
            Some(episodes[0].episode_id),
        )
        .await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both before and after")
    );

    // Verify each episode has valid data
    for episode in &episodes {
        assert!(episode.count > 0, "Episode count should be greater than 0");

        // Start time should be before or equal to end time
        assert!(
            episode.start_time <= episode.end_time,
            "Start time {:?} should be before or equal to end time {:?} for episode {}",
            episode.start_time,
            episode.end_time,
            episode.episode_id
        );
    }
}

#[tokio::test]
async fn test_clickhouse_query_episode_table_bounds() {
    let clickhouse = get_clickhouse().await;
    let bounds = clickhouse.query_episode_table_bounds().await.unwrap();
    println!("Episode table bounds: {bounds:#?}");

    // Verify bounds structure
    assert!(bounds.first_id.is_some(), "Should have a first_id");
    assert!(bounds.last_id.is_some(), "Should have a last_id");

    assert_eq!(
        bounds.first_id.unwrap().to_string(),
        "0192ced0-947e-74b3-a3d7-02fd2c54d637"
    );
    // The end and count are ~guaranteed to be trampled here since other tests do inference.
    // We test in UI e2e tests that the behavior is as expected
    // assert_eq!(
    //     bounds.last_id.unwrap().to_string(),
    //     "019926fd-1a06-7fe2-b7f4-23220893d62c"
    // );
    // assert_eq!(bounds.count, 20002095);
}
