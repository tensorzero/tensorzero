//! Shared test logic for EpisodeQueries implementations (ClickHouse and Postgres).

#![expect(clippy::print_stdout)]

use tensorzero_core::db::EpisodeQueries;

async fn test_query_episode_table(conn: impl EpisodeQueries) {
    // Test basic pagination
    let episodes = conn.query_episode_table(10, None, None).await.unwrap();
    println!("First 10 episodes: {episodes:#?}");

    assert_eq!(episodes.len(), 10, "Should return 10 episodes");

    // Verify episodes are in descending order by episode_id
    for i in 1..episodes.len() {
        assert!(
            episodes[i - 1].episode_id > episodes[i].episode_id,
            "Episodes should be in descending order by episode_id"
        );
    }

    // Test pagination with before (this should return 10 since there are lots of episodes)
    let episodes2 = conn
        .query_episode_table(10, Some(episodes[episodes.len() - 1].episode_id), None)
        .await
        .unwrap();
    assert_eq!(
        episodes2.len(),
        10,
        "Should return 10 episodes before cursor"
    );

    // Test pagination with after (should return 0 for most recent episodes)
    let episodes3 = conn
        .query_episode_table(10, None, Some(episodes[0].episode_id))
        .await
        .unwrap();
    assert_eq!(
        episodes3.len(),
        0,
        "Should return 0 episodes after the most recent"
    );

    let episodes3 = conn
        .query_episode_table(10, None, Some(episodes[4].episode_id))
        .await
        .unwrap();
    assert_eq!(
        episodes3.len(),
        4,
        "Should return 4 episodes after the 5th most recent"
    );

    // Test that before and after together throws error
    let result = conn
        .query_episode_table(
            10,
            Some(episodes[0].episode_id),
            Some(episodes[0].episode_id),
        )
        .await;
    assert!(
        result.is_err(),
        "Should error when both before and after are specified"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both before and after"),
        "Error message should mention that both before and after cannot be specified"
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
make_db_test!(test_query_episode_table);

async fn test_query_episode_table_bounds(conn: impl EpisodeQueries) {
    let bounds = conn.query_episode_table_bounds().await.unwrap();
    println!("Episode table bounds: {bounds:#?}");

    // Verify bounds structure
    assert!(bounds.first_id.is_some(), "Should have a first_id");
    assert!(bounds.last_id.is_some(), "Should have a last_id");

    assert_eq!(
        bounds.first_id.unwrap().to_string(),
        "0192ced0-947e-74b3-a3d7-02fd2c54d637",
        "first_id should match the expected value from test fixtures"
    );
    // The end and count are ~guaranteed to be trampled here since other tests do inference.
    // We just assert that we are returning something reasonable.
    assert!(bounds.count > 0, "Should have a count greater than 0");
    assert!(
        bounds.last_id.unwrap() > bounds.first_id.unwrap(),
        "Should have a last_id greater than first_id"
    );
}
make_db_test!(test_query_episode_table_bounds);
