//! Shared test logic for EpisodeQueries implementations (ClickHouse and Postgres).

#![expect(clippy::print_stdout)]

use tensorzero_core::config::Config;
use tensorzero_core::db::EpisodeQueries;
use tensorzero_core::db::clickhouse::query_builder::TagFilter;
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    InferenceFilter, TagComparisonOperator,
};

async fn test_query_episode_table(conn: impl EpisodeQueries) {
    let config = Config::default();

    // Test basic pagination
    let episodes = conn
        .query_episode_table(&config, 10, None, None, None, None)
        .await
        .unwrap();
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
        .query_episode_table(
            &config,
            10,
            Some(episodes[episodes.len() - 1].episode_id),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        episodes2.len(),
        10,
        "Should return 10 episodes before cursor"
    );

    // Test pagination with after (should return 0 for most recent episodes)
    let episodes3 = conn
        .query_episode_table(&config, 10, None, Some(episodes[0].episode_id), None, None)
        .await
        .unwrap();
    assert_eq!(
        episodes3.len(),
        0,
        "Should return 0 episodes after the most recent"
    );

    let episodes3 = conn
        .query_episode_table(&config, 10, None, Some(episodes[4].episode_id), None, None)
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
            &config,
            10,
            Some(episodes[0].episode_id),
            Some(episodes[0].episode_id),
            None,
            None,
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

async fn test_query_episode_table_with_function_name(conn: impl EpisodeQueries) {
    let config = Config::default();

    let episodes = conn
        .query_episode_table(
            &config,
            10,
            None,
            None,
            Some("extract_entities".to_string()),
            None,
        )
        .await
        .unwrap();
    println!("Episodes for extract_entities: {episodes:#?}");

    assert!(
        !episodes.is_empty(),
        "Should return at least one episode for function_name `extract_entities`"
    );

    for episode in &episodes {
        assert!(episode.count > 0, "Episode count should be greater than 0");
        assert!(
            episode.start_time <= episode.end_time,
            "Start time should be before or equal to end time"
        );
    }

    // A nonexistent function should return no episodes
    let empty_episodes = conn
        .query_episode_table(
            &config,
            10,
            None,
            None,
            Some("nonexistent_function_12345".to_string()),
            None,
        )
        .await
        .unwrap();
    assert!(
        empty_episodes.is_empty(),
        "Should return no episodes for nonexistent function_name"
    );
}
make_db_test!(test_query_episode_table_with_function_name);

async fn test_query_episode_table_with_function_name_and_filter(conn: impl EpisodeQueries) {
    let config = Config::default();

    let filter = InferenceFilter::Tag(TagFilter {
        key: "tensorzero::evaluation_name".to_string(),
        value: "entity_extraction".to_string(),
        comparison_operator: TagComparisonOperator::Equal,
    });

    let episodes = conn
        .query_episode_table(
            &config,
            10,
            None,
            None,
            Some("extract_entities".to_string()),
            Some(filter),
        )
        .await
        .unwrap();
    println!("Episodes for extract_entities with tag filter: {episodes:#?}");

    assert!(
        !episodes.is_empty(),
        "Should return at least one episode matching function_name + tag filter"
    );

    for episode in &episodes {
        assert!(episode.count > 0, "Episode count should be greater than 0");
        assert!(
            episode.start_time <= episode.end_time,
            "Start time should be before or equal to end time"
        );
    }

    // Verify that counts reflect full episode stats (not just filtered inferences)
    // by comparing with unfiltered results for the same function.
    // Paginate through unfiltered episodes until we've found all filtered episode IDs.
    let filtered_ids: std::collections::HashSet<_> =
        episodes.iter().map(|e| e.episode_id).collect();
    let mut unfiltered_map = std::collections::HashMap::new();
    let mut before_cursor = None;
    loop {
        let page = conn
            .query_episode_table(
                &config,
                100,
                before_cursor,
                None,
                Some("extract_entities".to_string()),
                None,
            )
            .await
            .unwrap();
        if page.is_empty() {
            break;
        }
        before_cursor = Some(page.last().unwrap().episode_id);
        for ep in page {
            unfiltered_map.insert(ep.episode_id, ep);
        }
        if filtered_ids
            .iter()
            .all(|id| unfiltered_map.contains_key(id))
        {
            break;
        }
    }

    // The filtered result should be a subset of the unfiltered result
    for filtered_ep in &episodes {
        let unfiltered_ep = unfiltered_map.get(&filtered_ep.episode_id);
        assert!(
            unfiltered_ep.is_some(),
            "Filtered episode {} should also appear in unfiltered results",
            filtered_ep.episode_id
        );
        // Episode stats should match (count, start_time, end_time, last_inference_id)
        // because filtering only selects which episodes to return, not which inferences to count
        let unfiltered_ep = unfiltered_ep.unwrap();
        assert_eq!(
            filtered_ep.count, unfiltered_ep.count,
            "Episode {} count should match between filtered and unfiltered queries",
            filtered_ep.episode_id
        );
        assert_eq!(
            filtered_ep.start_time, unfiltered_ep.start_time,
            "Episode {} start_time should match between filtered and unfiltered queries",
            filtered_ep.episode_id
        );
        assert_eq!(
            filtered_ep.end_time, unfiltered_ep.end_time,
            "Episode {} end_time should match between filtered and unfiltered queries",
            filtered_ep.episode_id
        );
        assert_eq!(
            filtered_ep.last_inference_id, unfiltered_ep.last_inference_id,
            "Episode {} last_inference_id should match between filtered and unfiltered queries",
            filtered_ep.episode_id
        );
    }
}
make_db_test!(test_query_episode_table_with_function_name_and_filter);
