use crate::db::PostgresConnectionInfo;
use chrono::Duration;
use sqlx::PgPool;
use tensorzero_core::db::RateLimitQueries;
use tensorzero_core::{db::ConsumeTicketsRequest, rate_limiting::ActiveRateLimitKey};

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_consume_tickets(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);
    let get_consume_tickets_requests = || {
        vec![ConsumeTicketsRequest {
            key: ActiveRateLimitKey("foo".to_string()),
            requested: 1,
            capacity: 5,
            refill_amount: 1,
            refill_interval: Duration::new(1, 0).unwrap(),
        }]
    };

    let mut handles = Vec::new();
    for _ in 0..10 {
        let conn_clone = conn.clone();
        let requests = get_consume_tickets_requests();
        let handle =
            tokio::spawn(async move { conn_clone.consume_tickets(requests).await.unwrap() });
        handles.push(handle);
    }

    let mut successful_count = 0;
    for handle in handles {
        let receipt = handle.await.unwrap();
        if receipt[0].success {
            successful_count += 1;
        }
    }

    assert_eq!(successful_count, 5);
}
