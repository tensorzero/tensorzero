//! Valkey-backed rate limiting implementation.

use async_trait::async_trait;
use redis::aio::ConnectionLike;
use serde::Deserialize;

use crate::db::{
    ConsumeTicketsReceipt, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsReceipt,
    ReturnTicketsRequest,
};
use crate::error::{Error, ErrorDetails};
use crate::rate_limiting::{ActiveRateLimitKey, RateLimitInterval};

use super::ValkeyConnectionInfo;

// Important: these types must match the response types for the Lua functions in tensorzero-core/src/db/valkey/lua/tensorzero_ratelimit.lua.
// Lint.IfEdited()

/// Response from tensorzero_consume_tickets_v2 Lua function
#[derive(Debug, Deserialize)]
struct ConsumeTicketsResponse {
    key: String,
    success: bool,
    remaining: i64,
    consumed: i64,
}

/// Response from tensorzero_return_tickets_v2 Lua function
#[derive(Debug, Deserialize)]
struct ReturnTicketsResponse {
    key: String,
    balance: i64,
}

/// Response from tensorzero_get_balance_v2 Lua function
#[derive(Debug, Deserialize)]
struct GetBalanceResponse {
    balance: i64,
}

// Lint.ThenEdit(tensorzero-core/src/db/valkey/lua/tensorzero_ratelimit.lua)

/// Execute consume_tickets against any async Redis-compatible connection.
async fn execute_consume_tickets<C: ConnectionLike>(
    conn: &mut C,
    requests: &[ConsumeTicketsRequest],
) -> Result<Vec<ConsumeTicketsReceipt>, Error> {
    // Build keys array
    let keys: Vec<&str> = requests.iter().map(|r| r.key.0.as_str()).collect();

    // Build args array: [requested, capacity, refill_amount, refill_interval_micros] per key
    let mut args: Vec<i64> = Vec::with_capacity(requests.len() * 4);
    for req in requests {
        args.push(req.requested as i64);
        args.push(req.capacity as i64);
        args.push(req.refill_amount as i64);
        args.push(req.refill_interval.to_microseconds() as i64);
    }

    // Call the versioned Valkey function
    let mut cmd = redis::cmd("FCALL");
    cmd.arg("tensorzero_consume_tickets_v2").arg(keys.len());
    for key in &keys {
        cmd.arg(*key);
    }
    for arg in &args {
        cmd.arg(*arg);
    }

    let result: String = cmd.query_async(conn).await?;

    // Parse JSON response
    let responses: Vec<ConsumeTicketsResponse> = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::ValkeyQuery {
            message: format!("Failed to parse consume_tickets response: {e}"),
        })
    })?;

    // Convert to receipts
    let receipts = responses
        .into_iter()
        .map(|r| ConsumeTicketsReceipt {
            key: ActiveRateLimitKey::new(r.key),
            success: r.success,
            tickets_remaining: r.remaining.max(0) as u64,
            tickets_consumed: r.consumed.max(0) as u64,
        })
        .collect();

    Ok(receipts)
}

/// Execute return_tickets against any async Redis-compatible connection.
async fn execute_return_tickets<C: ConnectionLike>(
    conn: &mut C,
    requests: &[ReturnTicketsRequest],
) -> Result<Vec<ReturnTicketsReceipt>, Error> {
    // Build keys array
    let keys: Vec<&str> = requests.iter().map(|r| r.key.0.as_str()).collect();

    // Build args array: [returned, capacity, refill_amount, refill_interval_micros] per key
    let mut args: Vec<i64> = Vec::with_capacity(requests.len() * 4);
    for req in requests {
        args.push(req.returned as i64);
        args.push(req.capacity as i64);
        args.push(req.refill_amount as i64);
        args.push(req.refill_interval.to_microseconds() as i64);
    }

    // Call the versioned Valkey function
    let mut cmd = redis::cmd("FCALL");
    cmd.arg("tensorzero_return_tickets_v2").arg(keys.len());
    for key in &keys {
        cmd.arg(*key);
    }
    for arg in &args {
        cmd.arg(*arg);
    }

    let result: String = cmd.query_async(conn).await?;

    // Parse JSON response
    let responses: Vec<ReturnTicketsResponse> = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::ValkeyQuery {
            message: format!("Failed to parse return_tickets response: {e}"),
        })
    })?;

    // Convert to receipts
    let receipts = responses
        .into_iter()
        .map(|r| ReturnTicketsReceipt {
            key: ActiveRateLimitKey(r.key),
            balance: r.balance.max(0) as u64,
        })
        .collect();

    Ok(receipts)
}

/// Execute get_balance against any async Redis-compatible connection.
async fn execute_get_balance<C: ConnectionLike>(
    conn: &mut C,
    key: &str,
    capacity: u64,
    refill_amount: u64,
    refill_interval: RateLimitInterval,
) -> Result<u64, Error> {
    // Use FCALL_RO for read-only operations (works with replicas)
    let result: String = redis::cmd("FCALL_RO")
        .arg("tensorzero_get_balance_v2")
        .arg(1) // number of keys
        .arg(key)
        .arg(capacity as i64)
        .arg(refill_amount as i64)
        .arg(refill_interval.to_microseconds() as i64)
        .query_async(conn)
        .await?;

    // Parse JSON response
    let response: GetBalanceResponse = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::ValkeyQuery {
            message: format!("Failed to parse get_balance response: {e}"),
        })
    })?;

    Ok(response.balance.max(0) as u64)
}

#[async_trait]
impl RateLimitQueries for ValkeyConnectionInfo {
    async fn consume_tickets(
        &self,
        requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReceipt>, Error> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let connection = self.get_connection().ok_or_else(|| {
            Error::new(ErrorDetails::ValkeyConnection {
                message: "Valkey connection is disabled".to_string(),
            })
        })?;

        let mut conn = connection.clone();
        execute_consume_tickets(&mut conn, requests).await
    }

    async fn return_tickets(
        &self,
        requests: Vec<ReturnTicketsRequest>,
    ) -> Result<Vec<ReturnTicketsReceipt>, Error> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let connection = self.get_connection().ok_or_else(|| {
            Error::new(ErrorDetails::ValkeyConnection {
                message: "Valkey connection is disabled".to_string(),
            })
        })?;

        let mut conn = connection.clone();
        execute_return_tickets(&mut conn, &requests).await
    }

    async fn get_balance(
        &self,
        key: &str,
        capacity: u64,
        refill_amount: u64,
        refill_interval: RateLimitInterval,
    ) -> Result<u64, Error> {
        let connection = self.get_connection().ok_or_else(|| {
            Error::new(ErrorDetails::ValkeyConnection {
                message: "Valkey connection is disabled".to_string(),
            })
        })?;

        let mut conn = connection.clone();
        execute_get_balance(&mut conn, key, capacity, refill_amount, refill_interval).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redis_test::{MockCmd, MockRedisConnection};

    // ===== EMPTY REQUEST TESTS =====

    #[tokio::test]
    async fn test_consume_tickets_empty_requests() {
        let client = ValkeyConnectionInfo::new_disabled();
        let result = client.consume_tickets(&[]).await;
        assert!(result.is_ok(), "empty requests should succeed");
        assert!(
            result.unwrap().is_empty(),
            "empty requests should return empty vec"
        );
    }

    #[tokio::test]
    async fn test_return_tickets_empty_requests() {
        let client = ValkeyConnectionInfo::new_disabled();
        let result = client.return_tickets(vec![]).await;
        assert!(result.is_ok(), "empty requests should succeed");
        assert!(
            result.unwrap().is_empty(),
            "empty requests should return empty vec"
        );
    }

    // ===== DISABLED CONNECTION TESTS =====

    #[tokio::test]
    async fn test_consume_tickets_disabled_connection() {
        let client = ValkeyConnectionInfo::new_disabled();
        let requests = vec![ConsumeTicketsRequest {
            key: ActiveRateLimitKey::new("test_key".to_string()),
            requested: 10,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Second,
        }];

        let result = client.consume_tickets(&requests).await;
        let err = result.expect_err("disabled connection should return error");
        assert!(
            matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
            "error should be ValkeyConnection"
        );
    }

    #[tokio::test]
    async fn test_return_tickets_disabled_connection() {
        let client = ValkeyConnectionInfo::new_disabled();
        let requests = vec![ReturnTicketsRequest {
            key: ActiveRateLimitKey::new("test_key".to_string()),
            returned: 5,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Second,
        }];

        let result = client.return_tickets(requests).await;
        assert!(result.is_err(), "disabled connection should return error");
        let err = result.err().unwrap();
        assert!(
            matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
            "error should be ValkeyConnection"
        );
    }

    #[tokio::test]
    async fn test_get_balance_disabled_connection() {
        let client = ValkeyConnectionInfo::new_disabled();

        let result = client
            .get_balance("test_key", 100, 10, RateLimitInterval::Second)
            .await;
        let err = result.expect_err("disabled connection should return error");
        assert!(
            matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
            "error should be ValkeyConnection"
        );
    }

    // ===== MOCK CONNECTION TESTS =====
    // These tests verify that we send the correct Redis commands

    #[tokio::test]
    async fn test_consume_tickets_sends_correct_command() {
        // Expected command: FCALL tensorzero_consume_tickets_v2 1 test_key 10 100 5 1000000
        let expected_response =
            r#"[{"key":"test_key","success":true,"remaining":90,"consumed":10}]"#;
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("FCALL")
                .arg("tensorzero_consume_tickets_v2")
                .arg(1) // numkeys
                .arg("test_key")
                .arg(10i64) // requested
                .arg(100i64) // capacity
                .arg(5i64) // refill_amount
                .arg(1_000_000i64), // refill_interval (1 second in micros)
            Ok(expected_response),
        )])
        .assert_all_commands_consumed();

        let requests = vec![ConsumeTicketsRequest {
            key: ActiveRateLimitKey::new("test_key".to_string()),
            requested: 10,
            capacity: 100,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Second,
        }];

        let result = execute_consume_tickets(&mut mock, &requests).await.unwrap();
        assert_eq!(result.len(), 1, "should return one receipt");
        assert!(result[0].success, "should be successful");
        assert_eq!(result[0].tickets_consumed, 10, "should consume 10 tickets");
        assert_eq!(
            result[0].tickets_remaining, 90,
            "should have 90 tickets remaining"
        );
    }

    #[tokio::test]
    async fn test_consume_tickets_multiple_keys_sends_correct_command() {
        // Expected: FCALL tensorzero_consume_tickets_v2 2 key1 key2 10 100 5 1000000 20 200 10 60000000
        let expected_response = r#"[{"key":"key1","success":true,"remaining":90,"consumed":10},{"key":"key2","success":true,"remaining":180,"consumed":20}]"#;
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("FCALL")
                .arg("tensorzero_consume_tickets_v2")
                .arg(2) // numkeys
                .arg("key1")
                .arg("key2")
                .arg(10i64) // key1 requested
                .arg(100i64) // key1 capacity
                .arg(5i64) // key1 refill_amount
                .arg(1_000_000i64) // key1 refill_interval (1 second)
                .arg(20i64) // key2 requested
                .arg(200i64) // key2 capacity
                .arg(10i64) // key2 refill_amount
                .arg(60_000_000i64), // key2 refill_interval (1 minute)
            Ok(expected_response),
        )])
        .assert_all_commands_consumed();

        let requests = vec![
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey::new("key1".to_string()),
                requested: 10,
                capacity: 100,
                refill_amount: 5,
                refill_interval: RateLimitInterval::Second,
            },
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey::new("key2".to_string()),
                requested: 20,
                capacity: 200,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
        ];

        let result = execute_consume_tickets(&mut mock, &requests).await.unwrap();
        assert_eq!(result.len(), 2, "should return two receipts");
        assert!(
            result[0].success && result[1].success,
            "both should succeed"
        );
    }

    #[tokio::test]
    async fn test_consume_tickets_failure_response() {
        let expected_response =
            r#"[{"key":"test_key","success":false,"remaining":5,"consumed":0}]"#;
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("FCALL")
                .arg("tensorzero_consume_tickets_v2")
                .arg(1)
                .arg("test_key")
                .arg(100i64) // requesting more than available
                .arg(100i64)
                .arg(10i64)
                .arg(1_000_000i64),
            Ok(expected_response),
        )])
        .assert_all_commands_consumed();

        let requests = vec![ConsumeTicketsRequest {
            key: ActiveRateLimitKey::new("test_key".to_string()),
            requested: 100,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Second,
        }];

        let result = execute_consume_tickets(&mut mock, &requests).await.unwrap();
        assert!(!result[0].success, "should fail when insufficient tokens");
        assert_eq!(
            result[0].tickets_consumed, 0,
            "should not consume any tickets"
        );
    }

    #[tokio::test]
    async fn test_return_tickets_sends_correct_command() {
        let expected_response = r#"[{"key":"test_key","balance":60}]"#;
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("FCALL")
                .arg("tensorzero_return_tickets_v2")
                .arg(1) // numkeys
                .arg("test_key")
                .arg(10i64) // returned
                .arg(100i64) // capacity
                .arg(5i64) // refill_amount
                .arg(1_000_000i64), // refill_interval
            Ok(expected_response),
        )])
        .assert_all_commands_consumed();

        let requests = vec![ReturnTicketsRequest {
            key: ActiveRateLimitKey::new("test_key".to_string()),
            returned: 10,
            capacity: 100,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Second,
        }];

        let result = execute_return_tickets(&mut mock, &requests).await.unwrap();
        assert_eq!(result.len(), 1, "should return one receipt");
        assert_eq!(result[0].balance, 60, "balance should be 60");
    }

    #[tokio::test]
    async fn test_get_balance_sends_correct_command() {
        let expected_response = r#"{"balance":75}"#;
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("FCALL_RO")
                .arg("tensorzero_get_balance_v2")
                .arg(1) // numkeys
                .arg("test_key")
                .arg(100i64) // capacity
                .arg(10i64) // refill_amount
                .arg(3_600_000_000i64), // refill_interval (1 hour in micros)
            Ok(expected_response),
        )])
        .assert_all_commands_consumed();

        let result = execute_get_balance(&mut mock, "test_key", 100, 10, RateLimitInterval::Hour)
            .await
            .unwrap();
        assert_eq!(result, 75, "balance should be 75");
    }

    #[tokio::test]
    async fn test_get_balance_all_intervals() {
        // Test that all interval types convert to correct microseconds
        let test_cases = vec![
            (RateLimitInterval::Second, 1_000_000i64),
            (RateLimitInterval::Minute, 60_000_000i64),
            (RateLimitInterval::Hour, 3_600_000_000i64),
            (RateLimitInterval::Day, 86_400_000_000i64),
        ];

        for (interval, expected_micros) in test_cases {
            let expected_response = r#"{"balance":100}"#;
            let mut mock = MockRedisConnection::new(vec![MockCmd::new(
                redis::cmd("FCALL_RO")
                    .arg("tensorzero_get_balance_v2")
                    .arg(1)
                    .arg("test_key")
                    .arg(100i64)
                    .arg(10i64)
                    .arg(expected_micros),
                Ok(expected_response),
            )])
            .assert_all_commands_consumed();

            let result = execute_get_balance(&mut mock, "test_key", 100, 10, interval)
                .await
                .unwrap();
            assert_eq!(
                result, 100,
                "balance should be 100 for interval {interval:?}"
            );
        }
    }
}
