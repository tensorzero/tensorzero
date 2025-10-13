use sqlx::postgres::types::PgInterval;

use crate::{
    db::{
        ConsumeTicketsReceipt, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsReceipt,
        ReturnTicketsRequest,
    },
    error::{Error, ErrorDetails},
    rate_limiting::ActiveRateLimitKey,
};

use super::PostgresConnectionInfo;

#[derive(Debug)]
pub struct BucketInfo {
    pub key: String,
    pub capacity: i64,
    pub refill_amount: i64,
    pub interval: PgInterval,
}

impl RateLimitQueries for PostgresConnectionInfo {
    async fn consume_tickets(
        &self,
        requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReceipt>, Error> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let pool = self.get_pool().ok_or_else(|| {
            Error::new(ErrorDetails::PostgresQuery {
                message: "Failed to consume tickets for rate limiting: PostgreSQL connection is disabled.".to_string(),
                function_name: None,
            })
        })?;

        let keys: Vec<String> = requests.iter().map(|r| r.key.0.clone()).collect();
        let requested_amounts: Vec<i64> = requests.iter().map(|r| r.requested as i64).collect();
        let capacities: Vec<i64> = requests.iter().map(|r| r.capacity as i64).collect();
        let refill_amounts: Vec<i64> = requests.iter().map(|r| r.refill_amount as i64).collect();
        let refill_intervals: Vec<PgInterval> =
            requests.iter().map(|r| r.refill_interval).collect();

        let responses = sqlx::query_as!(
            ConsumeTicketsResponse,
            "SELECT bucket_key as key, is_successful as success, tickets_remaining, tickets_consumed
             FROM consume_multiple_resource_tickets($1, $2, $3, $4, $5)",
            &keys,
            &requested_amounts,
            &capacities,
            &refill_amounts,
            &refill_intervals
        )
        .fetch_all(pool)
        .await?;

        let mut results = Vec::new();
        for response in responses {
            results.push(response.try_into()?);
        }

        Ok(results)
    }

    async fn return_tickets(
        &self,
        requests: Vec<ReturnTicketsRequest>,
    ) -> Result<Vec<ReturnTicketsReceipt>, Error> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let pool = self.get_pool().ok_or_else(|| {
            Error::new(ErrorDetails::PostgresQuery {
                message: "PostgreSQL connection is disabled".to_string(),
                function_name: None,
            })
        })?;

        // ideally we don't have to clone the keys here
        let keys: Vec<String> = requests.iter().map(|r| r.key.to_string()).collect();
        let amounts: Vec<i64> = requests.iter().map(|r| r.returned as i64).collect();
        let capacities: Vec<i64> = requests.iter().map(|r| r.capacity as i64).collect();
        let refill_amounts: Vec<i64> = requests.iter().map(|r| r.refill_amount as i64).collect();
        let refill_intervals: Vec<PgInterval> =
            requests.iter().map(|r| r.refill_interval).collect();

        let responses = sqlx::query_as!(
            ReturnTicketsResponse,
            "SELECT bucket_key_returned as key_returned, final_balance
             FROM return_multiple_resource_tickets($1, $2, $3, $4, $5)",
            &keys,
            &amounts,
            &capacities,
            &refill_amounts,
            &refill_intervals
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Database query failed: {e}"),
                function_name: Some("return_multiple_resource_tickets".to_string()),
            })
        })?;

        let mut results = Vec::new();
        for response in responses {
            results.push(response.try_into()?);
        }

        Ok(results)
    }

    async fn get_balance(
        &self,
        key: &str,
        capacity: u64,
        refill_amount: u64,
        refill_interval: PgInterval,
    ) -> Result<u64, Error> {
        let pool = self.get_pool().ok_or_else(|| {
            Error::new(ErrorDetails::PostgresQuery {
                message: "PostgreSQL connection is disabled".to_string(),
                function_name: None,
            })
        })?;

        let balance: Option<i64> = sqlx::query_scalar!(
            "SELECT get_resource_bucket_balance($1, $2, $3, $4)",
            key,
            capacity as i64,
            refill_amount as i64,
            refill_interval
        )
        .fetch_one(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Database query failed: {e}"),
                function_name: Some("get_resource_bucket_balance".to_string()),
            })
        })?;

        let balance = balance.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "i64",
                message: "Function returned NULL".to_string(),
            })
        })?;

        if balance < 0 {
            return Err(Error::new(ErrorDetails::PostgresResult {
                result_type: "i64",
                message: "Balance cannot be negative".to_string(),
            }));
        }

        Ok(balance as u64)
    }
}

/// Helper struct for querying consume tickets from sqlx
struct ConsumeTicketsResponse {
    pub key: Option<String>,
    pub success: Option<bool>,
    pub tickets_remaining: Option<i64>,
    pub tickets_consumed: Option<i64>,
}

impl TryFrom<ConsumeTicketsResponse> for ConsumeTicketsReceipt {
    type Error = Error;

    fn try_from(response: ConsumeTicketsResponse) -> Result<Self, Self::Error> {
        let key = response.key.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "consume_multiple_resource_tickets",
                message: "Missing key".to_string(),
            })
        })?;
        let success = response.success.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "consume_multiple_resource_tickets",
                message: "Missing success".to_string(),
            })
        })?;
        let tickets_remaining = response
            .tickets_remaining
            .ok_or_else(|| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "consume_multiple_resource_tickets",
                    message: "Missing tickets_remaining".to_string(),
                })
            })?
            .try_into()
            .map_err(|_| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "consume_multiple_resource_tickets",
                    message: "Invalid tickets_remaining value".to_string(),
                })
            })?;
        let tickets_consumed = response
            .tickets_consumed
            .ok_or_else(|| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "consume_multiple_resource_tickets",
                    message: "Missing tickets_consumed".to_string(),
                })
            })?
            .try_into()
            .map_err(|_| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "consume_multiple_resource_tickets",
                    message: "Invalid tickets_consumed value".to_string(),
                })
            })?;

        Ok(ConsumeTicketsReceipt {
            key: ActiveRateLimitKey::new(key),
            success,
            tickets_remaining,
            tickets_consumed,
        })
    }
}

/// Helper struct for querying return tokens from sqlx
struct ReturnTicketsResponse {
    pub key_returned: Option<String>,
    pub final_balance: Option<i64>,
}

impl TryFrom<ReturnTicketsResponse> for ReturnTicketsReceipt {
    type Error = Error;

    fn try_from(response: ReturnTicketsResponse) -> Result<Self, Self::Error> {
        let key_returned = response.key_returned.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "return_multiple_resource_tickets",
                message: "Missing key_returned".to_string(),
            })
        })?;
        let final_balance = response
            .final_balance
            .ok_or_else(|| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "return_multiple_resource_tickets",
                    message: "Missing final_balance".to_string(),
                })
            })?
            .try_into()
            .map_err(|_| {
                Error::new(ErrorDetails::PostgresResult {
                    result_type: "return_multiple_resource_tickets",
                    message: "Invalid final_balance value".to_string(),
                })
            })?;

        Ok(ReturnTicketsReceipt {
            key: ActiveRateLimitKey(key_returned),
            balance: final_balance,
        })
    }
}
