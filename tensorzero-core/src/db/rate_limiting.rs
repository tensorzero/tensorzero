use async_trait::async_trait;

#[cfg(test)]
use mockall::automock;

use crate::error::Error;
use crate::rate_limiting::ActiveRateLimitKey;
use sqlx::postgres::types::PgInterval;

#[async_trait]
#[cfg_attr(test, automock)]
pub trait RateLimitQueries: Send + Sync {
    /// This function will fail if any of the requests individually fail.
    /// It is an atomic operation so no tickets will be consumed if any request fails.
    async fn consume_tickets(
        &self,
        requests: &[ConsumeTicketsRequest],
    ) -> Result<Vec<ConsumeTicketsReceipt>, Error>;

    async fn return_tickets(
        &self,
        requests: Vec<ReturnTicketsRequest>,
    ) -> Result<Vec<ReturnTicketsReceipt>, Error>;

    async fn get_balance(
        &self,
        key: &str,
        capacity: u64,
        refill_amount: u64,
        refill_interval: PgInterval,
    ) -> Result<u64, Error>;
}

#[derive(Debug)]
pub struct ConsumeTicketsRequest {
    pub key: ActiveRateLimitKey,
    pub requested: u64,
    pub capacity: u64,
    pub refill_amount: u64,
    pub refill_interval: PgInterval,
}

#[derive(Debug)]
pub struct ConsumeTicketsReceipt {
    pub key: ActiveRateLimitKey,
    pub success: bool,
    pub tickets_remaining: u64,
    pub tickets_consumed: u64,
}

pub struct ReturnTicketsRequest {
    pub key: ActiveRateLimitKey,
    pub returned: u64,
    pub capacity: u64,
    pub refill_amount: u64,
    pub refill_interval: PgInterval,
}

pub struct ReturnTicketsReceipt {
    pub key: ActiveRateLimitKey,
    pub balance: u64,
}
