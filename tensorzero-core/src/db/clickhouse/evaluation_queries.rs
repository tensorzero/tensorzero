//! ClickHouse queries for evaluation statistics.

use std::collections::HashMap;

use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use super::select_queries::parse_count;
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;

#[async_trait]
impl EvaluationQueries for ClickHouseConnectionInfo {
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error> {
        let query = "SELECT toUInt32(uniqExact(value)) as count
                     FROM TagInference
                     WHERE key = 'tensorzero::evaluation_run_id'
                     FORMAT JSONEachRow"
            .to_string();

        let response = self.run_query_synchronous(query, &HashMap::new()).await?;
        parse_count(&response.response)
    }
}
