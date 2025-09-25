use async_trait::async_trait;

use crate::{
    db::{BanditQueries, FeedbackByVariant},
    error::Error,
};

use super::{escape_string_for_clickhouse_literal, ClickHouseConnectionInfo};

#[async_trait]
impl BanditQueries for ClickHouseConnectionInfo {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: String,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        let escaped_function_name = escape_string_for_clickhouse_literal(&function_name);
        let escaped_metric_name = escape_string_for_clickhouse_literal(metric_name);

        let variant_filter = match variant_names {
            None => String::new(),
            Some(names) if names.is_empty() => " AND 1=0".to_string(),
            Some(names) => {
                let escaped_names: Vec<String> = names
                    .iter()
                    .map(|name| format!("'{}'", escape_string_for_clickhouse_literal(name)))
                    .collect();
                format!(" AND variant_name IN ({})", escaped_names.join(", "))
            }
        };

        let query = format!(
            r"
            SELECT
                variant_name,
                avgMerge(feedback_mean) as feedback_mean,
                varSampStableMerge(feedback_variance) as feedback_variance,
                count
            FROM FeedbackByVariantStatistics
            WHERE function_name = '{escaped_function_name}' and metric_name = '{escaped_metric_name}'{variant_filter}
            GROUP BY variant_name
            FORMAT JSONEachRow"
        );
        self.run_query_synchronous_no_params_de(query).await
    }
}
