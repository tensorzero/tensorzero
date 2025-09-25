use async_trait::async_trait;

use crate::{
    db::{BanditQueries, FeedbackByVariant},
    error::Error,
};

use super::ClickHouseConnectionInfo;

#[async_trait]
impl BanditQueries for ClickHouseConnectionInfo {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: String,
        // TODO: if present, filter by variant_names
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        let query = format!(
            r"
            SELECT
                variant_name,
                avgMerge(feedback_mean) as feedback_mean,
                varSampStableMerge(feedback_variance) as feedback_variance,
                count
            FROM FeedbackByVariantStatistics
            WHERE function_name = '{function_name}' and metric_name = '{metric_name}'
            GROUP BY variant_name"
        );
        self.run_query_synchronous_no_params_de(query).await
    }
}
