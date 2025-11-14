use super::{check_column_exists, check_table_exists};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::utils::uuid::get_workflow_evaluation_cutoff_uuid;
use async_trait::async_trait;

/// Adds columns `dynamic_tools`, `dynamic_provider_tools`, `allowed_tools`, `tool_choice`, `parallel_tool_calls`
/// to ChatInference, ChatInferenceDatapoint, and BatchModelInference so that we can store more comprehensive info about
/// what happened with tool configuration.
/// Due to an issue with the migrations in 0038, we update them in place in this PR as well to allow the rest of the migration to proceed smoothly
pub struct Migration0041<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0041";

#[async_trait]
impl Migration for Migration0041<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let tables_to_check = [
            "ChatInferenceDatapoint",
            "ChatInference",
            "BatchModelInference",
            "EpisodeByIdChatView",
            "EpisodeByIdJsonView",
        ];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table_name} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let tables_to_check = [
            "ChatInferenceDatapoint",
            "ChatInference",
            "BatchModelInference",
        ];
        let columns_to_add = [
            "dynamic_tools",
            "dynamic_provider_tools",
            "allowed_tools",
            "tool_choice",
            "parallel_tool_calls",
        ];
        for table in &tables_to_check {
            for column in &columns_to_add {
                if !check_column_exists(self.clickhouse, table, column, MIGRATION_ID).await? {
                    return Ok(true);
                }
            }
        }
        let views_to_check = ["EpisodeByIdChatView", "EpisodeByIdJsonView"];
        for view in &views_to_check {
            let query = format!("SHOW CREATE TABLE {view};");
            let result = self
                .clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
            if result.response.contains("groupArrayState()(id)") {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let cutoff_uuid = get_workflow_evaluation_cutoff_uuid();

        // Note: in the next two commands we aren't including the `view_condition` because
        // the backfilling has already taken place by now so there's no point in keeping it.
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE EpisodeByIdChatView{on_cluster_name} MODIFY QUERY
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    1 as count,
                    groupArrayState(id) as inference_ids,
                    toUInt128(min(id)) as min_inference_id_uint,
                    toUInt128(max(id)) as max_inference_id_uint
                FROM ChatInference
                WHERE toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
                GROUP BY toUInt128(episode_id)",
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"ALTER TABLE EpisodeByIdJsonView{on_cluster_name} MODIFY QUERY
                SELECT
                    toUInt128(episode_id) as episode_id_uint,
                    1 as count,
                    groupArrayState(id) as inference_ids,
                    toUInt128(min(id)) as min_inference_id_uint,
                    toUInt128(max(id)) as max_inference_id_uint
                FROM JsonInference
                WHERE toUInt128(episode_id) < toUInt128(toUUID('{cutoff_uuid}'))
                GROUP BY toUInt128(episode_id)"
            ))
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInferenceDatapoint ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS dynamic_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS dynamic_provider_tools Array(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS allowed_tools Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS tool_choice Nullable(String)"
                    .to_string(),
            )
            .await?;
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE BatchModelInference ADD COLUMN IF NOT EXISTS parallel_tool_calls Nullable(Bool)"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // Note: we don't drop or roll back the materialized view changes here because there is
        // no point in re-adding a buggy change
        r"ALTER TABLE ChatInference DROP COLUMN dynamic_tools;
          ALTER TABLE ChatInference DROP COLUMN dynamic_provider_tools;
          ALTER TABLE ChatInference DROP COLUMN allowed_tools;
          ALTER TABLE ChatInference DROP COLUMN tool_choice;
          ALTER TABLE ChatInference DROP COLUMN parallel_tool_calls;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN dynamic_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN dynamic_provider_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN allowed_tools;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN tool_choice;
          ALTER TABLE ChatInferenceDatapoint DROP COLUMN parallel_tool_calls;
          ALTER TABLE BatchModelInference DROP COLUMN dynamic_tools;
          ALTER TABLE BatchModelInference DROP COLUMN dynamic_provider_tools;
          ALTER TABLE BatchModelInference DROP COLUMN allowed_tools;
          ALTER TABLE BatchModelInference DROP COLUMN tool_choice;
          ALTER TABLE BatchModelInference DROP COLUMN parallel_tool_calls;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
