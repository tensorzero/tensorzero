use sqlx::QueryBuilder;

use crate::{
    config::{Config, MetricConfigLevel, MetricConfigType},
    db::clickhouse::query_builder::{
        BooleanMetricFilter, DemonstrationFeedbackFilter, FloatMetricFilter, InferenceFilter,
        TagFilter, TimeFilter,
    },
    error::{Error, ErrorDetails},
};

/// Converts an InferenceFilter to Postgres SQL and pushes it to the query builder.
/// The query builder should already have a WHERE clause started (e.g., WHERE 1=1).
/// This function adds ` AND <filter>` to the query.
///
/// Returns an error if a metric name is invalid or its type doesn't match the filter.
pub(super) fn apply_inference_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    filter: Option<&InferenceFilter>,
    config: &Config,
) -> Result<(), Error> {
    let Some(f) = filter else {
        return Ok(());
    };
    query_builder.push(" AND ");
    apply_filter(query_builder, f, config)
}

// ===== Helper function to handle each type of filter =====
// Note: apply_filter outputs just the condition without any connective prefix.
// Callers are responsible for adding AND/OR as needed.

fn apply_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    filter: &InferenceFilter,
    config: &Config,
) -> Result<(), Error> {
    match filter {
        InferenceFilter::FloatMetric(fm) => apply_float_metric_filter(query_builder, fm, config),
        InferenceFilter::BooleanMetric(bm) => {
            apply_boolean_metric_filter(query_builder, bm, config)
        }
        InferenceFilter::DemonstrationFeedback(df) => {
            apply_demonstration_feedback_filter(query_builder, df);
            Ok(())
        }
        InferenceFilter::Tag(tag) => {
            apply_tag_filter(query_builder, tag);
            Ok(())
        }
        InferenceFilter::Time(time) => {
            apply_time_filter(query_builder, time);
            Ok(())
        }
        InferenceFilter::And { children } => apply_and_filter(query_builder, children, config),
        InferenceFilter::Or { children } => apply_or_filter(query_builder, children, config),
        InferenceFilter::Not { child } => apply_not_filter(query_builder, child, config),
    }
}

fn apply_float_metric_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    fm: &FloatMetricFilter,
    config: &Config,
) -> Result<(), Error> {
    let metric_config = config.metrics.get(&fm.metric_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidMetricName {
            metric_name: fm.metric_name.clone(),
        })
    })?;

    // Validate metric type
    if metric_config.r#type != MetricConfigType::Float {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Metric `{}` is not a float metric", fm.metric_name),
        }));
    }

    let join_column = match metric_config.level {
        MetricConfigLevel::Inference => "i.id",
        MetricConfigLevel::Episode => "i.episode_id",
    };
    let operator = fm.comparison_operator.to_postgres_operator();

    // Use EXISTS subquery to filter by metric value
    query_builder
        .push("EXISTS (SELECT 1 FROM tensorzero.float_metric_feedback f WHERE f.target_id = ");
    query_builder.push(join_column);
    query_builder.push(" AND f.metric_name = ");
    query_builder.push_bind(fm.metric_name.clone());
    query_builder.push(" AND f.value ");
    query_builder.push(operator);
    query_builder.push(" ");
    query_builder.push_bind(fm.value);
    query_builder.push(")");

    Ok(())
}

fn apply_boolean_metric_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    bm: &BooleanMetricFilter,
    config: &Config,
) -> Result<(), Error> {
    let metric_config = config.metrics.get(&bm.metric_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidMetricName {
            metric_name: bm.metric_name.clone(),
        })
    })?;

    // Validate metric type
    if metric_config.r#type != MetricConfigType::Boolean {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Metric `{}` is not a boolean metric", bm.metric_name),
        }));
    }

    let join_column = match metric_config.level {
        MetricConfigLevel::Inference => "i.id",
        MetricConfigLevel::Episode => "i.episode_id",
    };

    // Use EXISTS subquery to filter by metric value
    query_builder
        .push("EXISTS (SELECT 1 FROM tensorzero.boolean_metric_feedback f WHERE f.target_id = ");
    query_builder.push(join_column);
    query_builder.push(" AND f.metric_name = ");
    query_builder.push_bind(bm.metric_name.clone());
    query_builder.push(" AND f.value = ");
    query_builder.push_bind(bm.value);
    query_builder.push(")");

    Ok(())
}

fn apply_demonstration_feedback_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    df: &DemonstrationFeedbackFilter,
) {
    if df.has_demonstration {
        query_builder.push(
            "EXISTS (SELECT 1 FROM tensorzero.demonstration_feedback WHERE inference_id = i.id)",
        );
    } else {
        query_builder.push(
            "NOT EXISTS (SELECT 1 FROM tensorzero.demonstration_feedback WHERE inference_id = i.id)",
        );
    }
}

fn apply_tag_filter(query_builder: &mut QueryBuilder<sqlx::Postgres>, tag: &TagFilter) {
    let operator = tag.comparison_operator.to_postgres_operator();

    // For Postgres JSONB, we use the ->> operator to extract text and compare
    // We also check that the key exists.
    // (The semantics for tag != value is "tag must exist and must not be equal to value".
    query_builder.push("(i.tags ? ");
    query_builder.push_bind(tag.key.clone());
    query_builder.push(" AND i.tags->>");
    query_builder.push_bind(tag.key.clone());
    query_builder.push(" ");
    query_builder.push(operator);
    query_builder.push(" ");
    query_builder.push_bind(tag.value.clone());
    query_builder.push(")");
}

fn apply_time_filter(query_builder: &mut QueryBuilder<sqlx::Postgres>, time: &TimeFilter) {
    let operator = time.comparison_operator.to_postgres_operator();

    query_builder.push("i.created_at ");
    query_builder.push(operator);
    query_builder.push(" ");
    query_builder.push_bind(time.time);
}

fn apply_and_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    children: &[InferenceFilter],
    config: &Config,
) -> Result<(), Error> {
    // Empty AND is vacuously true
    if children.is_empty() {
        query_builder.push("TRUE");
        return Ok(());
    }

    // Single child doesn't need parentheses
    if children.len() == 1 {
        return apply_filter(query_builder, &children[0], config);
    }

    // Multiple children: (cond1 AND cond2 AND ...)
    query_builder.push("(");
    for (i, child) in children.iter().enumerate() {
        if i > 0 {
            query_builder.push(" AND ");
        }
        apply_filter(query_builder, child, config)?;
    }
    query_builder.push(")");

    Ok(())
}

fn apply_or_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    children: &[InferenceFilter],
    config: &Config,
) -> Result<(), Error> {
    // Empty OR is false
    if children.is_empty() {
        query_builder.push("FALSE");
        return Ok(());
    }

    // Single child doesn't need parentheses
    if children.len() == 1 {
        return apply_filter(query_builder, &children[0], config);
    }

    // Multiple children: (cond1 OR cond2 OR ...)
    query_builder.push("(");
    for (i, child) in children.iter().enumerate() {
        if i > 0 {
            query_builder.push(" OR ");
        }
        apply_filter(query_builder, child, config)?;
    }
    query_builder.push(")");

    Ok(())
}

fn apply_not_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    child: &InferenceFilter,
    config: &Config,
) -> Result<(), Error> {
    query_builder.push("NOT (");
    apply_filter(query_builder, child, config)?;
    query_builder.push(")");
    Ok(())
}

/// Tracks metric JOINs needed for ORDER BY clauses.
pub(super) struct MetricJoinRegistry {
    /// JOIN clauses to add to the query.
    joins: Vec<String>,
    /// Counter for generating unique join aliases.
    alias_counter: usize,
}

impl MetricJoinRegistry {
    pub(super) fn new() -> Self {
        Self {
            joins: Vec::new(),
            alias_counter: 0,
        }
    }

    /// Registers a metric join and returns the alias for the joined value.
    /// Uses LATERAL join to get the latest feedback value per target with filter pushdown.
    pub(super) fn register_metric_join(
        &mut self,
        metric_name: &str,
        metric_type: MetricConfigType,
        level: MetricConfigLevel,
    ) -> String {
        let alias = format!("metric_{}", self.alias_counter);
        self.alias_counter += 1;

        let table_name = metric_type.postgres_table_name();
        let inference_column = level.inference_column_name();

        // Use LATERAL join to correlate the subquery with the outer query.
        // This pushes down the filter on target_id, allowing PostgreSQL to use indexes
        // efficiently instead of scanning the entire feedback table.
        //
        // TODO(#5691): there's a small risk of SQL injection here because metric_name is
        // user configurable and is not escaped. In practice, metric names are in the config
        // so it's unlikely to be a problem. Still, we may want to revisit and refactor the
        // join clauses to enable parameterized joins.
        let join_clause = format!(
            r"
LEFT JOIN LATERAL (
    SELECT value
    FROM {table_name}
    WHERE target_id = i.{inference_column}
      AND metric_name = '{metric_name}'
    ORDER BY created_at DESC
    LIMIT 1
) AS {alias} ON true"
        );

        self.joins.push(join_clause);
        alias
    }

    /// Returns the JOIN clauses as a single string.
    pub(super) fn get_joins_sql(&self) -> String {
        self.joins.join("")
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use chrono::DateTime;
    use sqlx::QueryBuilder;

    use crate::config::ConfigFileGlob;
    use crate::db::clickhouse::query_builder::test_util::assert_query_equals;
    use crate::db::clickhouse::query_builder::{
        BooleanMetricFilter, FloatMetricFilter, InferenceFilter, TagFilter, TimeFilter,
    };
    use crate::endpoints::stored_inferences::v1::types::{
        DemonstrationFeedbackFilter, FloatComparisonOperator, TagComparisonOperator,
        TimeComparisonOperator,
    };

    use super::*;

    async fn get_e2e_config() -> Config {
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml"))
                .unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests()
    }

    #[test]
    fn test_apply_inference_filter_none() {
        let config = Config::default();
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("SELECT * FROM t WHERE 1=1");
        apply_inference_filter(&mut qb, None, &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(sql.as_str(), "SELECT * FROM t WHERE 1=1");
    }

    #[test]
    fn test_tag_filter_equal() {
        let config = Config::default();
        let filter = InferenceFilter::Tag(TagFilter {
            key: "env".to_string(),
            value: "prod".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND (i.tags ? $1 AND i.tags->>$2 = $3)",
        );
    }

    #[test]
    fn test_tag_filter_not_equal() {
        let config = Config::default();
        let filter = InferenceFilter::Tag(TagFilter {
            key: "env".to_string(),
            value: "test".to_string(),
            comparison_operator: TagComparisonOperator::NotEqual,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND (i.tags ? $1 AND i.tags->>$2 != $3)",
        );
    }

    #[test]
    fn test_time_filter_greater_than() {
        let config = Config::default();
        let filter = InferenceFilter::Time(TimeFilter {
            time: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .into(),
            comparison_operator: TimeComparisonOperator::GreaterThan,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(sql.as_str(), "WHERE 1=1 AND i.created_at > $1");
    }

    #[test]
    fn test_time_filter_less_than_or_equal() {
        let config = Config::default();
        let filter = InferenceFilter::Time(TimeFilter {
            time: DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z")
                .unwrap()
                .into(),
            comparison_operator: TimeComparisonOperator::LessThanOrEqual,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(sql.as_str(), "WHERE 1=1 AND i.created_at <= $1");
    }

    #[test]
    fn test_demonstration_feedback_filter_has_demonstration() {
        let config = Config::default();
        let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
            has_demonstration: true,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND EXISTS (SELECT 1 FROM tensorzero.demonstration_feedback WHERE inference_id = i.id)",
        );
    }

    #[test]
    fn test_demonstration_feedback_filter_no_demonstration() {
        let config = Config::default();
        let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
            has_demonstration: false,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND NOT EXISTS (SELECT 1 FROM tensorzero.demonstration_feedback WHERE inference_id = i.id)",
        );
    }

    #[tokio::test]
    async fn test_float_metric_filter() {
        let config = get_e2e_config().await;
        let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "jaccard_similarity".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND EXISTS (SELECT 1 FROM tensorzero.float_metric_feedback f WHERE f.target_id = i.id AND f.metric_name = $1 AND f.value > $2)",
        );
    }

    #[tokio::test]
    async fn test_boolean_metric_filter() {
        let config = get_e2e_config().await;
        let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "exact_match".to_string(),
            value: true,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND EXISTS (SELECT 1 FROM tensorzero.boolean_metric_feedback f WHERE f.target_id = i.id AND f.metric_name = $1 AND f.value = $2)",
        );
    }

    #[tokio::test]
    async fn test_invalid_metric_name() {
        let config = get_e2e_config().await;
        let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "nonexistent_metric".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        let result = apply_inference_filter(&mut qb, Some(&filter), &config);
        assert!(result.is_err(), "Expected error for invalid metric name");
    }

    #[test]
    fn test_and_filter_empty() {
        let config = Config::default();
        let filter = InferenceFilter::And { children: vec![] };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(sql.as_str(), "WHERE 1=1 AND TRUE");
    }

    #[test]
    fn test_and_filter_single_child() {
        let config = Config::default();
        let filter = InferenceFilter::And {
            children: vec![InferenceFilter::Tag(TagFilter {
                key: "env".to_string(),
                value: "prod".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            })],
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        // Single child should not have extra parentheses
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND (i.tags ? $1 AND i.tags->>$2 = $3)",
        );
    }

    #[test]
    fn test_and_filter_multiple_children() {
        let config = Config::default();
        let filter = InferenceFilter::And {
            children: vec![
                InferenceFilter::Tag(TagFilter {
                    key: "env".to_string(),
                    value: "prod".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilter::Time(TimeFilter {
                    time: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                        .unwrap()
                        .into(),
                    comparison_operator: TimeComparisonOperator::GreaterThan,
                }),
            ],
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND ((i.tags ? $1 AND i.tags->>$2 = $3) AND i.created_at > $4)",
        );
    }

    #[test]
    fn test_or_filter_empty() {
        let config = Config::default();
        let filter = InferenceFilter::Or { children: vec![] };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(sql.as_str(), "WHERE 1=1 AND FALSE");
    }

    #[test]
    fn test_or_filter_single_child() {
        let config = Config::default();
        let filter = InferenceFilter::Or {
            children: vec![InferenceFilter::Tag(TagFilter {
                key: "env".to_string(),
                value: "prod".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            })],
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        // Single child should not have extra parentheses
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND (i.tags ? $1 AND i.tags->>$2 = $3)",
        );
    }

    #[test]
    fn test_or_filter_multiple_children() {
        let config = Config::default();
        let filter = InferenceFilter::Or {
            children: vec![
                InferenceFilter::Tag(TagFilter {
                    key: "env".to_string(),
                    value: "prod".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilter::Tag(TagFilter {
                    key: "env".to_string(),
                    value: "staging".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND ((i.tags ? $1 AND i.tags->>$2 = $3) OR (i.tags ? $4 AND i.tags->>$5 = $6))",
        );
    }

    #[test]
    fn test_not_filter() {
        let config = Config::default();
        let filter = InferenceFilter::Not {
            child: Box::new(InferenceFilter::Tag(TagFilter {
                key: "env".to_string(),
                value: "test".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            })),
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND NOT ((i.tags ? $1 AND i.tags->>$2 = $3))",
        );
    }

    #[test]
    fn test_nested_and_or_filters() {
        let config = Config::default();
        // (env=prod AND version=v1) OR (env=staging)
        let filter = InferenceFilter::Or {
            children: vec![
                InferenceFilter::And {
                    children: vec![
                        InferenceFilter::Tag(TagFilter {
                            key: "env".to_string(),
                            value: "prod".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                        InferenceFilter::Tag(TagFilter {
                            key: "version".to_string(),
                            value: "v1".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                    ],
                },
                InferenceFilter::Tag(TagFilter {
                    key: "env".to_string(),
                    value: "staging".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND (((i.tags ? $1 AND i.tags->>$2 = $3) AND (i.tags ? $4 AND i.tags->>$5 = $6)) OR (i.tags ? $7 AND i.tags->>$8 = $9))",
        );
    }

    #[tokio::test]
    async fn test_episode_level_metric_uses_episode_id() {
        let config = get_e2e_config().await;
        // goal_achieved is an episode-level boolean metric in the e2e config
        let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "goal_achieved".to_string(),
            value: true,
        });
        let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("WHERE 1=1");
        apply_inference_filter(&mut qb, Some(&filter), &config).unwrap();
        // Should use i.episode_id instead of i.id for episode-level metrics
        let sql = qb.sql();
        assert_query_equals(
            sql.as_str(),
            "WHERE 1=1 AND EXISTS (SELECT 1 FROM tensorzero.boolean_metric_feedback f WHERE f.target_id = i.episode_id AND f.metric_name = $1 AND f.value = $2)",
        );
    }

    #[test]
    fn test_metric_join_registry() {
        let mut registry = MetricJoinRegistry::new();

        let alias1 = registry.register_metric_join(
            "test_metric",
            MetricConfigType::Float,
            MetricConfigLevel::Inference,
        );
        assert_eq!(alias1, "metric_0", "First alias should be metric_0");

        let alias2 = registry.register_metric_join(
            "another_metric",
            MetricConfigType::Boolean,
            MetricConfigLevel::Episode,
        );
        assert_eq!(alias2, "metric_1", "Second alias should be metric_1");

        let joins_sql = registry.get_joins_sql();
        assert_query_equals(
            &joins_sql,
            r"
LEFT JOIN LATERAL (
    SELECT value
    FROM tensorzero.float_metric_feedback
    WHERE target_id = i.id
      AND metric_name = 'test_metric'
    ORDER BY created_at DESC
    LIMIT 1
) AS metric_0 ON true
LEFT JOIN LATERAL (
    SELECT value
    FROM tensorzero.boolean_metric_feedback
    WHERE target_id = i.episode_id
      AND metric_name = 'another_metric'
    ORDER BY created_at DESC
    LIMIT 1
) AS metric_1 ON true",
        );
    }
}
