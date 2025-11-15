use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::export_schema;

use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{ClickhouseType, QueryParameter};
use crate::endpoints::stored_inferences::v1::types::{TagFilter, TimeFilter};

/// Filter tree for querying datapoints.
/// This is similar to `InferenceFilter` but without metric filters, as datapoints don't have associated metrics.
#[derive(JsonSchema, ts_rs::TS, Clone, Debug, Deserialize, Serialize)]
#[export_schema]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatapointFilter {
    /// Filter by tag key-value pair
    #[schemars(title = "TagDatapointFilter")]
    Tag(TagFilter),

    /// Filter by datapoint update time
    #[schemars(title = "TimeDatapointFilter")]
    Time(TimeFilter),

    /// Logical AND of multiple filters
    #[schemars(title = "AndDatapointFilter")]
    And { children: Vec<DatapointFilter> },

    /// Logical OR of multiple filters
    #[schemars(title = "OrDatapointFilter")]
    Or { children: Vec<DatapointFilter> },

    /// Logical NOT of a filter
    #[schemars(title = "NotDatapointFilter")]
    Not { child: Box<DatapointFilter> },
}

impl DatapointFilter {
    /// Converts the filter tree to a ClickHouse SQL WHERE clause.
    ///
    /// This method generates SQL fragments that can be used in WHERE clauses for querying datapoints.
    /// It handles
    /// - Tag filtering using ClickHouse map syntax (e.g., `tags[{key}] = {value}`).
    /// - Time filtering against the `updated_at` field.
    /// - Logical operations (AND, OR, NOT).
    ///
    /// Parameters:
    /// - original_table_alias: The alias of the original table (e.g. "i" for ChatInferenceDatapoint). This
    ///   allows us to refer to columns in the original table in the SQL.
    ///
    /// Returns a tuple consists of: A SQL string fragment suitable for inclusion in a WHERE clause (without AND),
    /// and a vector of query parameters to add to parameter bindings.
    pub fn to_clickhouse_sql(&self, original_table_alias: &str) -> (String, Vec<QueryParameter>) {
        let table_prefix = if original_table_alias.is_empty() {
            String::new()
        } else {
            format!("{original_table_alias}.")
        };

        let mut query_params = Vec::new();
        let mut param_idx_counter = 0;
        let sql = self.to_clickhouse_subquery_sql(
            &table_prefix,
            &mut query_params,
            &mut param_idx_counter,
        );
        (sql, query_params)
    }

    fn to_clickhouse_subquery_sql(
        &self,
        table_prefix: &str,
        query_params: &mut Vec<QueryParameter>,
        param_idx_counter: &mut usize,
    ) -> String {
        match self {
            DatapointFilter::Tag(TagFilter {
                key,
                value,
                comparison_operator,
            }) => {
                // Generate parameter placeholders for tag key and value
                let key_placeholder =
                    add_parameter(key, ClickhouseType::String, query_params, param_idx_counter);
                let value_placeholder = add_parameter(
                    value,
                    ClickhouseType::String,
                    query_params,
                    param_idx_counter,
                );
                let comparison_operator = comparison_operator.to_clickhouse_operator();
                format!("{table_prefix}tags[{key_placeholder}] {comparison_operator} {value_placeholder}")
            }
            DatapointFilter::Time(TimeFilter {
                time,
                comparison_operator,
            }) => {
                let time_placeholder = add_parameter(
                    time.to_string(),
                    ClickhouseType::String,
                    query_params,
                    param_idx_counter,
                );
                let comparison_operator = comparison_operator.to_clickhouse_operator();
                format!(
                    "{table_prefix}updated_at {comparison_operator} parseDateTimeBestEffort({time_placeholder})"
                )
            }
            DatapointFilter::And { children } => {
                if children.is_empty() {
                    // Empty AND is true
                    return "TRUE".to_string();
                }
                let clauses: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_subquery_sql(
                            table_prefix,
                            query_params,
                            param_idx_counter,
                        )
                    })
                    .collect();
                format!("({})", clauses.join(" AND "))
            }
            DatapointFilter::Or { children } => {
                if children.is_empty() {
                    // Empty OR is false
                    return "FALSE".to_string();
                }
                let clauses: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_subquery_sql(
                            table_prefix,
                            query_params,
                            param_idx_counter,
                        )
                    })
                    .collect();
                format!("({})", clauses.join(" OR "))
            }
            DatapointFilter::Not { child } => {
                let child_sql =
                    child.to_clickhouse_subquery_sql(table_prefix, query_params, param_idx_counter);
                format!("NOT ({child_sql})")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;

    use crate::db::clickhouse::query_builder::{TagComparisonOperator, TimeComparisonOperator};

    #[test]
    fn test_tag_filter_equal() {
        let filter = DatapointFilter::Tag(TagFilter {
            key: "tag_key".to_string(),
            value: "tag_value".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "tags[{p0:String}] = {p1:String}");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "tag_key");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "tag_value");
    }

    #[test]
    fn test_tag_filter_not_equal() {
        let filter = DatapointFilter::Tag(TagFilter {
            key: "tag_key".to_string(),
            value: "tag_value".to_string(),
            comparison_operator: TagComparisonOperator::NotEqual,
        });

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "tags[{p0:String}] != {p1:String}");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "tag_key");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "tag_value");
    }

    #[test]
    fn test_tag_filter_writes_table_prefix() {
        let filter = DatapointFilter::Tag(TagFilter {
            key: "tag_key".to_string(),
            value: "tag_value".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });

        let (sql, _) = filter.to_clickhouse_sql("original");

        assert_eq!(sql, "original.tags[{p0:String}] = {p1:String}");
    }

    #[test]
    fn test_time_filter_less_than() {
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap(); // 2023-01-01 00:00:00 UTC
        let filter = DatapointFilter::Time(TimeFilter {
            time: test_time,
            comparison_operator: TimeComparisonOperator::LessThan,
        });

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "updated_at < parseDateTimeBestEffort({p0:String})");
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0],
            QueryParameter {
                name: "p0".to_string(),
                value: "2023-01-01 00:00:00 UTC".to_string()
            }
        );
    }

    #[test]
    fn test_time_filter_greater_than() {
        let test_time = DateTime::from_timestamp(1609459200, 0).unwrap(); // 2021-01-01 00:00:00 UTC
        let filter = DatapointFilter::Time(TimeFilter {
            time: test_time,
            comparison_operator: TimeComparisonOperator::GreaterThan,
        });

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "updated_at > parseDateTimeBestEffort({p0:String})");
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0],
            QueryParameter {
                name: "p0".to_string(),
                value: "2021-01-01 00:00:00 UTC".to_string()
            }
        );
    }

    #[test]
    fn test_time_filter_parameter() {
        let test_time = DateTime::from_timestamp(1640995200, 0).unwrap(); // 2022-01-01 00:00:00 UTC
        let filter = DatapointFilter::Time(TimeFilter {
            time: test_time,
            comparison_operator: TimeComparisonOperator::Equal,
        });

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "updated_at = parseDateTimeBestEffort({p0:String})");
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0],
            QueryParameter {
                name: "p0".to_string(),
                value: "2022-01-01 00:00:00 UTC".to_string()
            }
        );
    }

    #[test]
    fn test_time_filter_writes_table_prefix() {
        let test_time = DateTime::from_timestamp(1640995200, 0).unwrap(); // 2022-01-01 00:00:00 UTC
        let filter = DatapointFilter::Time(TimeFilter {
            time: test_time,
            comparison_operator: TimeComparisonOperator::Equal,
        });

        let (sql, params) = filter.to_clickhouse_sql("i");

        assert_eq!(sql, "i.updated_at = parseDateTimeBestEffort({p0:String})");
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0],
            QueryParameter {
                name: "p0".to_string(),
                value: "2022-01-01 00:00:00 UTC".to_string()
            }
        );
    }

    #[test]
    fn test_all_time_comparison_operators() {
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let operators = vec![
            (TimeComparisonOperator::LessThan, "<"),
            (TimeComparisonOperator::LessThanOrEqual, "<="),
            (TimeComparisonOperator::Equal, "="),
            (TimeComparisonOperator::GreaterThan, ">"),
            (TimeComparisonOperator::GreaterThanOrEqual, ">="),
            (TimeComparisonOperator::NotEqual, "!="),
        ];

        for (op, expected_op_str) in operators {
            let filter = DatapointFilter::Time(TimeFilter {
                time: test_time,
                comparison_operator: op,
            });

            let (sql, _) = filter.to_clickhouse_sql("");
            let expected_sql =
                format!("updated_at {expected_op_str} parseDateTimeBestEffort({{p0:String}})");
            assert_eq!(sql, expected_sql, "Failed for operator: {expected_op_str}");
        }
    }

    #[test]
    fn test_and_with_two_tag_filters() {
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Tag(TagFilter {
                    key: "environment".to_string(),
                    value: "production".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "region".to_string(),
                    value: "us-west".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "(tags[{p0:String}] = {p1:String} AND tags[{p2:String}] = {p3:String})"
        );
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "environment");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "production");
        assert_eq!(params[2].name, "p2");
        assert_eq!(params[2].value, "region");
        assert_eq!(params[3].name, "p3");
        assert_eq!(params[3].value, "us-west");
    }

    #[test]
    fn test_or_with_two_tag_filters() {
        let filter = DatapointFilter::Or {
            children: vec![
                DatapointFilter::Tag(TagFilter {
                    key: "priority".to_string(),
                    value: "high".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "priority".to_string(),
                    value: "critical".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "(tags[{p0:String}] = {p1:String} OR tags[{p2:String}] = {p3:String})"
        );
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_not_tag_filter() {
        let filter = DatapointFilter::Not {
            child: Box::new(DatapointFilter::Tag(TagFilter {
                key: "archived".to_string(),
                value: "true".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            })),
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "NOT (tags[{p0:String}] = {p1:String})");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "archived");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "true");
    }

    #[test]
    fn test_not_time_filter() {
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let filter = DatapointFilter::Not {
            child: Box::new(DatapointFilter::Time(TimeFilter {
                time: test_time,
                comparison_operator: TimeComparisonOperator::GreaterThan,
            })),
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "NOT (updated_at > parseDateTimeBestEffort({p0:String}))"
        );
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "2023-01-01 00:00:00 UTC");
    }

    #[test]
    fn test_empty_and_returns_true() {
        let filter = DatapointFilter::And { children: vec![] };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "TRUE");
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_empty_or_returns_false() {
        let filter = DatapointFilter::Or { children: vec![] };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "FALSE");
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_nested_and_or() {
        // AND { OR { tag1, tag2 }, time }
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Or {
                    children: vec![
                        DatapointFilter::Tag(TagFilter {
                            key: "team".to_string(),
                            value: "alpha".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                        DatapointFilter::Tag(TagFilter {
                            key: "team".to_string(),
                            value: "beta".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                    ],
                },
                DatapointFilter::Time(TimeFilter {
                    time: test_time,
                    comparison_operator: TimeComparisonOperator::GreaterThan,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "((tags[{p0:String}] = {p1:String} OR tags[{p2:String}] = {p3:String}) AND updated_at > parseDateTimeBestEffort({p4:String}))"
        );
        assert_eq!(params.len(), 5);
    }

    #[test]
    fn test_nested_or_and() {
        // OR { AND { tag1, time }, tag2 }
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let filter = DatapointFilter::Or {
            children: vec![
                DatapointFilter::And {
                    children: vec![
                        DatapointFilter::Tag(TagFilter {
                            key: "status".to_string(),
                            value: "active".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                        DatapointFilter::Time(TimeFilter {
                            time: test_time,
                            comparison_operator: TimeComparisonOperator::LessThan,
                        }),
                    ],
                },
                DatapointFilter::Tag(TagFilter {
                    key: "priority".to_string(),
                    value: "urgent".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "((tags[{p0:String}] = {p1:String} AND updated_at < parseDateTimeBestEffort({p2:String})) OR tags[{p3:String}] = {p4:String})"
        );
        assert_eq!(params.len(), 5);
    }

    #[test]
    fn test_deeply_nested_filters() {
        // AND { OR { AND { tag1, tag2 }, tag3 }, time }
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Or {
                    children: vec![
                        DatapointFilter::And {
                            children: vec![
                                DatapointFilter::Tag(TagFilter {
                                    key: "env".to_string(),
                                    value: "prod".to_string(),
                                    comparison_operator: TagComparisonOperator::Equal,
                                }),
                                DatapointFilter::Tag(TagFilter {
                                    key: "region".to_string(),
                                    value: "us".to_string(),
                                    comparison_operator: TagComparisonOperator::Equal,
                                }),
                            ],
                        },
                        DatapointFilter::Tag(TagFilter {
                            key: "test".to_string(),
                            value: "true".to_string(),
                            comparison_operator: TagComparisonOperator::Equal,
                        }),
                    ],
                },
                DatapointFilter::Time(TimeFilter {
                    time: test_time,
                    comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "(((tags[{p0:String}] = {p1:String} AND tags[{p2:String}] = {p3:String}) OR tags[{p4:String}] = {p5:String}) AND updated_at >= parseDateTimeBestEffort({p6:String}))");
        assert_eq!(params.len(), 7);
    }

    #[test]
    fn test_not_with_complex_child() {
        // NOT ( AND { tag1, tag2 } )
        let filter = DatapointFilter::Not {
            child: Box::new(DatapointFilter::And {
                children: vec![
                    DatapointFilter::Tag(TagFilter {
                        key: "deleted".to_string(),
                        value: "true".to_string(),
                        comparison_operator: TagComparisonOperator::Equal,
                    }),
                    DatapointFilter::Tag(TagFilter {
                        key: "archived".to_string(),
                        value: "true".to_string(),
                        comparison_operator: TagComparisonOperator::Equal,
                    }),
                ],
            }),
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "NOT ((tags[{p0:String}] = {p1:String} AND tags[{p2:String}] = {p3:String}))"
        );
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_parameter_counter_increments() {
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Tag(TagFilter {
                    key: "a".to_string(),
                    value: "1".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "b".to_string(),
                    value: "2".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "c".to_string(),
                    value: "3".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };

        let (_, params) = filter.to_clickhouse_sql("");

        assert_eq!(params.len(), 6);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[2].name, "p2");
        assert_eq!(params[3].name, "p3");
        assert_eq!(params[4].name, "p4");
        assert_eq!(params[5].name, "p5");
    }

    #[test]
    fn test_parameter_uniqueness() {
        let test_time = DateTime::from_timestamp(1672531200, 0).unwrap();
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Tag(TagFilter {
                    key: "x".to_string(),
                    value: "y".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Time(TimeFilter {
                    time: test_time,
                    comparison_operator: TimeComparisonOperator::GreaterThan,
                }),
            ],
        };

        let (_, params) = filter.to_clickhouse_sql("");

        // Verify all parameter names are unique
        let names: std::collections::HashSet<_> = params.iter().map(|p| &p.name).collect();
        assert_eq!(names.len(), params.len());
    }

    #[test]
    fn test_multiple_tags_with_or() {
        // Real-world use case: multiple possible values for a tag
        let filter = DatapointFilter::Or {
            children: vec![
                DatapointFilter::Tag(TagFilter {
                    key: "status".to_string(),
                    value: "pending".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "status".to_string(),
                    value: "in_progress".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                DatapointFilter::Tag(TagFilter {
                    key: "status".to_string(),
                    value: "review".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "(tags[{p0:String}] = {p1:String} OR tags[{p2:String}] = {p3:String} OR tags[{p4:String}] = {p5:String})"
        );
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_time_range_with_and() {
        // Real-world use case: time range filter
        let start_time = DateTime::from_timestamp(1640995200, 0).unwrap(); // 2022-01-01
        let end_time = DateTime::from_timestamp(1672531200, 0).unwrap(); // 2023-01-01
        let filter = DatapointFilter::And {
            children: vec![
                DatapointFilter::Time(TimeFilter {
                    time: start_time,
                    comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
                }),
                DatapointFilter::Time(TimeFilter {
                    time: end_time,
                    comparison_operator: TimeComparisonOperator::LessThan,
                }),
            ],
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(
            sql,
            "(updated_at >= parseDateTimeBestEffort({p0:String}) AND updated_at < parseDateTimeBestEffort({p1:String}))"
        );
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "2022-01-01 00:00:00 UTC");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "2023-01-01 00:00:00 UTC");
    }

    #[test]
    fn test_exclude_specific_tag_with_not() {
        // Real-world use case: exclude certain tagged datapoints
        let filter = DatapointFilter::Not {
            child: Box::new(DatapointFilter::Tag(TagFilter {
                key: "test".to_string(),
                value: "skip".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            })),
        };

        let (sql, params) = filter.to_clickhouse_sql("");

        assert_eq!(sql, "NOT (tags[{p0:String}] = {p1:String})");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "p0");
        assert_eq!(params[0].value, "test");
        assert_eq!(params[1].name, "p1");
        assert_eq!(params[1].value, "skip");
    }
}
