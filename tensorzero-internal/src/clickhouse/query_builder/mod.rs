#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceOutputSource {
    Inference,
    Demonstration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    NotEqual,
}

impl FloatComparisonOperator {
    pub fn to_clickhouse_operator(&self) -> &str {
        match self {
            FloatComparisonOperator::LessThan => "<",
            FloatComparisonOperator::LessThanOrEqual => "<=",
            FloatComparisonOperator::Equal => "=",
            FloatComparisonOperator::GreaterThan => ">",
            FloatComparisonOperator::GreaterThanOrEqual => ">=",
            FloatComparisonOperator::NotEqual => "!=",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FloatMetricNode {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[derive(Debug, Clone)]
pub struct BooleanMetricNode {
    pub metric_name: String,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub enum FilterOperatorNode {
    Float(FloatMetricNode),
    Boolean(BooleanMetricNode),
}

#[derive(Debug, Clone)]
pub struct FilterNode {
    pub operator: FilterOperatorNode,
}
