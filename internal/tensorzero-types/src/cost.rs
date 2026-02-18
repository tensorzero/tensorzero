use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

pub type UninitializedCostConfig = Vec<UninitializedCostConfigEntry>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UninitializedCostConfigEntry {
    #[serde(flatten)]
    pub pointer: CostPointerConfig,
    #[serde(flatten)]
    pub rate: UninitializedCostRate,
    #[serde(default)]
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UninitializedCostRate {
    #[serde(default, with = "rust_decimal::serde::float_option")]
    pub cost_per_million: Option<Decimal>,
    #[serde(default, with = "rust_decimal::serde::float_option")]
    pub cost_per_unit: Option<Decimal>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CostPointerConfig {
    Unified {
        pointer: String,
    },
    Split {
        pointer_nonstreaming: String,
        pointer_streaming: String,
    },
}
