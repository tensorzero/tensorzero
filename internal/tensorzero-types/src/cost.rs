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
#[serde(untagged)]
pub enum UninitializedCostRate {
    PerMillion {
        #[serde(with = "rust_decimal::serde::float")]
        cost_per_million: Decimal,
    },
    PerUnit {
        #[serde(with = "rust_decimal::serde::float")]
        cost_per_unit: Decimal,
    },
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
