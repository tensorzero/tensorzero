use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

pub type UninitializedCostConfig = Vec<UninitializedCostConfigEntry>;

pub type UninitializedUnifiedCostConfig =
    Vec<UninitializedCostConfigEntry<UnifiedCostPointerConfig>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UninitializedCostConfigEntry<P = CostPointerConfig> {
    #[serde(flatten)]
    pub pointer: P,
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
pub struct CostPointerConfig {
    #[serde(default)]
    pub pointer: Option<String>,
    #[serde(default)]
    pub pointer_nonstreaming: Option<String>,
    #[serde(default)]
    pub pointer_streaming: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UnifiedCostPointerConfig {
    pub pointer: String,
}
