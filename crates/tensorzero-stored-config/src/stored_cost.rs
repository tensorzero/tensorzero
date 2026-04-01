use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tensorzero_types::{
    CostPointerConfig, UnifiedCostPointerConfig, UninitializedCostConfig,
    UninitializedCostConfigEntry, UninitializedCostRate, UninitializedUnifiedCostConfig,
};

// --- Cost config (provider-level `cost` field) ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredCostConfig {
    pub entries: Vec<StoredCostConfigEntry>,
}

impl From<&UninitializedCostConfig> for StoredCostConfig {
    fn from(config: &UninitializedCostConfig) -> Self {
        StoredCostConfig {
            entries: config
                .iter()
                .map(|entry| StoredCostConfigEntry {
                    pointer: entry.pointer.pointer.clone(),
                    pointer_nonstreaming: entry.pointer.pointer_nonstreaming.clone(),
                    pointer_streaming: entry.pointer.pointer_streaming.clone(),
                    cost_per_million: entry.rate.cost_per_million,
                    cost_per_unit: entry.rate.cost_per_unit,
                    required: Some(entry.required),
                })
                .collect(),
        }
    }
}

impl From<StoredCostConfig> for UninitializedCostConfig {
    fn from(stored: StoredCostConfig) -> Self {
        stored
            .entries
            .into_iter()
            .map(|entry| UninitializedCostConfigEntry {
                pointer: CostPointerConfig {
                    pointer: entry.pointer,
                    pointer_nonstreaming: entry.pointer_nonstreaming,
                    pointer_streaming: entry.pointer_streaming,
                },
                rate: UninitializedCostRate {
                    cost_per_million: entry.cost_per_million,
                    cost_per_unit: entry.cost_per_unit,
                },
                required: entry.required.unwrap_or_default(),
            })
            .collect()
    }
}

/// Stored equivalent of `UninitializedCostConfigEntry<CostPointerConfig>`.
/// Flattened fields are represented explicitly (no `#[serde(flatten)]`).
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredCostConfigEntry {
    pub pointer: Option<String>,
    pub pointer_nonstreaming: Option<String>,
    pub pointer_streaming: Option<String>,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
    pub required: Option<bool>,
}

// --- Unified cost config (provider-level `batch_cost` field) ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredUnifiedCostConfig {
    pub entries: Vec<StoredUnifiedCostConfigEntry>,
}

impl From<&UninitializedUnifiedCostConfig> for StoredUnifiedCostConfig {
    fn from(config: &UninitializedUnifiedCostConfig) -> Self {
        StoredUnifiedCostConfig {
            entries: config
                .iter()
                .map(|entry| StoredUnifiedCostConfigEntry {
                    pointer: entry.pointer.pointer.clone(),
                    cost_per_million: entry.rate.cost_per_million,
                    cost_per_unit: entry.rate.cost_per_unit,
                    required: Some(entry.required),
                })
                .collect(),
        }
    }
}

impl From<StoredUnifiedCostConfig> for UninitializedUnifiedCostConfig {
    fn from(stored: StoredUnifiedCostConfig) -> Self {
        stored
            .entries
            .into_iter()
            .map(|entry| UninitializedCostConfigEntry {
                pointer: UnifiedCostPointerConfig {
                    pointer: entry.pointer,
                },
                rate: UninitializedCostRate {
                    cost_per_million: entry.cost_per_million,
                    cost_per_unit: entry.cost_per_unit,
                },
                required: entry.required.unwrap_or_default(),
            })
            .collect()
    }
}

/// Stored equivalent of `UninitializedCostConfigEntry<UnifiedCostPointerConfig>`.
/// Flattened fields are represented explicitly (no `#[serde(flatten)]`).
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredUnifiedCostConfigEntry {
    pub pointer: String,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
    pub required: Option<bool>,
}
