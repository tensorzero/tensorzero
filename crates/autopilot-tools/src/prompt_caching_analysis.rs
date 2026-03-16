//! Prompt caching optimization analysis.
//!
//! Given a batch of inferences for the same function/variant, this module
//! detects opportunities to reduce cost and latency through prompt caching.
//!
//! The analysis is provider-agnostic at the core (prefix stability, temporal
//! analysis, shuffle detection) with provider-specific output for breakpoint
//! recommendations and cost estimates.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[cfg(feature = "ts-bindings")]
use ts_rs::TS;

// ── Input types ────────────────────────────────────────────────────────────

/// A single inference sample for prompt caching analysis.
#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct InferenceSample {
    /// Canonical JSON of the system prompt (serialized `System` field).
    /// Empty string if no system prompt.
    pub system_text: String,
    /// Canonical JSON of each message, in order.
    pub message_texts: Vec<String>,
    /// Canonical JSON of tool definitions. Empty string if no tools.
    pub tools_text: String,
    /// Total input tokens reported by the provider.
    pub input_tokens: Option<u32>,
    /// Timestamp as seconds since epoch (for temporal analysis).
    pub timestamp_secs: f64,
    /// Model name (for min-threshold lookup).
    pub model_name: String,
}

/// Pricing configuration for prompt caching cost estimation.
///
/// All prices are in dollars per million tokens. Derive these from the model
/// provider's `cost` config in `tensorzero.toml`:
///
/// ```toml
/// cost = [
///   { pointer = "/usage/input_tokens", cost_per_million = 3.00 },
///   { pointer = "/usage/cache_read_input_tokens", cost_per_million = 0.30 },
///   { pointer = "/usage/cache_creation_input_tokens", cost_per_million = 3.75 },
/// ]
/// ```
///
/// The base input price maps to the `input_tokens` / `prompt_tokens` pointer,
/// cache read to `cache_read_input_tokens`, and cache write to
/// `cache_creation_input_tokens`.
#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CachingPricingConfig {
    /// Base input token price (dollars per million tokens).
    pub input_price_per_million: f64,
    /// Cache read token price (dollars per million tokens).
    /// For Anthropic this is typically 0.1x the input price.
    pub cache_read_price_per_million: f64,
    /// Cache write token price (dollars per million tokens).
    /// For Anthropic 5-min TTL this is typically 1.25x the input price.
    pub cache_write_price_per_million: f64,
}

impl Default for CachingPricingConfig {
    fn default() -> Self {
        // Anthropic Sonnet-class defaults
        Self {
            input_price_per_million: 3.0,
            cache_read_price_per_million: 0.3,
            cache_write_price_per_million: 3.75,
        }
    }
}

// ── Output types ───────────────────────────────────────────────────────────

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct PromptCachingReport {
    pub sample_size: usize,
    pub time_range_secs: Option<(f64, f64)>,
    pub model_name: String,

    pub segment_stability: SegmentStability,
    pub prefix_analysis: PrefixAnalysis,
    pub temporal: TemporalAnalysis,
    pub cost_estimate: CostEstimate,
    pub shuffle_detection: ShuffleDetection,
    pub breakpoint_recommendations: Vec<BreakpointRecommendation>,
    pub invalidation_warnings: Vec<String>,

    /// Composite score 0–100.
    pub opportunity_score: f64,
    pub summary: String,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SegmentStability {
    /// Fraction of consecutive pairs with identical tools.
    pub tools: f64,
    /// Fraction of consecutive pairs with identical system prompt.
    pub system: f64,
    /// Per-position stability for messages (position → fraction identical).
    pub messages_by_position: Vec<f64>,
    /// Number of unique tool definitions across all samples.
    pub tools_unique_count: usize,
    /// Number of unique system prompts across all samples.
    pub system_unique_count: usize,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct PrefixAnalysis {
    /// Mean fraction of the request that is a common prefix with the next request.
    pub mean_prefix_ratio: f64,
    pub p25_prefix_ratio: f64,
    pub p75_prefix_ratio: f64,
    pub min_prefix_ratio: f64,
    /// Estimated mean cacheable tokens (from prefix).
    pub mean_cacheable_tokens: f64,
    /// Estimated mean uncached tokens.
    pub mean_uncached_tokens: f64,
    /// Whether the cacheable prefix meets the model's minimum token threshold.
    pub meets_min_threshold: bool,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TemporalAnalysis {
    /// Median inter-request interval in seconds.
    pub p50_interval_secs: f64,
    /// 90th percentile inter-request interval.
    pub p90_interval_secs: f64,
    /// Fraction of intervals under 5 minutes.
    pub pct_within_5min: f64,
    /// Fraction of intervals under 1 hour.
    pub pct_within_1hr: f64,
    /// Estimated cache hit rate under 5-min TTL.
    pub estimated_hit_rate_5min: f64,
    /// Estimated cache hit rate under 1-hour TTL.
    pub estimated_hit_rate_1hr: f64,
    /// Fraction of intervals under 1 second (concurrent requests).
    pub burst_ratio: f64,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CostEstimate {
    /// Total estimated cost of all samples without caching (dollars).
    pub total_cost_no_caching: f64,
    /// Total estimated cost with caching enabled (dollars).
    pub total_cost_with_caching: f64,
    /// Projected savings percentage.
    pub savings_pct: f64,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ShuffleDetection {
    /// Mean shuffle score (Jaccard - prefix_ratio for messages).
    pub mean_shuffle_score: f64,
    pub shuffle_detected: bool,
    pub recommendation: Option<String>,
}

#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct BreakpointRecommendation {
    /// Human-readable position, e.g. "after system", "after messages[2]".
    pub position: String,
    /// JSON pointer for `extra_body`, e.g. "/system/0/cache_control".
    pub extra_body_pointer: String,
    /// Value to set at the pointer.
    pub extra_body_value: serde_json::Value,
    /// Estimated cacheable tokens up to this breakpoint.
    pub estimated_cacheable_tokens: f64,
    /// Estimated incremental savings percentage from adding this breakpoint.
    pub incremental_savings_pct: f64,
}

// ── Constants ──────────────────────────────────────────────────────────────

/// Minimum token thresholds for prompt caching by model.
fn min_cache_tokens(model: &str) -> u32 {
    // Match on known prefixes/substrings
    if model.contains("opus-4-6") || model.contains("opus-4-5") {
        4096
    } else if model.contains("sonnet-4-6") {
        2048
    } else if model.contains("sonnet-4-5")
        || model.contains("sonnet-4")
        || model.contains("sonnet-3-7")
        || model.contains("opus-4-1")
        || model.contains("opus-4")
    {
        1024
    } else if model.contains("haiku-4-5") {
        4096
    } else if model.contains("haiku-3-5") || model.contains("haiku-3") {
        2048
    } else {
        // Conservative default
        2048
    }
}

// ── Core algorithm ─────────────────────────────────────────────────────────

/// Concatenate all segments into a single string following the cache hierarchy:
/// tools → system → messages
fn build_cache_string(sample: &InferenceSample) -> String {
    let estimated_len = sample.tools_text.len()
        + sample.system_text.len()
        + sample.message_texts.iter().map(|m| m.len()).sum::<usize>()
        + 4 // separators
        + sample.message_texts.len(); // per-message '|'
    let mut s = String::with_capacity(estimated_len);
    s.push_str(&sample.tools_text);
    s.push_str("||");
    s.push_str(&sample.system_text);
    s.push_str("||");
    for msg in &sample.message_texts {
        s.push_str(msg);
        s.push('|');
    }
    s
}

/// Longest common prefix length between two strings (byte-level).
fn lcp_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// Compute a percentile from a sorted slice. Uses linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(
        sorted.windows(2).all(|w| w[0] <= w[1] || w[0].is_nan()),
        "percentile input must be sorted"
    );
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ── Phase 2: Segment stability ─────────────────────────────────────────────

fn compute_segment_stability(samples: &[InferenceSample]) -> SegmentStability {
    let n = samples.len();
    if n < 2 {
        return SegmentStability {
            tools: 1.0,
            system: 1.0,
            messages_by_position: vec![],
            tools_unique_count: if n == 1 { 1 } else { 0 },
            system_unique_count: if n == 1 { 1 } else { 0 },
        };
    }

    let pairs = n - 1;

    // Tools stability
    let tools_matches = samples
        .windows(2)
        .filter(|w| w[0].tools_text == w[1].tools_text)
        .count();

    // System stability
    let system_matches = samples
        .windows(2)
        .filter(|w| w[0].system_text == w[1].system_text)
        .count();

    // Per-position message stability
    let max_msgs = samples
        .iter()
        .map(|s| s.message_texts.len())
        .max()
        .unwrap_or(0);
    let mut msg_stability = Vec::with_capacity(max_msgs);
    for pos in 0..max_msgs {
        let mut matching = 0usize;
        let mut comparable = 0usize;
        for w in samples.windows(2) {
            if let (Some(a), Some(b)) = (w[0].message_texts.get(pos), w[1].message_texts.get(pos)) {
                comparable += 1;
                if a == b {
                    matching += 1;
                }
            }
        }
        msg_stability.push(if comparable > 0 {
            matching as f64 / comparable as f64
        } else {
            0.0
        });
    }

    // Unique counts
    let tools_unique: HashSet<&str> = samples.iter().map(|s| s.tools_text.as_str()).collect();
    let system_unique: HashSet<&str> = samples.iter().map(|s| s.system_text.as_str()).collect();

    SegmentStability {
        tools: tools_matches as f64 / pairs as f64,
        system: system_matches as f64 / pairs as f64,
        messages_by_position: msg_stability,
        tools_unique_count: tools_unique.len(),
        system_unique_count: system_unique.len(),
    }
}

// ── Phase 3 & 4: Prefix analysis ──────────────────────────────────────────

fn compute_prefix_analysis(
    samples: &[InferenceSample],
    cache_strings: &[String],
    model: &str,
) -> PrefixAnalysis {
    if samples.len() < 2 {
        return PrefixAnalysis {
            mean_prefix_ratio: 0.0,
            p25_prefix_ratio: 0.0,
            p75_prefix_ratio: 0.0,
            min_prefix_ratio: 0.0,
            mean_cacheable_tokens: 0.0,
            mean_uncached_tokens: 0.0,
            meets_min_threshold: false,
        };
    }

    let mut prefix_ratios = Vec::with_capacity(samples.len() - 1);
    let mut cacheable_tokens_list = Vec::with_capacity(samples.len() - 1);
    let mut uncached_tokens_list = Vec::with_capacity(samples.len() - 1);

    for i in 0..samples.len() - 1 {
        let lcp = lcp_len(&cache_strings[i], &cache_strings[i + 1]);
        let total = cache_strings[i].len().max(1);
        let ratio = lcp as f64 / total as f64;
        prefix_ratios.push(ratio);

        // Token estimation
        if let Some(input_tokens) = samples[i].input_tokens {
            let input_tokens = input_tokens as f64;
            let cacheable = ratio * input_tokens;
            cacheable_tokens_list.push(cacheable);
            uncached_tokens_list.push(input_tokens - cacheable);
        }
    }

    let mean_prefix = prefix_ratios.iter().sum::<f64>() / prefix_ratios.len() as f64;

    prefix_ratios.sort_by(f64::total_cmp);
    let sorted_ratios = &prefix_ratios;
    let mean_cacheable = if cacheable_tokens_list.is_empty() {
        0.0
    } else {
        cacheable_tokens_list.iter().sum::<f64>() / cacheable_tokens_list.len() as f64
    };
    let mean_uncached = if uncached_tokens_list.is_empty() {
        0.0
    } else {
        uncached_tokens_list.iter().sum::<f64>() / uncached_tokens_list.len() as f64
    };

    let min_threshold = min_cache_tokens(model) as f64;

    PrefixAnalysis {
        mean_prefix_ratio: mean_prefix,
        p25_prefix_ratio: percentile(sorted_ratios, 0.25),
        p75_prefix_ratio: percentile(sorted_ratios, 0.75),
        min_prefix_ratio: sorted_ratios.first().copied().unwrap_or(0.0),
        mean_cacheable_tokens: mean_cacheable,
        mean_uncached_tokens: mean_uncached,
        meets_min_threshold: mean_cacheable >= min_threshold,
    }
}

// ── Phase 5: Temporal analysis ─────────────────────────────────────────────

fn compute_temporal_analysis(
    samples: &[InferenceSample],
    mean_prefix_ratio: f64,
) -> TemporalAnalysis {
    if samples.len() < 2 {
        return TemporalAnalysis {
            p50_interval_secs: 0.0,
            p90_interval_secs: 0.0,
            pct_within_5min: 0.0,
            pct_within_1hr: 0.0,
            estimated_hit_rate_5min: 0.0,
            estimated_hit_rate_1hr: 0.0,
            burst_ratio: 0.0,
        };
    }

    let mut intervals: Vec<f64> = samples
        .windows(2)
        .map(|w| (w[1].timestamp_secs - w[0].timestamp_secs).abs())
        .collect();

    intervals.sort_by(f64::total_cmp);

    let total = intervals.len() as f64;
    let within_5min = intervals.iter().filter(|&&i| i < 300.0).count() as f64;
    let within_1hr = intervals.iter().filter(|&&i| i < 3600.0).count() as f64;
    let bursts = intervals.iter().filter(|&&i| i < 1.0).count() as f64;

    let pct_5min = within_5min / total;
    let pct_1hr = within_1hr / total;

    TemporalAnalysis {
        p50_interval_secs: percentile(&intervals, 0.5),
        p90_interval_secs: percentile(&intervals, 0.9),
        pct_within_5min: pct_5min,
        pct_within_1hr: pct_1hr,
        estimated_hit_rate_5min: pct_5min * mean_prefix_ratio,
        estimated_hit_rate_1hr: pct_1hr * mean_prefix_ratio,
        burst_ratio: bursts / total,
    }
}

// ── Phase 6: Cost estimation ───────────────────────────────────────────────

fn compute_cost_estimate(
    samples: &[InferenceSample],
    prefix_analysis: &PrefixAnalysis,
    temporal: &TemporalAnalysis,
    pricing: &CachingPricingConfig,
) -> CostEstimate {
    let base_per_token = pricing.input_price_per_million / 1_000_000.0;
    let cache_read_per_token = pricing.cache_read_price_per_million / 1_000_000.0;
    let cache_write_per_token = pricing.cache_write_price_per_million / 1_000_000.0;

    let mut total_no_cache = 0.0;
    let mut total_with_cache = 0.0;

    for sample in samples {
        let input_tokens = sample.input_tokens.unwrap_or(0) as f64;
        let cacheable = prefix_analysis.mean_prefix_ratio * input_tokens;
        let uncached = input_tokens - cacheable;

        // No caching: all tokens at base price
        let cost_no_cache = input_tokens * base_per_token;

        // Cache hit: cacheable tokens at read price, rest at base
        let cost_hit = cacheable * cache_read_per_token + uncached * base_per_token;

        // Cache miss: cacheable tokens at write price, rest at base
        let cost_miss = cacheable * cache_write_per_token + uncached * base_per_token;

        // Use the best available hit rate estimate
        let hit_rate = temporal
            .estimated_hit_rate_5min
            .max(temporal.estimated_hit_rate_1hr);

        total_no_cache += cost_no_cache;
        total_with_cache += hit_rate * cost_hit + (1.0 - hit_rate) * cost_miss;
    }

    let savings_pct = if total_no_cache > 0.0 {
        1.0 - total_with_cache / total_no_cache
    } else {
        0.0
    };

    CostEstimate {
        total_cost_no_caching: total_no_cache,
        total_cost_with_caching: total_with_cache,
        savings_pct,
    }
}

// ── Phase 7: Shuffle detection ─────────────────────────────────────────────

fn compute_shuffle_detection(samples: &[InferenceSample]) -> ShuffleDetection {
    if samples.len() < 2 {
        return ShuffleDetection {
            mean_shuffle_score: 0.0,
            shuffle_detected: false,
            recommendation: None,
        };
    }

    let mut shuffle_scores = Vec::with_capacity(samples.len() - 1);

    for w in samples.windows(2) {
        // Build sets of (role_index, content_hash) for each sample's messages
        let set_a: HashSet<u64> = w[0].message_texts.iter().map(|m| hash_string(m)).collect();
        let set_b: HashSet<u64> = w[1].message_texts.iter().map(|m| hash_string(m)).collect();

        let intersection = set_a.intersection(&set_b).count() as f64;
        let union = set_a.union(&set_b).count().max(1) as f64;
        let jaccard = intersection / union;

        // Sequence match: count matching message positions
        let min_len = w[0].message_texts.len().min(w[1].message_texts.len());
        let max_len = w[0]
            .message_texts
            .len()
            .max(w[1].message_texts.len())
            .max(1);
        let matching_positions = w[0]
            .message_texts
            .iter()
            .zip(w[1].message_texts.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let seq_match = if min_len == 0 {
            0.0
        } else {
            matching_positions as f64 / max_len as f64
        };

        // Shuffle = content overlap minus sequential match
        let shuffle = (jaccard - seq_match).max(0.0);
        shuffle_scores.push(shuffle);
    }

    let mean_shuffle = shuffle_scores.iter().sum::<f64>() / shuffle_scores.len() as f64;
    let detected = mean_shuffle > 0.05;

    ShuffleDetection {
        mean_shuffle_score: mean_shuffle,
        shuffle_detected: detected,
        recommendation: if mean_shuffle > 0.3 {
            Some(
                "Significant content reordering detected across requests. \
                 Reorder messages to place static content (few-shot examples, context) \
                 before dynamic content (user queries) to unlock prefix caching."
                    .to_string(),
            )
        } else if detected {
            Some(
                "Mild content reordering detected. Consider stabilizing message \
                 ordering to improve cache hit rates."
                    .to_string(),
            )
        } else {
            None
        },
    }
}

/// Hash a string for set-based comparison.
/// Note: `DefaultHasher` is not guaranteed stable across Rust versions, but that's
/// fine here — hashes are only compared within a single analysis run, never persisted.
fn hash_string(s: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// ── Phase 8: Breakpoint recommendations ────────────────────────────────────

fn compute_breakpoint_recommendations(
    samples: &[InferenceSample],
    stability: &SegmentStability,
    model: &str,
) -> Vec<BreakpointRecommendation> {
    let min_threshold = min_cache_tokens(model) as f64;
    let mut recommendations = Vec::new();

    // Estimate chars-per-token from sample data
    let chars_per_token = estimate_chars_per_token(samples);

    // Cumulative token estimate as we add segments
    let mut cumulative_chars: f64 = 0.0;

    // 1. Tools segment
    if stability.tools > 0.9 {
        let avg_tools_chars =
            samples.iter().map(|s| s.tools_text.len()).sum::<usize>() as f64 / samples.len() as f64;
        cumulative_chars += avg_tools_chars;
        let est_tokens = cumulative_chars / chars_per_token;

        if est_tokens >= min_threshold && avg_tools_chars > 0.0 {
            recommendations.push(BreakpointRecommendation {
                position: "after tools".to_string(),
                extra_body_pointer: "/tools/-1/cache_control".to_string(),
                extra_body_value: serde_json::json!({"type": "ephemeral"}),
                estimated_cacheable_tokens: est_tokens,
                incremental_savings_pct: 0.0, // filled below
            });
        }
    }

    // 2. System segment
    if stability.system > 0.9 {
        let avg_system_chars = samples.iter().map(|s| s.system_text.len()).sum::<usize>() as f64
            / samples.len() as f64;
        cumulative_chars += avg_system_chars;
        let est_tokens = cumulative_chars / chars_per_token;

        if est_tokens >= min_threshold && avg_system_chars > 0.0 {
            recommendations.push(BreakpointRecommendation {
                position: "after system".to_string(),
                extra_body_pointer: "/system/0/cache_control".to_string(),
                extra_body_value: serde_json::json!({"type": "ephemeral"}),
                estimated_cacheable_tokens: est_tokens,
                incremental_savings_pct: 0.0,
            });
        }
    }

    // 3. Stable message positions
    for (pos, &stab) in stability.messages_by_position.iter().enumerate() {
        if stab < 0.9 {
            break; // Once messages become unstable, stop
        }
        let avg_msg_chars = samples
            .iter()
            .filter_map(|s| s.message_texts.get(pos))
            .map(|m| m.len())
            .sum::<usize>() as f64
            / samples.len() as f64;
        cumulative_chars += avg_msg_chars;
        let est_tokens = cumulative_chars / chars_per_token;

        if est_tokens >= min_threshold {
            recommendations.push(BreakpointRecommendation {
                position: format!("after messages[{pos}]"),
                extra_body_pointer: format!("/messages/{pos}/content/0/cache_control"),
                extra_body_value: serde_json::json!({"type": "ephemeral"}),
                estimated_cacheable_tokens: est_tokens,
                incremental_savings_pct: 0.0,
            });
        }
    }

    // Limit to 4 breakpoints (API max), keep the most valuable ones
    // (last ones cache the most content)
    if recommendations.len() > 4 {
        // Keep the last 4 (most cumulative tokens)
        let start = recommendations.len() - 4;
        recommendations.drain(..start);
    }

    // Estimate incremental savings for each breakpoint
    let avg_input_tokens = samples
        .iter()
        .filter_map(|s| s.input_tokens)
        .map(|t| t as f64)
        .sum::<f64>()
        / samples
            .iter()
            .filter(|s| s.input_tokens.is_some())
            .count()
            .max(1) as f64;

    for rec in &mut recommendations {
        let cacheable_ratio = rec.estimated_cacheable_tokens / avg_input_tokens.max(1.0);
        // Savings = cacheable portion * (1.0 - 0.1) = 90% of cacheable tokens at read price
        rec.incremental_savings_pct = cacheable_ratio * 0.9;
    }

    recommendations
}

fn estimate_chars_per_token(samples: &[InferenceSample]) -> f64 {
    let mut total_chars = 0usize;
    let mut total_tokens = 0u64;

    for s in samples {
        if let Some(tokens) = s.input_tokens {
            let chars = s.tools_text.len()
                + s.system_text.len()
                + s.message_texts.iter().map(|m| m.len()).sum::<usize>();
            total_chars += chars;
            total_tokens += tokens as u64;
        }
    }

    if total_tokens > 0 {
        total_chars as f64 / total_tokens as f64
    } else {
        4.0 // fallback: ~4 chars per token for English
    }
}

// ── Phase: Invalidation warnings ───────────────────────────────────────────

fn compute_invalidation_warnings(
    samples: &[InferenceSample],
    stability: &SegmentStability,
) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for dynamic tools (tools changing between requests)
    if stability.tools_unique_count > 1 {
        warnings.push(format!(
            "Tool definitions change across requests ({} unique sets). \
             This invalidates the entire cache hierarchy. \
             Consider stabilizing tool definitions.",
            stability.tools_unique_count
        ));
    }

    // Check for multiple models
    let models: HashSet<&str> = samples.iter().map(|s| s.model_name.as_str()).collect();
    if models.len() > 1 {
        warnings.push(format!(
            "Multiple models detected ({models:?}). Each model has its own cache. \
             Analyze per-model for accurate results."
        ));
    }

    // Check for message count variation (could indicate different conversation depths)
    let min_msgs = samples
        .iter()
        .map(|s| s.message_texts.len())
        .min()
        .unwrap_or(0);
    let max_msgs = samples
        .iter()
        .map(|s| s.message_texts.len())
        .max()
        .unwrap_or(0);
    if max_msgs > 20 {
        warnings.push(format!(
            "Some requests have {max_msgs} message blocks, exceeding the 20-block \
             lookback limit. Add intermediate cache breakpoints to ensure cache hits."
        ));
    }
    if max_msgs > min_msgs * 2 && min_msgs > 0 {
        warnings.push(format!(
            "Message count varies widely ({min_msgs}–{max_msgs}). \
             This may indicate mixed single-turn and multi-turn usage, \
             which reduces cache effectiveness."
        ));
    }

    warnings
}

// ── Opportunity score ──────────────────────────────────────────────────────

fn compute_opportunity_score(
    prefix: &PrefixAnalysis,
    temporal: &TemporalAnalysis,
    cost: &CostEstimate,
) -> f64 {
    let cacheable_pct = prefix.mean_prefix_ratio;
    let hit_rate = temporal
        .estimated_hit_rate_5min
        .max(temporal.estimated_hit_rate_1hr);
    let savings = cost.savings_pct.max(0.0);
    let threshold_gate = if prefix.meets_min_threshold { 1.0 } else { 0.0 };

    // Weighted product scaled to 0-100
    let raw = cacheable_pct * hit_rate * savings * threshold_gate;
    // Scale: the theoretical max of raw is 1.0 * 1.0 * 1.0 * 1.0 = 1.0
    // But realistic best case is ~0.9 * 0.9 * 0.8 = 0.648
    // Scale so that 0.5 maps to ~80
    let score = (raw.sqrt() * 100.0).min(100.0);
    (score * 10.0).round() / 10.0
}

fn generate_summary(score: f64, prefix: &PrefixAnalysis, cost: &CostEstimate) -> String {
    if score >= 80.0 {
        format!(
            "High opportunity: {:.0}% of input tokens are cacheable on average, \
             potential savings of {:.0}%.",
            prefix.mean_prefix_ratio * 100.0,
            cost.savings_pct * 100.0
        )
    } else if score >= 50.0 {
        format!(
            "Moderate opportunity: {:.0}% of input tokens overlap between consecutive requests. \
             Review breakpoint recommendations.",
            prefix.mean_prefix_ratio * 100.0,
        )
    } else if score >= 20.0 {
        format!(
            "Low opportunity: {:.0}% prefix overlap detected, but savings are modest ({:.0}%).",
            prefix.mean_prefix_ratio * 100.0,
            cost.savings_pct * 100.0
        )
    } else {
        "No significant prompt caching opportunity detected.".to_string()
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Analyze a batch of inference samples for prompt caching optimization opportunities.
///
/// Samples should be from the same `(function_name, variant_name)` pair,
/// sorted by timestamp ascending.
///
/// `pricing` provides the base input token price and cache multipliers,
/// derived from the model provider's `cost` configuration.
pub fn analyze_prompt_caching(
    samples: &[InferenceSample],
    pricing: &CachingPricingConfig,
) -> PromptCachingReport {
    let model = samples
        .first()
        .map(|s| s.model_name.clone())
        .unwrap_or_default();

    let time_range = match (samples.first(), samples.last()) {
        (Some(first), Some(last)) if samples.len() >= 2 => {
            Some((first.timestamp_secs, last.timestamp_secs))
        }
        _ => None,
    };

    // Phase 1: Build cache strings
    let cache_strings: Vec<String> = samples.iter().map(build_cache_string).collect();

    // Phase 2: Segment stability
    let segment_stability = compute_segment_stability(samples);

    // Phase 3 & 4: Prefix analysis
    let prefix_analysis = compute_prefix_analysis(samples, &cache_strings, &model);

    // Phase 5: Temporal analysis
    let temporal = compute_temporal_analysis(samples, prefix_analysis.mean_prefix_ratio);

    // Phase 6: Cost estimation
    let cost_estimate = compute_cost_estimate(samples, &prefix_analysis, &temporal, pricing);

    // Phase 7: Shuffle detection
    let shuffle_detection = compute_shuffle_detection(samples);

    // Phase 8: Breakpoint recommendations
    let breakpoint_recommendations =
        compute_breakpoint_recommendations(samples, &segment_stability, &model);

    // Invalidation warnings
    let invalidation_warnings = compute_invalidation_warnings(samples, &segment_stability);

    // Score & summary
    let opportunity_score = compute_opportunity_score(&prefix_analysis, &temporal, &cost_estimate);
    let summary = generate_summary(opportunity_score, &prefix_analysis, &cost_estimate);

    PromptCachingReport {
        sample_size: samples.len(),
        time_range_secs: time_range,
        model_name: model,
        segment_stability,
        prefix_analysis,
        temporal,
        cost_estimate,
        shuffle_detection,
        breakpoint_recommendations,
        invalidation_warnings,
        opportunity_score,
        summary,
    }
}

// ── Conversion helpers ─────────────────────────────────────────────────────

/// Convert a `StoredInference` into an `InferenceSample` for analysis.
///
/// This is the bridge between TensorZero's inference storage and the
/// analysis algorithm. The caller provides the model_name and timestamp
/// since those come from different fields depending on the inference type.
pub fn stored_input_to_sample(
    system: &Option<tensorzero_core::inference::types::System>,
    messages: &[serde_json::Value],
    tools_json: &str,
    input_tokens: Option<u32>,
    timestamp_secs: f64,
    model_name: &str,
) -> InferenceSample {
    let system_text = match system {
        Some(tensorzero_core::inference::types::System::Text(s)) => s.clone(),
        Some(tensorzero_core::inference::types::System::Template(args)) => {
            serde_json::to_string(args).unwrap_or_default()
        }
        None => String::new(),
    };

    let message_texts: Vec<String> = messages
        .iter()
        .map(|m| serde_json::to_string(m).unwrap_or_default())
        .collect();

    InferenceSample {
        system_text,
        message_texts,
        tools_text: tools_json.to_string(),
        input_tokens,
        timestamp_secs,
        model_name: model_name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(
        system: &str,
        messages: Vec<&str>,
        tools: &str,
        tokens: u32,
        ts: f64,
    ) -> InferenceSample {
        InferenceSample {
            system_text: system.to_string(),
            message_texts: messages.into_iter().map(String::from).collect(),
            tools_text: tools.to_string(),
            input_tokens: Some(tokens),
            timestamp_secs: ts,
            model_name: "claude-sonnet-4-5".to_string(),
        }
    }

    #[test]
    fn test_lcp_len() {
        assert_eq!(lcp_len("abcdef", "abcxyz"), 3);
        assert_eq!(lcp_len("hello", "hello"), 5);
        assert_eq!(lcp_len("abc", "xyz"), 0);
        assert_eq!(lcp_len("", "abc"), 0);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.5) - 3.0).abs() < 0.001);
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.001);
        assert!((percentile(&data, 1.0) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_stability_identical_requests() {
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| {
                make_sample(
                    "You are a helpful assistant.",
                    vec!["msg1", "msg2"],
                    r#"[{"name":"tool1"}]"#,
                    1000,
                    i as f64 * 60.0,
                )
            })
            .collect();

        let stability = compute_segment_stability(&samples);
        assert!((stability.tools - 1.0).abs() < 0.001);
        assert!((stability.system - 1.0).abs() < 0.001);
        assert_eq!(stability.messages_by_position.len(), 2);
        assert!((stability.messages_by_position[0] - 1.0).abs() < 0.001);
        assert!((stability.messages_by_position[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_stability_varying_messages() {
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| {
                make_sample(
                    "You are a helpful assistant.",
                    vec![&format!("user query {i}")],
                    r#"[{"name":"tool1"}]"#,
                    1000,
                    i as f64 * 60.0,
                )
            })
            .collect();

        let stability = compute_segment_stability(&samples);
        assert!(
            (stability.tools - 1.0).abs() < 0.001,
            "tools should be stable"
        );
        assert!(
            (stability.system - 1.0).abs() < 0.001,
            "system should be stable"
        );
        assert!(
            stability.messages_by_position[0] < 0.5,
            "messages should be unstable"
        );
    }

    #[test]
    fn test_prefix_analysis_high_overlap() {
        // System + tools identical, only last message varies
        let long_system = "x".repeat(5000);
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| {
                make_sample(
                    &long_system,
                    vec![&format!("q{i}")],
                    r#"[{"name":"tool1"}]"#,
                    2000,
                    i as f64 * 60.0,
                )
            })
            .collect();

        let cache_strings: Vec<String> = samples.iter().map(build_cache_string).collect();
        let prefix = compute_prefix_analysis(&samples, &cache_strings, "claude-sonnet-4-5");

        assert!(
            prefix.mean_prefix_ratio > 0.9,
            "should have >90% prefix overlap, got {}",
            prefix.mean_prefix_ratio
        );
    }

    #[test]
    fn test_prefix_analysis_no_overlap() {
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| {
                make_sample(
                    &format!("system {i}"),
                    vec![&format!("q{i}")],
                    &format!(r#"[{{"name":"tool{i}"}}]"#),
                    1000,
                    i as f64 * 60.0,
                )
            })
            .collect();

        let cache_strings: Vec<String> = samples.iter().map(build_cache_string).collect();
        let prefix = compute_prefix_analysis(&samples, &cache_strings, "claude-sonnet-4-5");

        assert!(
            prefix.mean_prefix_ratio < 0.5,
            "should have low overlap, got {}",
            prefix.mean_prefix_ratio
        );
    }

    #[test]
    fn test_temporal_analysis() {
        // Requests every 2 minutes — should be within 5-min TTL
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| make_sample("sys", vec!["msg"], "tools", 1000, i as f64 * 120.0))
            .collect();

        let temporal = compute_temporal_analysis(&samples, 0.9);
        assert!(
            (temporal.p50_interval_secs - 120.0).abs() < 1.0,
            "median should be ~120s"
        );
        assert!(
            (temporal.pct_within_5min - 1.0).abs() < 0.001,
            "all within 5min"
        );
    }

    #[test]
    fn test_temporal_analysis_sparse() {
        // Requests every 10 minutes — outside 5-min TTL
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| make_sample("sys", vec!["msg"], "tools", 1000, i as f64 * 600.0))
            .collect();

        let temporal = compute_temporal_analysis(&samples, 0.9);
        assert!(
            temporal.pct_within_5min < 0.01,
            "none within 5min, got {}",
            temporal.pct_within_5min
        );
        assert!(
            (temporal.pct_within_1hr - 1.0).abs() < 0.001,
            "all within 1hr"
        );
    }

    #[test]
    fn test_shuffle_detection_no_shuffle() {
        let samples: Vec<InferenceSample> = (0..10)
            .map(|i| {
                make_sample(
                    "sys",
                    vec!["example1", "example2", &format!("q{i}")],
                    "tools",
                    1000,
                    i as f64 * 60.0,
                )
            })
            .collect();

        let shuffle = compute_shuffle_detection(&samples);
        assert!(!shuffle.shuffle_detected, "should not detect shuffle");
    }

    #[test]
    fn test_shuffle_detection_with_shuffle() {
        let samples = vec![
            make_sample("sys", vec!["A", "B", "C"], "tools", 1000, 0.0),
            make_sample("sys", vec!["C", "A", "B"], "tools", 1000, 60.0),
            make_sample("sys", vec!["B", "C", "A"], "tools", 1000, 120.0),
            make_sample("sys", vec!["A", "C", "B"], "tools", 1000, 180.0),
        ];

        let shuffle = compute_shuffle_detection(&samples);
        assert!(shuffle.shuffle_detected, "should detect shuffle");
        assert!(
            shuffle.mean_shuffle_score > 0.05,
            "shuffle score should be significant"
        );
    }

    #[test]
    fn test_full_analysis_high_opportunity() {
        let long_system = "x".repeat(10000);
        let tools = r#"[{"name":"search","description":"Search the web","input_schema":{}}]"#;
        let samples: Vec<InferenceSample> = (0..50)
            .map(|i| {
                InferenceSample {
                    system_text: long_system.clone(),
                    message_texts: vec![format!("user query {i}")],
                    tools_text: tools.to_string(),
                    input_tokens: Some(5000),
                    timestamp_secs: i as f64 * 30.0, // every 30 seconds
                    model_name: "claude-sonnet-4-5".to_string(),
                }
            })
            .collect();

        let report = analyze_prompt_caching(&samples, &CachingPricingConfig::default());

        assert!(
            report.opportunity_score > 50.0,
            "should be a good opportunity, got {}",
            report.opportunity_score
        );
        assert!(
            report.prefix_analysis.mean_prefix_ratio > 0.9,
            "high prefix overlap expected"
        );
        assert!(
            report.cost_estimate.savings_pct > 0.3,
            "should have significant savings"
        );
        assert!(
            !report.breakpoint_recommendations.is_empty(),
            "should recommend breakpoints"
        );
    }

    #[test]
    fn test_full_analysis_no_opportunity() {
        let samples: Vec<InferenceSample> = (0..20)
            .map(|i| {
                InferenceSample {
                    system_text: format!("unique system {i}"),
                    message_texts: vec![format!("unique msg {i}")],
                    tools_text: format!(r#"[{{"name":"tool{i}"}}]"#),
                    input_tokens: Some(500),
                    timestamp_secs: i as f64 * 7200.0, // every 2 hours
                    model_name: "claude-sonnet-4-5".to_string(),
                }
            })
            .collect();

        let report = analyze_prompt_caching(&samples, &CachingPricingConfig::default());

        assert!(
            report.opportunity_score < 20.0,
            "should be no opportunity, got {}",
            report.opportunity_score
        );
    }

    #[test]
    fn test_invalidation_warnings_dynamic_tools() {
        let samples = vec![
            make_sample("sys", vec!["msg"], r#"[{"name":"tool1"}]"#, 1000, 0.0),
            make_sample("sys", vec!["msg"], r#"[{"name":"tool2"}]"#, 1000, 60.0),
        ];

        let stability = compute_segment_stability(&samples);
        let warnings = compute_invalidation_warnings(&samples, &stability);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("Tool definitions change")),
            "should warn about dynamic tools"
        );
    }

    #[test]
    fn test_single_sample() {
        let samples = vec![make_sample("sys", vec!["msg"], "tools", 1000, 0.0)];

        let report = analyze_prompt_caching(&samples, &CachingPricingConfig::default());
        assert_eq!(report.sample_size, 1);
        // Should not panic, just return empty/zero analysis
        assert!(report.time_range_secs.is_none());
    }

    #[test]
    fn test_empty_samples() {
        let report = analyze_prompt_caching(&[], &CachingPricingConfig::default());
        assert_eq!(report.sample_size, 0);
    }
}
