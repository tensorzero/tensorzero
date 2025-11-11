//! Sampling utilities for GEPA optimization
//!
//! This module provides functions for:
//! - Frequency-based weighted sampling from Pareto frontier
//! - Random sampling from datasets

use std::collections::HashMap;

use rand::Rng;

use tensorzero_core::{
    error::{Error, ErrorDetails},
    stored_inference::RenderedSample,
};

/// Sample a variant name proportional to its frequency
/// Uses cumulative distribution for weighted random sampling
#[expect(dead_code)]
pub fn sample_by_frequency(frequencies: &HashMap<String, usize>) -> Result<String, Error> {
    if frequencies.is_empty() {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample from empty frequency map".to_string(),
        }));
    }

    // Calculate total frequency
    let total_frequency: usize = frequencies.values().sum();

    if total_frequency == 0 {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample when all frequencies are zero".to_string(),
        }));
    }

    // Generate random number in [0, total_frequency)
    let mut rng = rand::rng();
    let mut random_value = rng.random_range(0..total_frequency);

    // Iterate through frequencies using cumulative distribution
    for (variant_name, &frequency) in frequencies {
        if random_value < frequency {
            return Ok(variant_name.clone());
        }
        random_value -= frequency;
    }

    // Fallback (should never happen due to total_frequency check)
    // Return the first variant as a safety measure
    Ok(frequencies
        .keys()
        .next()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "Frequency map unexpectedly empty".to_string(),
            })
        })?
        .clone())
}

/// Sample a random subset of examples from a dataset without replacement
/// Uses Fisher-Yates shuffle algorithm for efficient sampling
#[expect(dead_code)]
pub fn random_sample(examples: &[RenderedSample], n: usize) -> Vec<RenderedSample> {
    // If n >= examples.len(), return all examples
    if n >= examples.len() {
        return examples.to_vec();
    }

    // Create a vector of indices and shuffle only the first n elements
    let mut rng = rand::rng();
    let mut indices: Vec<usize> = (0..examples.len()).collect();

    // Partial Fisher-Yates shuffle: only shuffle the first n elements
    for i in 0..n {
        let j = rng.random_range(i..indices.len());
        indices.swap(i, j);
    }

    // Collect the first n sampled examples
    indices[0..n].iter().map(|&i| examples[i].clone()).collect()
}
