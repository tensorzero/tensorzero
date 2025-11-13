//! Sampling utilities for GEPA optimization
//!
//! This module provides functions for:
//! - Frequency-based weighted sampling from Pareto frontier
//! - Random sampling from datasets

use std::collections::HashMap;

use rand::seq::IndexedRandom;

use tensorzero_core::{
    error::{Error, ErrorDetails},
    stored_inference::RenderedSample,
};

/// Sample a variant name proportional to its frequency
/// Uses weighted random sampling from the `rand` crate
#[cfg_attr(not(test), expect(dead_code))]
pub fn sample_by_frequency(frequencies: &HashMap<String, usize>) -> Result<String, Error> {
    if frequencies.is_empty() {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample from empty frequency map".to_string(),
        }));
    }

    // Check if all frequencies are zero
    let total_frequency: usize = frequencies.values().sum();
    if total_frequency == 0 {
        return Err(Error::new(ErrorDetails::InternalError {
            message: "Cannot sample when all frequencies are zero".to_string(),
        }));
    }

    let items: Vec<_> = frequencies.iter().collect();
    let mut rng = rand::rng();

    items
        .choose_weighted(&mut rng, |(_, &count)| count)
        .map(|(name, _)| (*name).clone())
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Weighted sampling failed: {e}"),
            })
        })
}

/// Sample a random subset of examples from a dataset without replacement
/// Uses random sampling from the `rand` crate
#[cfg_attr(not(test), expect(dead_code))]
pub fn random_sample(examples: &[RenderedSample], n: usize) -> Vec<RenderedSample> {
    let mut rng = rand::rng();
    examples.choose_multiple(&mut rng, n).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tensorzero_core::{
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text,
        },
        stored_inference::StoredOutput,
        tool::DynamicToolParams,
    };
    use uuid::Uuid;

    /// Helper function to create a minimal test RenderedSample with a unique identifier
    fn create_test_sample(id: &str) -> RenderedSample {
        let output = Some(vec![ContentBlockChatOutput::Text(Text {
            text: format!("Test output {id}"),
        })]);

        RenderedSample {
            function_name: format!("test_function_{id}"),
            input: ModelInput {
                system: Some(format!("System message {id}")),
                messages: vec![ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: format!("Test input {id}"),
                    })],
                }],
            },
            stored_input: StoredInput {
                system: Some(System::Text(format!("System message {id}"))),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: format!("Test input {id}"),
                    })],
                }],
            },
            output: output.clone(),
            stored_output: output.map(StoredOutput::Chat),
            episode_id: Some(Uuid::now_v7()),
            inference_id: Some(Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::from([("id".to_string(), id.to_string())]),
        }
    }

    // Tests for sample_by_frequency

    #[test]
    fn test_sample_by_frequency_empty_map() {
        let frequencies = HashMap::new();
        let result = sample_by_frequency(&frequencies);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(
            error.get_details(),
            ErrorDetails::InternalError { message }
            if message.contains("empty frequency map")
        ));
    }

    #[test]
    fn test_sample_by_frequency_all_zeros() {
        let mut frequencies = HashMap::new();
        frequencies.insert("variant_a".to_string(), 0);
        frequencies.insert("variant_b".to_string(), 0);

        let result = sample_by_frequency(&frequencies);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(
            error.get_details(),
            ErrorDetails::InternalError { message }
            if message.contains("all frequencies are zero")
        ));
    }

    #[test]
    fn test_sample_by_frequency_single_variant() {
        let mut frequencies = HashMap::new();
        frequencies.insert("variant_a".to_string(), 100);

        let result = sample_by_frequency(&frequencies);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "variant_a");
    }

    #[test]
    fn test_sample_by_frequency_multiple_variants() {
        let mut frequencies = HashMap::new();
        frequencies.insert("variant_a".to_string(), 50);
        frequencies.insert("variant_b".to_string(), 50);
        frequencies.insert("variant_c".to_string(), 50);

        // Sample multiple times to ensure all variants can be selected
        let mut seen_variants = HashSet::new();
        for _ in 0..30 {
            let result = sample_by_frequency(&frequencies);
            assert!(result.is_ok());
            let variant = result.unwrap();
            assert!(
                frequencies.contains_key(&variant),
                "Sampled variant {variant} not in original frequencies"
            );
            seen_variants.insert(variant);
        }

        // With equal frequencies and 30 samples, we should see all variants
        assert_eq!(
            seen_variants.len(),
            3,
            "Not all variants were sampled in 30 attempts"
        );
    }

    // Tests for random_sample

    #[test]
    fn test_random_sample_empty() {
        let examples: Vec<RenderedSample> = vec![];
        let result = random_sample(&examples, 5);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_random_sample_n_zero() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
        ];

        let result = random_sample(&examples, 0);
        assert_eq!(result.len(), 0); // n=0 returns empty vector
    }

    #[test]
    fn test_random_sample_n_equals_length() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
        ];

        let result = random_sample(&examples, 3);
        assert_eq!(result.len(), 3);

        // Verify all examples are present
        for example in &examples {
            assert!(
                result
                    .iter()
                    .any(|r| r.tags.get("id") == example.tags.get("id")),
                "Original example not found in result"
            );
        }
    }

    #[test]
    fn test_random_sample_n_greater_than_length() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
        ];

        let result = random_sample(&examples, 10);
        assert_eq!(result.len(), 3);

        // Should return all examples
        assert_eq!(result.len(), examples.len());
    }

    #[test]
    fn test_random_sample_subset() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
            create_test_sample("4"),
            create_test_sample("5"),
        ];

        let n = 3;
        let result = random_sample(&examples, n);

        // Check correct number of samples
        assert_eq!(result.len(), n);

        // Verify all sampled items are from the original set
        for sample in &result {
            assert!(
                examples
                    .iter()
                    .any(|e| e.tags.get("id") == sample.tags.get("id")),
                "Sampled item not found in original examples"
            );
        }

        // Verify no duplicates in sampled subset
        let mut seen_ids = HashSet::new();
        for sample in &result {
            let id = sample.tags.get("id").unwrap();
            assert!(seen_ids.insert(id.clone()), "Duplicate sample found: {id}");
        }
    }

    #[test]
    fn test_random_sample_randomness() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
            create_test_sample("4"),
            create_test_sample("5"),
        ];

        let n = 2;
        let mut all_results = vec![];

        // Sample multiple times
        for _ in 0..10 {
            let result = random_sample(&examples, n);
            let ids: Vec<String> = result
                .iter()
                .map(|s| s.tags.get("id").unwrap().clone())
                .collect();
            all_results.push(ids);
        }

        // Check that we got different results (not all the same)
        let first_result = &all_results[0];
        let all_same = all_results.iter().all(|r| r == first_result);

        assert!(
            !all_same,
            "All samples were identical - random_sample is not random"
        );

        // Verify we can sample different combinations
        let unique_combinations: HashSet<_> = all_results.into_iter().collect();
        assert!(
            unique_combinations.len() > 1,
            "Only one unique combination found - random_sample is not random"
        );
    }

    #[test]
    fn test_random_sample_all_elements_eventually_sampled() {
        let examples = vec![
            create_test_sample("1"),
            create_test_sample("2"),
            create_test_sample("3"),
        ];

        let n = 1;
        let mut seen_ids = HashSet::new();

        // Sample one element many times
        for _ in 0..20 {
            let result = random_sample(&examples, n);
            assert_eq!(result.len(), 1);
            seen_ids.insert(result[0].tags.get("id").unwrap().clone());
        }

        // With 20 samples of 1 element from 3, we should see all 3 elements
        assert_eq!(
            seen_ids.len(),
            3,
            "Not all elements were sampled: {seen_ids:?}"
        );
    }
}
