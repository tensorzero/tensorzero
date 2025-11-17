use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

impl Usage {
    pub fn zero() -> Usage {
        Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
        }
    }

    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }

    /// Sum an iterator of Usage values.
    /// If any usage has None for a field, the sum for that field becomes None.
    pub fn sum_iter_strict<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::zero(), |acc, usage| Usage {
            input_tokens: match (acc.input_tokens, usage.input_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            output_tokens: match (acc.output_tokens, usage.output_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        })
    }

    /// Sum two `Usage` instances.
    /// `None` contaminates on both sides.
    pub fn sum_strict(&mut self, other: &Usage) {
        self.input_tokens = match (self.input_tokens, other.input_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        self.output_tokens = match (self.output_tokens, other.output_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_sum_all_some() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            Usage {
                input_tokens: Some(5),
                output_tokens: Some(15),
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(18));
        assert_eq!(sum.output_tokens, Some(42));
    }

    #[test]
    fn test_usage_sum_with_none() {
        // If any usage has None for a field, the sum for that field becomes None
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
            },
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(7),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None); // None because one usage had None
        assert_eq!(sum.output_tokens, Some(42)); // All had Some, so sum is Some
    }

    #[test]
    fn test_usage_sum_both_fields_none() {
        let usages = vec![
            Usage {
                input_tokens: Some(10),
                output_tokens: None,
            },
            Usage {
                input_tokens: None,
                output_tokens: Some(15),
            },
        ];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, None);
        assert_eq!(sum.output_tokens, None);
    }

    #[test]
    fn test_usage_sum_empty() {
        let usages: Vec<Usage> = vec![];
        let sum = Usage::sum_iter_strict(usages.into_iter());
        // Empty sum should return Some(0) for both fields since we start with Some(0)
        assert_eq!(sum.input_tokens, Some(0));
        assert_eq!(sum.output_tokens, Some(0));
    }

    #[test]
    fn test_usage_sum_single() {
        let usages = vec![Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
        }];

        let sum = Usage::sum_iter_strict(usages.into_iter());
        assert_eq!(sum.input_tokens, Some(10));
        assert_eq!(sum.output_tokens, Some(20));
    }
}
