use std::ops::RangeInclusive;

/// A collection of disjoint intervals. New intervals can be added via `add_interval`,
/// which will automatically be merged with existing overlapping intervals.
/// The final intervals can be obtained via `into_disjoint_intervals`
#[derive(Debug)]
pub struct DisjointIntervals<T: Ord + Clone> {
    // Note that we don't do any sorting or merging until `into_disjoint_intervals` is called
    intervals: Vec<RangeInclusive<T>>,
}

impl<T: Ord + Clone> DisjointIntervals<T> {
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Adds a new interval to the set of disjoint intervals
    pub fn add_interval(&mut self, interval: RangeInclusive<T>) {
        self.intervals.push(interval);
    }

    /// Gets the disjoint intervals in sorted order
    pub fn into_disjoint_intervals(mut self) -> Vec<RangeInclusive<T>> {
        // TODO - come up with a more efficient implementation
        self.intervals.sort_by_key(|i| i.start().clone());

        let mut sorted_disjoint_interval: Vec<RangeInclusive<T>> =
            Vec::with_capacity(self.intervals.len());

        for interval in self.intervals {
            // The most recent disjoint interval overlaps with the new interval, so merge them
            if let Some(last) = sorted_disjoint_interval.last_mut()
                && last.end() >= interval.start()
            {
                *last = RangeInclusive::new(
                    last.start().min(interval.start()).clone(),
                    last.end().max(interval.end()).clone(),
                );
            } else {
                // The new interval starts after the most recent disjoint interval (so it starts after all of the previous intervals)
                // Add it as a new interval
                sorted_disjoint_interval.push(interval.clone());
            }
        }
        sorted_disjoint_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_intervals() {
        let intervals: DisjointIntervals<i32> = DisjointIntervals::new();
        let result = intervals.into_disjoint_intervals();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_interval() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=5);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=5);
    }

    #[test]
    fn test_non_overlapping_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=3);
        intervals.add_interval(5..=7);
        intervals.add_interval(10..=12);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1..=3);
        assert_eq!(result[1], 5..=7);
        assert_eq!(result[2], 10..=12);
    }

    #[test]
    fn test_overlapping_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=5);
        intervals.add_interval(3..=8);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=8);
    }

    #[test]
    fn test_multiple_overlapping_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=5);
        intervals.add_interval(3..=8);
        intervals.add_interval(7..=10);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=10);
    }

    #[test]
    fn test_adjacent_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=3);
        intervals.add_interval(3..=5);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=5);
    }

    #[test]
    fn test_unsorted_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(10..=12);
        intervals.add_interval(1..=3);
        intervals.add_interval(5..=7);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1..=3);
        assert_eq!(result[1], 5..=7);
        assert_eq!(result[2], 10..=12);
    }

    #[test]
    fn test_contained_interval() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=10);
        intervals.add_interval(3..=5);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=10);
    }

    #[test]
    fn test_complex_merge() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=3);
        intervals.add_interval(2..=6);
        intervals.add_interval(8..=10);
        intervals.add_interval(15..=18);
        intervals.add_interval(9..=11);
        intervals.add_interval(5..=7);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1..=7);
        assert_eq!(result[1], 8..=11);
        assert_eq!(result[2], 15..=18);
    }

    #[test]
    fn test_identical_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(5..=10);
        intervals.add_interval(5..=10);
        intervals.add_interval(5..=10);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 5..=10);
    }

    #[test]
    fn test_single_point_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=1);
        intervals.add_interval(3..=3);
        intervals.add_interval(5..=5);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1..=1);
        assert_eq!(result[1], 3..=3);
        assert_eq!(result[2], 5..=5);
    }

    #[test]
    fn test_single_point_overlapping() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=2);
        intervals.add_interval(2..=3);
        intervals.add_interval(3..=4);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=4);
    }

    #[test]
    fn test_u64_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(100u64..=200u64);
        intervals.add_interval(150u64..=250u64);
        intervals.add_interval(300u64..=400u64);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 100u64..=250u64);
        assert_eq!(result[1], 300u64..=400u64);
    }

    #[test]
    fn test_reverse_order_merge() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(10..=15);
        intervals.add_interval(5..=12);
        intervals.add_interval(1..=7);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1..=15);
    }

    #[test]
    fn test_gaps_between_merged_intervals() {
        let mut intervals = DisjointIntervals::new();
        intervals.add_interval(1..=3);
        intervals.add_interval(2..=4);
        intervals.add_interval(10..=12);
        intervals.add_interval(11..=13);
        intervals.add_interval(20..=22);
        let result = intervals.into_disjoint_intervals();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1..=4);
        assert_eq!(result[1], 10..=13);
        assert_eq!(result[2], 20..=22);
    }
}
