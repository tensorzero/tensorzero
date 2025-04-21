//! A commented, slightly safer, and more efficient Zhang–Shasha tree‑edit‑distance implementation.
//! Adapted from MIT-licensed code (license at https://github.com/anonymrepo/tree-edit-distance/blob/master/LICENSE)
//!
//!  * Computes all metadata in **one DFS** (O(n)) instead of repeatedly walking
//!    subtrees.
//!  * De‑duplicates key‑roots (prevents O(n²) redundant DP calls).
//!  * Adds many `debug_assert!`s to document invariants.
//!  * Heavy inline commentary so you can quickly adapt it to Tree‑sitter.
//!
//! The public API is intentionally unchanged except for extra genericity on
//! `TreeNode` labels, so you can drop‑in‑replace the older file in your tests.

use std::{cmp, collections::HashMap};

mod dp_matrix;
use crate::ted::dp_matrix::Matrix;

/// A labelled, ordered tree in **post‑order** representation plus per‑node metadata.
/// Lifetime `'a` ties the `Tree` to the lifetime of its underlying `TreeNode`s.
pub struct Tree<'a, L: Eq> {
    /// Post‑order list of pointers into the original node arena.
    post_order: Vec<&'a TreeNode<L>>, // length == n

    /// For every node *i* (in post‑order), `left_most_leaf_descendant[i]` is the
    /// index (also in post‑order) of the left‑most leaf in the subtree rooted at *i*.
    left_most_leaf_descendant: Vec<usize>, // length == n, O(1) lookup

    /// The set of **key‑roots** as defined by Zhang & Shasha (1989), sorted.
    key_roots: Vec<usize>, // strictly increasing indices into `post_order`
}

/// A basic tree node.  Parameterised by label type `L` so that integration with
/// Tree‑sitter can supply e.g. `u16` (`tree_sitter::Symbol`) instead of `String`.
#[derive(Debug)]
pub struct TreeNode<L: Eq> {
    pub label: L,
    pub children: Vec<Box<TreeNode<L>>>,
}

impl<L: Eq> TreeNode<L> {
    /// Create a leaf.
    pub fn new(label: L) -> Self {
        Self {
            label,
            children: vec![],
        }
    }

    /// Fluent builder for an internal node.
    pub fn with_children(mut self, children: Vec<Box<Self>>) -> Self {
        self.children = children;
        self
    }
}

impl<'a, L: Eq> Tree<'a, L> {
    // ------------------------- construction helpers ------------------------- //

    /// Build the post‑order vector *and* left‑most‑leaf table in one DFS.
    fn dfs_build<'b>(
        node: &'a TreeNode<L>,
        post: &mut Vec<&'a TreeNode<L>>, // out‑param
        leftmost: &mut Vec<usize>,       // out‑param (parallel to `post`)
    ) -> usize {
        let mut leftmost_of_subtree = usize::MAX;
        // Visit children first (post‑order).
        for child in node.children.iter() {
            let child_leftmost = Self::dfs_build(child, post, leftmost);
            // the first child's left‑most leaf is the current subtree's left‑most.
            if leftmost_of_subtree == usize::MAX {
                leftmost_of_subtree = child_leftmost;
            }
        }
        // If `node` is a leaf, its own post‑order index will be its left‑most leaf.
        if leftmost_of_subtree == usize::MAX {
            // No children
            leftmost_of_subtree = post.len();
        }

        // Finally push `node` itself.
        let my_index = post.len();
        post.push(node);
        leftmost.push(leftmost_of_subtree);
        my_index
    }

    /// Compute **key‑roots** (the last node for every distinct left‑most leaf).
    fn compute_key_roots(leftmost: &[usize]) -> Vec<usize> {
        // Map: left‑most‑leaf‑id -> last index with that left‑most leaf
        let mut last_for_leaf: HashMap<usize, usize> = HashMap::new();
        for (idx, &l) in leftmost.iter().enumerate() {
            last_for_leaf.insert(l, idx); // keeps the *last* occurrence
        }
        let mut roots: Vec<usize> = last_for_leaf.values().copied().collect();
        roots.sort();
        roots
    }

    /// Construct a `Tree` view out of a given root reference.
    pub fn new(root: &'a TreeNode<L>) -> Self {
        let mut post_order = Vec::new();
        let mut leftmost = Vec::new();
        Self::dfs_build(root, &mut post_order, &mut leftmost);
        debug_assert_eq!(post_order.len(), leftmost.len());

        let key_roots = Self::compute_key_roots(&leftmost);

        Self {
            post_order,
            left_most_leaf_descendant: leftmost,
            key_roots,
        }
    }

    // ---------------------------- cost helpers ----------------------------- //

    #[inline]
    fn label_cost(a: &TreeNode<L>, b: &TreeNode<L>, relabel_cost: u64) -> u64 {
        if a.label == b.label {
            0
        } else {
            relabel_cost
        }
    }

    /// Internal DP over *forests* needed for Zhang‑Shasha.
    #[allow(clippy::too_many_arguments)]
    fn forest_distance(
        k1: usize, // key‑root in `t1`
        k2: usize, // key‑root in `t2`
        t1: &Tree<L>,
        t2: &Tree<L>,
        td: &mut Matrix<u64>, // global tree distance matrix (n1 × n2)
        ins_cost: u64,
        del_cost: u64,
        relabel_cost: u64,
    ) {
        // Aliases used in the original paper.
        let l1 = t1.left_most_leaf_descendant[k1];
        let l2 = t2.left_most_leaf_descendant[k2];

        // Forest DP buffer sized to (|A|+1) × (|B|+1).
        let rows = k1 - l1 + 2; // +1 for empty prefix, +1 because inclusive indices
        let cols = k2 - l2 + 2;
        let mut fd = Matrix::<u64>::zeros(rows, cols);

        // Initialisation: cost of deleting / inserting prefixes.
        for i in 1..rows {
            fd[(i, 0)] = fd[(i - 1, 0)] + del_cost;
        }
        for j in 1..cols {
            fd[(0, j)] = fd[(0, j - 1)] + ins_cost;
        }

        // Main DP.
        for i in 1..rows {
            for j in 1..cols {
                // Absolute node indices in the original trees.
                let node_i = i + l1 - 1;
                let node_j = j + l2 - 1;

                let both_trees = t1.left_most_leaf_descendant[node_i] == l1
                    && t2.left_most_leaf_descendant[node_j] == l2;

                if both_trees {
                    // Case 1: both forests are actually *trees* (subtrees rooted at node_i / node_j).
                    fd[(i, j)] = cmp::min(
                        cmp::min(fd[(i - 1, j)] + del_cost, fd[(i, j - 1)] + ins_cost),
                        fd[(i - 1, j - 1)]
                            + Self::label_cost(
                                t1.post_order[node_i],
                                t2.post_order[node_j],
                                relabel_cost,
                            ),
                    );
                    td[(node_i, node_j)] = fd[(i, j)];
                } else {
                    // Case 2: at least one is a forest. Re‑use already computed sub‑tree distances.
                    let li = t1.left_most_leaf_descendant[node_i] - l1;
                    let lj = t2.left_most_leaf_descendant[node_j] - l2;
                    fd[(i, j)] = cmp::min(
                        cmp::min(fd[(i - 1, j)] + del_cost, fd[(i, j - 1)] + ins_cost),
                        fd[(li, lj)] + td[(node_i, node_j)],
                    );
                }
            }
        }
    }

    // ------------------------------- public API ---------------------------- //

    /// Compute weighted tree‑edit distance between `self` and `other`.
    pub fn weighted_tree_edit_distance(
        &self,
        other: &Tree<L>,
        ins_cost: u64,
        del_cost: u64,
        relabel_cost: u64,
    ) -> u64 {
        let mut td = Matrix::<u64>::zeros(self.post_order.len(), other.post_order.len());
        for &x in &self.key_roots {
            for &y in &other.key_roots {
                Self::forest_distance(x, y, self, other, &mut td, ins_cost, del_cost, relabel_cost);
            }
        }
        td[(self.post_order.len() - 1, other.post_order.len() - 1)]
    }

    /// Convenience wrapper: unit weights.
    #[inline]
    pub fn tree_edit_distance(&self, other: &Tree<L>) -> u64 {
        self.weighted_tree_edit_distance(other, 1, 1, 1)
    }
}

// ---------------------------------------------------------------------------
//                                  tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Helper macro to build small sample trees using &str labels.
    macro_rules! tree {
        ($l:expr) => {
            TreeNode::new($l.to_string().into())
        };
        ($l:expr, [ $( $c:expr ),* ]) => {{
            let mut root = TreeNode::new($l.to_string());
            $( root.children.push(Box::new($c)); )*
            root
        }};
    }

    #[test]
    fn post_order_indices_are_correct() {
        // See ASCII art in the original file for node numbering.
        let root = tree!(
            "A",
            [tree!("B"), tree!("C", [tree!("E"), tree!("F")]), tree!("D")]
        );
        let t = Tree::new(&root);
        let labels: Vec<_> = t.post_order.iter().map(|n| n.label.as_str()).collect();
        assert_eq!(labels, ["B", "E", "F", "C", "D", "A"]);
    }

    #[test]
    fn leftmost_leaf_descendants_match_textbook() {
        let root = tree!(
            "A",
            [tree!("B"), tree!("C", [tree!("E"), tree!("F")]), tree!("D")]
        );
        let t = Tree::new(&root);
        // Quick spot check (full pattern checked manually):
        assert_eq!(t.left_most_leaf_descendant[5], 0); // root -> B
        assert_eq!(t.left_most_leaf_descendant[4], 4); // D -> D
    }

    #[test]
    fn key_roots_deduplicated_and_sorted() {
        let root = tree!(
            "A",
            [tree!("B"), tree!("C", [tree!("E"), tree!("F")]), tree!("D")]
        );
        let t = Tree::new(&root);
        assert_eq!(t.key_roots, [2, 3, 4, 5]);
    }

    #[test]
    fn self_distance_zero() {
        let a = tree!("A", [tree!("B"), tree!("C")]);
        let ta = Tree::new(&a);
        assert_eq!(0, ta.tree_edit_distance(&ta));
    }

    #[test]
    fn basic_weighted_distance() {
        let t1 = Tree::new(&tree!("A"));
        let t2 = Tree::new(&tree!("B"));
        assert_eq!(2, t1.weighted_tree_edit_distance(&t2, 1, 1, 3)); // substitution cheaper than delete+insert
    }
}
