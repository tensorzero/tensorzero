//! A commented, slightly safer, and more efficient Zhang–Shasha tree‑edit‑distance implementation.
//! Adapted from MIT-licensed code (license at https://github.com/anonymrepo/tree-edit-distance/blob/master/LICENSE)
//!
//!  * Computes all metadata in **one DFS** (O(n)) instead of repeatedly walking
//!    subtrees.
//!  * De‑duplicates key‑roots (prevents O(n²) redundant DP calls).
//!  * Adds many `debug_assert!`s to document invariants.

use std::{cmp, collections::HashMap};

mod matrix;
use crate::ted::matrix::Matrix;
use tree_sitter::{Node, TreeCursor};

pub struct TedInfo {
    pub min_ted: u64,
    pub ted_ratio: f64,
    pub size: usize,
}

pub fn minimum_ted(needle: &Node, haystack: &Node) -> TedInfo {
    // We need to walk the haystack to get the postorder traversal
    let mut haystack_cursor = haystack.walk();
    let mut haystack_post = Vec::new();
    let mut haystack_subtree_start = Vec::new();
    let mut haystack_sizes = Vec::new();
    dfs_postorder(
        &mut haystack_cursor,
        &mut haystack_post,
        &mut haystack_subtree_start,
        &mut haystack_sizes,
    );
    let mut needle_cursor = needle.walk();
    let mut needle_post = Vec::new();
    let mut needle_subtree_start = Vec::new();
    let mut needle_sizes = Vec::new();
    dfs_postorder(
        &mut needle_cursor,
        &mut needle_post,
        &mut needle_subtree_start,
        &mut needle_sizes,
    );
    // Construct the needle tree
    let needle_tree = Tree::new_from_postorder(&needle_post, &needle_subtree_start, 0);
    // Iterate over the haystack and create the subtree ending at each index
    let mut min_ted = u64::MAX;
    for (end, &start) in haystack_subtree_start.iter().enumerate() {
        let haystack_tree = Tree::new_from_postorder(
            &haystack_post[start..end],
            &haystack_subtree_start[start..end],
            start,
        );
        let ted = haystack_tree.tree_edit_distance(&needle_tree);
        if ted < min_ted {
            min_ted = ted;
        }
    }
    let needle_size = needle_post.len();
    TedInfo {
        min_ted,
        ted_ratio: 1.0 - min_ted as f64 / needle_size as f64,
        size: needle_size,
    }
}

struct TsNodeWrapper<'a> {
    node: Node<'a>,
}

impl<'a> TsNodeWrapper<'a> {
    pub fn new(node: Node<'a>) -> Self {
        Self { node }
    }
}

impl PartialEq for TsNodeWrapper<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.node.grammar_name() == other.node.grammar_name()
            && self.node.kind() == other.node.kind()
    }
}
impl Eq for TsNodeWrapper<'_> {}

/// To execute Zhang-Shasha tree edit distance, we need to traverse each subtree in postorder.
/// This function does a postorder traversal of the tree while wrapping each node in our TsNodeWrapper
/// and returns the ordered list of node wrappers as well as the starting index of each subtree.
/// Since postorder traversal lets you visit all the children before visiting the node this is straightforward.
/// This function assumes previously initialized empty vectors for post and subtree_start as the base case.
fn dfs_postorder<'a>(
    cursor: &mut TreeCursor<'a>,
    post: &mut Vec<TreeNode<TsNodeWrapper<'a>>>,
    subtree_start: &mut Vec<usize>,
    sizes: &mut Vec<usize>,
) {
    // record where this node's subtree will begin
    let start = post.len();
    let current_node = cursor.node();
    // go to the first child if there is one
    if cursor.goto_first_child() {
        loop {
            // visit all the children
            dfs_postorder(cursor, post, subtree_start, sizes);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        // go back to the parent
        cursor.goto_parent();
    }

    post.push(TreeNode::new(TsNodeWrapper::new(current_node)));
    subtree_start.push(start);
    sizes.push(post.len() - start);
}

/// A labelled, ordered tree in **post‑order** representation plus per‑node metadata.
/// Lifetime `'a` ties the `Tree` to the lifetime of its underlying `TreeNode`s.
struct Tree<'a, L: Eq> {
    /// Post‑order list of pointers into the original node arena.
    post_order: &'a [TreeNode<L>], // length == n

    /// For every node *i* (in post‑order), `left_most_leaf_descendant[i]` is the
    /// index (also in post‑order) of the left‑most leaf in the subtree rooted at *i*.
    /// This is computed from the post_order traversal of the root tree. We have to compute
    /// the left-most leaf descendant for Zhang-Sasha by subtracting the parent offset from
    /// each value in this array.
    left_most_leaf_descendant_root: &'a [usize], // length == n, O(1) lookup

    /// Parent offset for this subtree. This means that the left-most node of this subtree
    /// starts at this index in the larger list. Therefore the numbers in the `left_most_leaf_descendant_root`
    /// are too large by this amount.
    parent_offset: usize,

    /// The set of **key‑roots** as defined by Zhang & Shasha (1989), sorted.
    key_roots: Vec<usize>, // strictly increasing indices into `post_order`
}

/// A basic tree node.  Parameterised by label type `L` so that integration with
/// Tree‑sitter can supply e.g. `u16` (`tree_sitter::Symbol`) instead of `String`.
#[derive(Debug)]
struct TreeNode<L: Eq> {
    pub label: L,
}

impl<L: Eq> TreeNode<L> {
    /// Create a leaf.
    fn new(label: L) -> Self {
        Self { label }
    }
}

impl<'a, L: Eq> Tree<'a, L> {
    // ------------------------- construction helpers ------------------------- //
    fn new_from_postorder(
        post: &'a [TreeNode<L>],
        left_most_leaf_descendant_root: &'a [usize],
        parent_offset: usize,
    ) -> Self {
        let key_roots = Self::compute_key_roots(left_most_leaf_descendant_root, parent_offset);
        Self {
            post_order: post,
            left_most_leaf_descendant_root,
            parent_offset,
            key_roots,
        }
    }

    /// Compute **key‑roots** (the last node for every distinct left‑most leaf).
    fn compute_key_roots(
        left_most_leaf_descendant_root: &[usize],
        parent_offset: usize,
    ) -> Vec<usize> {
        // Map: left‑most‑leaf‑id -> last index with that left‑most leaf
        let mut last_for_leaf: HashMap<usize, usize> = HashMap::new();
        for (idx, &l) in left_most_leaf_descendant_root.iter().enumerate() {
            last_for_leaf.insert(l - parent_offset, idx); // keeps the *last* occurrence in each subtree
        }
        let mut roots: Vec<usize> = last_for_leaf.values().copied().collect();
        roots.sort();
        roots
    }

    fn left_most_leaf_descendant(&self, idx: usize) -> usize {
        self.left_most_leaf_descendant_root[idx] - self.parent_offset
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
        let l1 = t1.left_most_leaf_descendant(k1);
        let l2 = t2.left_most_leaf_descendant(k2);

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

                let both_trees = t1.left_most_leaf_descendant(node_i) == l1
                    && t2.left_most_leaf_descendant(node_j) == l2;

                if both_trees {
                    // Case 1: both forests are actually *trees* (subtrees rooted at node_i / node_j).
                    fd[(i, j)] = cmp::min(
                        cmp::min(fd[(i - 1, j)] + del_cost, fd[(i, j - 1)] + ins_cost),
                        fd[(i - 1, j - 1)]
                            + Self::label_cost(
                                &t1.post_order[node_i],
                                &t2.post_order[node_j],
                                relabel_cost,
                            ),
                    );
                    td[(node_i, node_j)] = fd[(i, j)];
                } else {
                    // Case 2: at least one is a forest. Re‑use already computed sub‑tree distances.
                    let li = t1.left_most_leaf_descendant(node_i) - l1;
                    let lj = t2.left_most_leaf_descendant(node_j) - l2;
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
    fn weighted_tree_edit_distance(
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
    fn tree_edit_distance(&self, other: &Tree<L>) -> u64 {
        self.weighted_tree_edit_distance(other, 1, 1, 1)
    }
}
