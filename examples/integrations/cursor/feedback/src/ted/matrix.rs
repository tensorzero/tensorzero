//! A *minimal* 2‑D matrix abstraction that mimics the tiny slice of
//! `nalgebra::DMatrix` you use in `tree_edit_distance.rs` (namely
//! `DMatrix::<u64>::zeros` and indexing via `(row, col)` tuple).
//!
//!
//! Only two operations are implemented because they are all the Zhang–Shasha
//! DP needs:
//!   1. `Matrix::<T>::zeros(rows, cols)` *(T: Default + Clone)*
//!   2. `matrix[(i, j)]` and `matrix[(i, j)] = value` via `Index`/`IndexMut`.
//!
//! TODO: preallocate this to the largest size needed to avoid repeated allocations
//! and frees.

use std::ops::{Index, IndexMut};

/// A simple row‑major dense matrix.
#[derive(Clone, Debug)]
pub struct Matrix<T> {
    data: Vec<T>, // flat storage: row * cols + col
    rows: usize,
    cols: usize,
}

impl<T: Default + Clone> Matrix<T> {
    /// Construct a `rows × cols` matrix filled with `T::default()`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        debug_assert!(
            rows > 0 && cols > 0,
            "Matrix::zeros called with zero dimension"
        );
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }
}

impl<T> Matrix<T> {
    /// Internal helper: convert (row, col) to flat index.
    #[inline]
    fn idx(&self, r: usize, c: usize) -> usize {
        debug_assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        r * self.cols + c
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        &self.data[self.idx(r, c)]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        let idx = self.idx(r, c);
        &mut self.data[idx]
    }
}
