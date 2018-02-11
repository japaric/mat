//! Statically sized matrices for `no_std` applications
//!
//! This library provides support for creating and performing mathematical operations on *statically
//! sized* matrices. That is matrices whose dimensions are known at compile time. The main use case
//! for this library are `no_std` programs where a memory allocator is not available.
//!
//! Since the matrices are statically allocated the dimensions of the matrix are stored in the type
//! system and used to prevent invalid operations (e.g. adding a 3x4 matrix to a 4x3 matrix) at
//! compile time.
//!
//! For performance reasons all operations, except for the indexing `get` method, are lazy and
//! perform no actual computation. An expression like `a * b + c;` simply builds an *expression
//! tree*. `get` can be used to force evaluation of such a tree; see below:
//!
//! ```
//! #![feature(proc_macro)]
//!
//! use mat::mat;
//! use mat::traits::Matrix;
//!
//! // 2 by 3 matrix
//! let a = mat![
//!     [1, 2, 3],
//!     [3, 4, 5],
//! ];
//!
//! // 3 by 2 matrix
//! let b = mat![
//!     [1, 2],
//!     [3, 4],
//!     [5, 6],
//! ];
//!
//! // build an expression tree
//! let c = &a * &b;
//!
//! // partially evaluate the tree
//! assert_eq!(c.get(0, 0), 22);
//! ```
//!
//! This program does *not* allocate and compute a whole new matrix C of size 2x2; it simply
//! performs the operations required to get the element at row 0 and column 0 that such matrix C
//! would have.
//!
//! # Out of scope
//!
//! The following features are out of scope for this library.
//!
//! - Operation that require dynamic memory allocation
//! - SIMD acceleration
//! - n-dimensional arrays
//!
//! If you are looking for such features check out the [`ndarray`] crate.
//!
//! [`ndarray`]: https://crates.io/crates/ndarray
//!
//! # Development status
//!
//! This library is unlikely to see much development until support for [const generics] lands in the
//! compiler.
//!
//! [const generics]: https://github.com/rust-lang/rust/issues/44580

#![deny(missing_docs)]
#![deny(warnings)]
#![feature(proc_macro)]
#![feature(unsize)]
#![no_std]

extern crate mat_macros;
#[doc(hidden)]
pub extern crate typenum;

use core::ops;
use core::marker::{PhantomData, Unsize};
use core::fmt;

pub use mat_macros::mat;
use typenum::Unsigned;

pub mod traits;

use traits::{Matrix, UnsafeGet, Zero};

/// Statically allocated (row major order) matrix
#[derive(Clone)]
pub struct Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NCOLS: Unsigned,
    NROWS: Unsigned,
    T: Copy,
{
    buffer: BUFFER,
    ty: PhantomData<[T; 0]>,
    nrows: PhantomData<NROWS>,
    ncols: PhantomData<NCOLS>,
}

/// The product of two matrices
#[derive(Clone, Copy)]
pub struct Product<L, R> {
    l: L,
    r: R,
}

/// The sum of two matrices
#[derive(Clone, Copy)]
pub struct Sum<L, R> {
    l: L,
    r: R,
}

/// The transpose of a matrix
#[derive(Clone, Copy)]
pub struct Transpose<M> {
    m: M,
}

impl<T, BUFFER, NROWS, NCOLS> Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NROWS: Unsigned,
    NCOLS: Unsigned,
    T: Copy,
{
    #[doc(hidden)]
    pub unsafe fn new(buffer: BUFFER) -> Self {
        Mat {
            buffer,
            ty: PhantomData,
            nrows: PhantomData,
            ncols: PhantomData,
        }
    }
}

impl<T, BUFFER, NROWS, NCOLS> fmt::Debug for Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NROWS: Unsigned,
    NCOLS: Unsigned,
    T: Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut is_first = true;
        let slice: &[T] = &self.buffer;
        f.write_str("[")?;
        for row in slice.chunks(NCOLS::to_usize()) {
            if is_first {
                is_first = false;
            } else {
                f.write_str(", ")?;
            }

            write!(f, "{:?}", row)?;
        }
        f.write_str("]")
    }
}

impl<'a, T, BUFFER, NROWS, NCOLS> Matrix for &'a Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NROWS: Unsigned,
    NCOLS: Unsigned,
    T: Copy,
{
    type NROWS = NROWS;
    type NCOLS = NCOLS;
}

impl<'a, T, BUFFER, NROWS, NCOLS> UnsafeGet for &'a Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NROWS: Unsigned,
    NCOLS: Unsigned,
    T: Copy,
{
    type Elem = T;

    unsafe fn unsafe_get(self, r: usize, c: usize) -> T {
        let slice: &[T] = &self.buffer;
        *slice.get_unchecked(r * NCOLS::to_usize() + c)
    }
}

impl<'a, T, BUFFER, NROWS, NCOLS, R> ops::Mul<R> for &'a Mat<T, BUFFER, NROWS, NCOLS>
where
    BUFFER: Unsize<[T]>,
    NROWS: Unsigned,
    NCOLS: Unsigned,
    T: Copy,
    R: Matrix<NROWS = NCOLS>,
{
    type Output = Product<&'a Mat<T, BUFFER, NROWS, NCOLS>, R>;

    fn mul(self, rhs: R) -> Self::Output {
        Product { l: self, r: rhs }
    }
}

impl<M> traits::Transpose for M
where
    M: Matrix,
{
}

impl<M> Matrix for Transpose<M>
where
    M: Matrix,
{
    // NOTE reversed size!
    type NROWS = M::NCOLS;
    type NCOLS = M::NROWS;
}

impl<M> UnsafeGet for Transpose<M>
where
    M: Matrix,
{
    type Elem = M::Elem;

    unsafe fn unsafe_get(self, r: usize, c: usize) -> M::Elem {
        // NOTE reversed indices!
        self.m.unsafe_get(c, r)
    }
}

impl<L, R> ops::Mul<R> for Transpose<L>
where
    L: Matrix,
    R: Matrix<NROWS = L::NROWS>,
{
    type Output = Product<Transpose<L>, R>;

    fn mul(self, rhs: R) -> Self::Output {
        Product { l: self, r: rhs }
    }
}

impl<L, R, T> Matrix for Product<L, R>
where
    L: Matrix<Elem = T>,
    R: Matrix<Elem = T>,
    T: ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Copy + Zero,
{
    type NROWS = L::NROWS;
    type NCOLS = R::NCOLS;
}

impl<T, L, R> UnsafeGet for Product<L, R>
where
    L: Matrix<Elem = T>,
    R: Matrix<Elem = T>,
    T: ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Copy + Zero,
{
    type Elem = T;

    unsafe fn unsafe_get(self, r: usize, c: usize) -> T {
        let mut sum = T::zero();
        for i in 0..self.l.ncols() {
            sum = sum + self.l.unsafe_get(r, i) * self.r.unsafe_get(i, c);
        }
        sum
    }
}

impl<L, R, RHS> ops::Add<RHS> for Product<L, R>
where
    L: Matrix,
    R: Matrix,
    RHS: Matrix<NROWS = L::NROWS, NCOLS = R::NCOLS>,
{
    type Output = Sum<Product<L, R>, RHS>;

    fn add(self, rhs: RHS) -> Self::Output {
        Sum { l: self, r: rhs }
    }
}

impl<T, L, R> Matrix for Sum<L, R>
where
    L: Matrix<Elem = T>,
    R: Matrix<Elem = T>,
    T: ops::Add<T, Output = T> + Copy,
{
    type NROWS = L::NROWS;
    type NCOLS = L::NCOLS;
}

impl<T, L, R> UnsafeGet for Sum<L, R>
where
    L: Matrix<Elem = T>,
    R: Matrix<Elem = T>,
    T: ops::Add<T, Output = T> + Copy,
{
    type Elem = T;

    unsafe fn unsafe_get(self, r: usize, c: usize) -> T {
        self.l.unsafe_get(r, c) + self.r.unsafe_get(r, c)
    }
}
