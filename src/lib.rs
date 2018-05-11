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
//! #[macro_use]
//! extern crate mat;
//!
//! use mat::traits::Matrix;
//!
//! fn main() {
//!     // 2 by 3 matrix
//!     let a = mat!(i32, [
//!         [1, 2, 3],
//!         [3, 4, 5],
//!     ]);
//!
//!     // 3 by 2 matrix
//!     let b = mat!(i32, [
//!         [1, 2],
//!         [3, 4],
//!         [5, 6],
//!     ]);
//!
//!     // build an expression tree
//!     let c = &a * &b;
//!
//!     // partially evaluate the tree
//!     assert_eq!(c.get(0, 0), 22);
//! }
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
//! - Operations that require dynamic memory allocation
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
#![no_std]

extern crate generic_array;

pub mod traits;

use core::marker::PhantomData;
use core::{fmt, ops};

pub use generic_array::typenum::consts;
use generic_array::typenum::consts::U1;
pub use generic_array::typenum::Quot as __Quot;
use generic_array::typenum::{Prod, Unsigned};
use generic_array::{ArrayLength, GenericArray};

use traits::{Matrix, UnsafeGet, Zero};

/// Macro to construct a `Mat`rix
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate mat;
///
/// fn main() {
///     let a = mat!(i32, [
///         [1, 2],
///         [3, 4],
///     ]);
///
///     assert_eq!(a[0][0], 1);
///     assert_eq!(a[0][1], 2);
///     assert_eq!(a[1][0], 3);
///     assert_eq!(a[1][1], 4);
/// }
/// ```
#[macro_export]
macro_rules! mat {
    ($ty:ty, [$([$($e:expr),*],)+]) => ({
        extern crate core;

        type NROWS = __nrows!($crate::consts::U0; [ $([ $($e),* ],)* ] );
        type NELEMS = __nelems!($crate::consts::U0; [ $( $($e),* ,)* ]);
        type NCOLS = $crate::__Quot<NELEMS, NROWS>;

        unsafe {
            core::mem::transmute::<_, $crate::Mat<$ty, NROWS, NCOLS>>(
                [ $( $({ let e: $ty = $e; e }),* ),* ]
            )
        }
    })
}

#[doc(hidden)]
#[macro_export]
macro_rules! __nrows {
    ($i:ty; []) => {
        $i
    };

    ($i:ty; [ [$($head:expr),*], $( [$($tail:expr),*] ,)*]) => {
        __nrows!($crate::__Inc<$i>; [$( [$($tail),*] ,)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __nelems {
    ($i:ty; []) => {
        $i
    };
    ($i:ty; [$head:expr, $($tail:expr,)*]) => {
        __nelems!($crate::__Inc<$i>; [ $($tail,)* ])
    };
}

#[doc(hidden)]
pub type __Inc<T> = generic_array::typenum::Sum<T, U1>;

/// Row view into a `Mat`rix
pub struct Row<T, NCOLS>
where
    NCOLS: ArrayLength<T>,
{
    buffer: GenericArray<T, NCOLS>,
}

impl<T, NCOLS> ops::Index<usize> for Row<T, NCOLS>
where
    NCOLS: ArrayLength<T>,
{
    type Output = T;

    fn index(&self, c: usize) -> &T {
        assert!(c < NCOLS::to_usize());

        unsafe { self.buffer.get_unchecked(c) }
    }
}

impl<T, NCOLS> ops::IndexMut<usize> for Row<T, NCOLS>
where
    NCOLS: ArrayLength<T>,
{
    fn index_mut(&mut self, c: usize) -> &mut T {
        assert!(c < NCOLS::to_usize());

        unsafe { self.buffer.get_unchecked_mut(c) }
    }
}

/// Statically allocated (row major order) matrix
#[derive(Clone)]
pub struct Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS>,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
{
    buffer: GenericArray<T, Prod<NROWS, NCOLS>>,
    _nrows: PhantomData<NROWS>,
    _ncols: PhantomData<NCOLS>,
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

impl<T, NROWS, NCOLS> fmt::Debug for Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS>,
    NCOLS: Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.buffer.chunks(NCOLS::to_usize()))
            .finish()
    }
}

impl<'a, T, NROWS, NCOLS> Matrix for &'a Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS> + Unsigned,
    NCOLS: Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
    T: Copy,
{
    type NROWS = NROWS;
    type NCOLS = NCOLS;
}

impl<'a, T, NROWS, NCOLS> UnsafeGet for &'a Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS> + Unsigned,
    NCOLS: Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
    T: Copy,
{
    type Elem = T;

    unsafe fn unsafe_get(self, r: usize, c: usize) -> T {
        *self.buffer.get_unchecked(r * NCOLS::to_usize() + c)
    }
}

impl<T, NROWS, NCOLS> ops::Index<usize> for Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS> + Unsigned,
    NCOLS: ArrayLength<T> + Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
{
    type Output = Row<T, NCOLS>;

    fn index(&self, r: usize) -> &Row<T, NCOLS> {
        assert!(r < NROWS::to_usize());

        unsafe {
            &*(self.buffer.get_unchecked(r * NCOLS::to_usize()) as *const _ as *const Row<_, _>)
        }
    }
}

impl<T, NROWS, NCOLS> ops::IndexMut<usize> for Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS> + Unsigned,
    NCOLS: ArrayLength<T> + Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
{
    fn index_mut(&mut self, r: usize) -> &mut Row<T, NCOLS> {
        assert!(r < NROWS::to_usize());

        unsafe {
            &mut *(self.buffer.get_unchecked_mut(r * NCOLS::to_usize()) as *mut _ as *mut Row<_, _>)
        }
    }
}

impl<'a, T, NROWS, NCOLS, R> ops::Mul<R> for &'a Mat<T, NROWS, NCOLS>
where
    NROWS: ops::Mul<NCOLS>,
    NCOLS: Unsigned,
    Prod<NROWS, NCOLS>: ArrayLength<T>,
    R: Matrix<NROWS = NCOLS>,
{
    type Output = Product<&'a Mat<T, NROWS, NCOLS>, R>;

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
