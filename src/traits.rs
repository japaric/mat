//! Traits

use typenum::Unsigned;

/// The transpose operation
pub trait Transpose: Copy {
    /// Transposes the matrix
    fn t(self) -> super::Transpose<Self> {
        super::Transpose { m: self }
    }
}

/// A matrix
pub trait Matrix: UnsafeGet {
    /// Number of rows
    type NROWS: Unsigned;
    /// Number of columns
    type NCOLS: Unsigned;

    /// Returns the element at row `r` and column `c`
    ///
    /// # Panics
    ///
    /// This operation panics if `r` or `c` exceed the matrix dimensions
    fn get(self, r: usize, c: usize) -> Self::Elem {
        assert!(r < self.nrows() && c < self.ncols());

        unsafe { self.unsafe_get(r, c) }
    }

    /// Returns the size of the matrix
    fn size(self) -> (usize, usize) {
        (Self::NROWS::to_usize(), Self::NCOLS::to_usize())
    }

    /// Returns the number of rows of the matrix
    fn nrows(self) -> usize {
        self.size().0
    }

    /// Returns the number of columns of the matrix
    fn ncols(self) -> usize {
        self.size().1
    }
}

/// Unsafe indexing
// NOTE(`: Copy`) this bound is a lint against expression trees that take ownership of `Mat`
pub trait UnsafeGet: Copy {
    /// The matrix element type
    // NOTE(`: Copy`) let's narrow down the problem to matrices that contain only primitive types
    type Elem: Copy;

    /// Returns the element at row `r` and column `c` with performing bounds checks
    unsafe fn unsafe_get(self, r: usize, c: usize) -> Self::Elem;
}

/// Types that have a "zero" value
pub trait Zero {
    /// Returns the value of this type that represents the number zero
    fn zero() -> Self;
}

macro_rules! zero {
    ($($ty:ty),+) => {
        $(
            impl Zero for $ty {
                fn zero() -> Self {
                    0
                }
            }
        )+
    }
}

zero!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl Zero for f32 {
    fn zero() -> f32 {
        0.
    }
}

impl Zero for f64 {
    fn zero() -> f64 {
        0.
    }
}
