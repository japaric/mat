# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [v0.2.0] - 2018-05-11

### Added

- Implementations of `Index` and `IndexMut` to `Mat` to allow C like indexing. `&mat[r]` returns a
  view into one of `r`-th row of the matrix; `mat[r][c]` returns the element at row `r` and column
  `c`.

### Changed

- [breaking-change] The type parameters of `Mat` have changed. The new type signature is now `Mat<T,
  NROWS, NCOLS>`.

- [breaking-change] The procedural `mat!` macro have been replaced by a `macro_rules!` macro. The
  syntax of the new macro is slightly different: it must first include the type of the elements of
  the matrix and then the matrix itself. See the API docs for an example.

## [v0.1.1] - 2018-04-08

### Fixed

- Updated dependencies to fix the build on recent nightlies

## v0.1.0 - 2018-02-19

Initial release

[Unreleased]: https://github.com/japaric/mat/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/japaric/mat/compare/v0.1.1...v0.2.0
[v0.1.1]: https://github.com/japaric/mat/compare/v0.1.0...v0.1.1
