#![deny(warnings)]
#![feature(proc_macro)]

extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro::TokenStream;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::synom::Synom;
use syn::{Expr, ExprArray, Ident};

struct Mat {
    rows: Punctuated<ExprArray, Token![,]>,
}

impl Synom for Mat {
    named!(parse -> Self, do_parse!(
        rows: call!(Punctuated::parse_terminated_nonempty) >> (Mat { rows })
    ));
}

/// A macro to construct matrices
#[proc_macro]
pub fn mat(input: TokenStream) -> TokenStream {
    let mat: Mat = syn::parse(input).unwrap();

    // check consistent number of columns
    let nrows = mat.rows.len();
    let ncols = mat.rows.iter().next().expect("BUG: zero rows").elems.len();

    for row in mat.rows.iter() {
        for (i, expr) in row.elems.iter().enumerate() {
            if i >= ncols {
                expr.span()
                    .unstable()
                    .error(format!("expected {} elements", ncols,))
                    .emit();
            }
        }
    }

    let size = nrows * ncols;
    let elems: Vec<&Expr> = mat.rows.iter().flat_map(|row| row.elems.iter()).collect();

    let nrows_ty = Ident::from(format!("U{}", nrows));
    let ncols_ty = Ident::from(format!("U{}", ncols));

    quote!(unsafe {
        extern crate mat;
        mat::Mat::<_, [_; #size], mat::typenum::#nrows_ty, mat::typenum::#ncols_ty>::new([#(#elems,)*])
    }).into()
}
