//! `shannon_gpu_kernels!` / `shannon_cpu_kernels!` — collapse the per-kernel
//! adapter boilerplate into one declaration line per kernel (week-1 plan
//! task 2.0; Day-2 task 2.7).
//!
//! # The two constraints that shape this macro
//!
//! **1. `#[cuda_module]` scans its module body SYNTACTICALLY for `#[kernel]`**
//! (matching the last attribute path segment only). Anything still wrapped in
//! an unexpanded macro invocation — or behind a `mod x;` / `include!` boundary
//! — is invisible and SILENTLY skipped. Attributes on an item are expanded
//! *before* bang-macros inside the item's body, so the module handed to
//! `#[cuda_module]` must already contain literal `fn` items. This macro is
//! therefore continuation-passing: it parses every row to finished kernel
//! tokens FIRST, and only the final state emits the `#[cuda_module]` module
//! around them. (Same reasoning as cuda-oxide's own `vectorization` example.)
//!
//! **2. Parameter types must stay raw tokens.** A `$t:ty` fragment reaches the
//! downstream `#[kernel]` proc macro as an opaque nonterminal, which defeats
//! cuda-macros' syntactic parameter classification (`&[T]` vs scalar) and
//! surfaces as E0637 deep in its expansion. The muncher splits parameters on
//! top-level commas as plain `tt`s. Consequence: parameter types must not
//! contain top-level commas (no bare tuples) — fine for kernel ABIs, which
//! cuda-oxide restricts anyway.
//!
//! # Requirements on the invoking crate
//!
//! GPU macro: direct dependencies on `cuda-device`, `cuda-core`, `cuda-host`
//! (the `#[cuda_module]` expansion emits absolute `::cuda_core::…` paths).
//! CPU macro: a direct dependency on `rayon`.
//!
//! # Row shape
//!
//! ```text
//! name(param: Type, …) -> Ret = path::to::body;
//! ```
//!
//! where the body function has signature `fn(i: usize, params…) -> Ret`.
//! The generated GPU kernel writes through `DisjointSlice<Ret>` (race-free by
//! construction); the generated CPU function writes through `&mut [Ret]` under
//! rayon. Kernels that do not fit the elementwise map shape (adjoints,
//! scatter) go in the trailing `@raw { … }` block, verbatim.

/// Emit a complete `#[cuda_module] pub mod kernels { … }` from declaration
/// rows plus optional verbatim items.
#[macro_export]
macro_rules! shannon_gpu_kernels {
    ( $($input:tt)* ) => {
        $crate::__shannon_gpu_build! { @rows [] $($input)* }
    };
}

/// Internal CPS builder for `shannon_gpu_kernels!`. Do not use directly.
///
/// States:
///   `@rows  [acc] remaining-rows…`                — consume one row header
///   `@param [acc] hdr… [sig] [names] (params) (rest)` — split params on commas
///   `@ty    …ditto… [ty] (params) (rest)`         — accumulate one type
///   `@emit`                                       — append one finished fn to acc
/// Terminal arms wrap `acc` (+ optional raw items) in the `#[cuda_module]`.
#[doc(hidden)]
#[macro_export]
macro_rules! __shannon_gpu_build {
    // ── Terminal: all rows consumed ─────────────────────────────────────────
    (@rows [$($acc:tt)*] @raw { $($extra:tt)* }) => {
        #[cuda_device::cuda_module]
        pub mod kernels {
            #[allow(unused_imports)]
            use super::*;
            $($acc)*
            $($extra)*
        }
    };
    (@rows [$($acc:tt)*]) => {
        #[cuda_device::cuda_module]
        pub mod kernels {
            #[allow(unused_imports)]
            use super::*;
            $($acc)*
        }
    };
    // ── Consume one row header; hand its params to the splitter ─────────────
    (@rows [$($acc:tt)*]
        $(#[$meta:meta])* $name:ident( $($params:tt)* ) -> $ret:ty = $body:path;
        $($rest:tt)*
    ) => {
        $crate::__shannon_gpu_build! {
            @param [$($acc)*] [$(#[$meta])*] $name [$ret] [$body] [] [] ( $($params)* , ) ( $($rest)* )
        }
    };
    // ── Param splitter ──────────────────────────────────────────────────────
    // Done (possibly the lone comma of an empty list) → emit this kernel.
    (@param [$($acc:tt)*] [$($meta:tt)*] $name:ident [$ret:ty] [$body:path]
        [$($sig:tt)*] [$($pn:ident)*] ( $(,)? ) ( $($rest:tt)* )
    ) => {
        $crate::__shannon_gpu_build! {
            @rows [
                $($acc)*
                $($meta)*
                #[allow(clippy::too_many_arguments)] // arity mirrors the user's row
                #[cuda_device::kernel]
                pub fn $name($($sig)* mut __out: cuda_device::DisjointSlice<$ret>) {
                    let __idx = cuda_device::thread::index_1d();
                    let __i = __idx.get();
                    if let Some(__slot) = __out.get_mut(__idx) {
                        *__slot = $body(__i, $($pn),*);
                    }
                }
            ]
            $($rest)*
        }
    };
    // `name :` starts a parameter → switch to type accumulation.
    (@param $acc:tt $m:tt $name:ident $r:tt $b:tt [$($sig:tt)*] [$($pn:ident)*]
        ( $p:ident : $($ptoks:tt)* ) $rest:tt
    ) => {
        $crate::__shannon_gpu_build! {
            @ty $acc $m $name $r $b [$($sig)* $p :] [$($pn)* $p] [] ( $($ptoks)* ) $rest
        }
    };
    // Top-level comma ends the current type → back to @param.
    (@ty $acc:tt $m:tt $name:ident $r:tt $b:tt [$($sig:tt)*] $pns:tt [$($ty:tt)*]
        ( , $($ptoks:tt)* ) $rest:tt
    ) => {
        $crate::__shannon_gpu_build! {
            @param $acc $m $name $r $b [$($sig)* $($ty)* ,] $pns ( $($ptoks)* ) $rest
        }
    };
    // Any other token belongs to the current type.
    (@ty $acc:tt $m:tt $name:ident $r:tt $b:tt $sig:tt $pns:tt [$($ty:tt)*]
        ( $tk:tt $($ptoks:tt)* ) $rest:tt
    ) => {
        $crate::__shannon_gpu_build! {
            @ty $acc $m $name $r $b $sig $pns [$($ty)* $tk] ( $($ptoks)* ) $rest
        }
    };
}

/// Emit rayon CPU adapters from the same declaration rows.
///
/// (No muncher needed here: the expansion is consumed by rustc directly,
/// where `$t:ty` nonterminals are perfectly transparent.)
#[macro_export]
macro_rules! shannon_cpu_kernels {
    (
        $( $(#[$meta:meta])* $name:ident( $($p:ident : $t:ty),* $(,)? ) -> $ret:ty = $body:path; )*
    ) => {
        $(
            $(#[$meta])*
            #[allow(clippy::too_many_arguments)] // arity mirrors the user's row
            pub fn $name($($p: $t,)* __out: &mut [$ret]) {
                use ::rayon::prelude::*;
                __out
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(__i, __slot)| *__slot = $body(__i, $($p),*));
            }
        )*
    };
}
