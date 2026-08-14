//! shannon-autodiff — the gradient oracle (Day 1), Tape + optimizers (Day 6).
//!
//! The oracle was built BEFORE any adjoint it validates — the plan's most
//! important process decision (week-1 plan §11.3). The classic failure mode
//! of a hand-rolled AD system is plausible-but-wrong gradients; finite
//! differences are the only independent oracle available.
//!
//! Day 6 adds the rest of the crate's reason to exist: the borrow-scoped
//! `Tape<'a>` (records hold `&'a` borrows; `backward` consumes the tape and
//! releases them) and the host-side `Sgd`/`Adam` optimizers.

mod gradcheck;
mod optim;
mod tape;

pub use gradcheck::{GradError, grad_fd, gradcheck};
pub use optim::{Adam, Sgd};
pub use tape::{LaunchRecord, Tape};
