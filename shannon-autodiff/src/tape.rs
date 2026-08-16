//! `Tape<'a>` — the reverse-mode record of taped launches (Day-6 plan §4.1). 📦
//!
//! THE design decision: records hold BORROWS (`Box<dyn Fn + 'a>`), not `Arc`
//! handles — the week-1 plan's "the tape stores references, not values" made
//! literal. Consequences, all deliberate:
//!
//! - `backward(self)` CONSUMES the tape. The reverse walk runs, the records
//!   drop, and every borrow they held ends — which is exactly what makes
//!   `&mut params` legal again for the optimizer step. A reusable `reset()`
//!   would extend the borrow region across the step and (correctly) fail to
//!   compile.
//! - The overwrite hazard is a COMPILE ERROR, not a LIMITATIONS entry: taped
//!   map-shaped ops downgrade their `&'a mut` output to `&'a` and return it,
//!   so nothing can mutate a recorded buffer until backward. (Warp needs a
//!   runtime verifier for this, `wp.config.verify_autograd_array_access`,
//!   warp/_src/tape.py:285; rustc does it for free.) The one hole: a raw
//!   const-ref scatter launch aliasing a taped buffer — convention only,
//!   see the Day-6 plan, pitfall 6.
//! - Records are not `Send`: the week-1 tape is single-threaded host code.
//!
//! Seeding `loss.grad = 1` is the CALLER's job, in the mutation phase BEFORE
//! anything records — at backward time the records hold shared borrows of the
//! loss array, so `grad_mut` is unreachable. See `ops::begin_iteration` in
//! shannon-examples and the Day-6 plan §4.2.
//!
//! The tape is host code: `Box`/`Vec`/`dyn Fn` are all available — the
//! no-heap rule is device-side only (CUDA-OXIDE-AUTODIFF-REFERENCE §5).

/// One taped launch: a label for diagnostics and the type-erased adjoint
/// replay closure, capturing `&'a` borrows of every buffer it needs.
pub struct LaunchRecord<'a> {
    pub label: &'static str,
    pub dim: usize,
    replay: Box<dyn Fn() -> anyhow::Result<()> + 'a>,
}

#[derive(Default)]
pub struct Tape<'a> {
    records: Vec<LaunchRecord<'a>>,
    paused: bool,
}

impl<'a> Tape<'a> {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            paused: false,
        }
    }

    /// Record one adjoint replay. No-op while paused.
    pub fn record(
        &mut self,
        label: &'static str,
        dim: usize,
        replay: impl Fn() -> anyhow::Result<()> + 'a,
    ) {
        if !self.paused {
            self.records.push(LaunchRecord {
                label,
                dim,
                replay: Box::new(replay),
            });
        }
    }

    /// Suspend recording — untaped forward queries (the Chamfer
    /// correspondence) run between `pause()` and `resume()`. In week 1
    /// recording is opt-in (only `ops::` helpers record), so this is a
    /// belt-and-braces interlock; week-2's ambient-recording `launch!` makes
    /// it load-bearing. No RAII guard: a guard holding `&mut Tape` would
    /// block the very calls it is supposed to be guarding.
    pub fn pause(&mut self) {
        self.paused = true;
    }
    pub fn resume(&mut self) {
        self.paused = false;
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    /// Record labels in forward order — for tape summaries and tests.
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.records.iter().map(|r| r.label)
    }

    /// The reverse walk. Seeding is NOT done here (see module docs) — the
    /// caller seeded `loss.grad = 1` in the mutation phase. Consumes the
    /// tape; on return every borrow the records held has ended.
    pub fn backward(self) -> anyhow::Result<()> {
        for rec in self.records.iter().rev() {
            (rec.replay)().map_err(|e| anyhow::anyhow!("backward of `{}`: {e:?}", rec.label))?;
        }
        Ok(())
    }
}
