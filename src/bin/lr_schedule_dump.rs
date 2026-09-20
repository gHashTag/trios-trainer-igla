//! Dump the headline learning-rate schedule, one line per step, as raw bits.
//!
//! WHY
//!
//! `scripts/det_math_isa_probe.py` runs the trainer for TEN steps, because a
//! 12000-step run on two ISAs is not a thing one can iterate on. With
//! `warmup = steps / 10 = 1` at ten steps, `cosine_lr` takes its linear-warmup
//! branch once and then its cosine branch nine times on `p` values that are
//! multiples of 1/9 - and the probe reports the learning-rate bits as `same` at
//! every one of those steps. It is therefore tempting to read a 10-step
//! cross-ISA MATCH as evidence about the 12000-step headline.
//!
//! That reading is FALSE, and this binary is what makes it false by
//! measurement rather than by caution. `cosine_lr` calls `f32::cos`, which is a
//! libm call; the `det-math` feature replaces `exp` and NOTHING ELSE. If `cosf`
//! disagrees between two instruction sets on any of the 12000 arguments the
//! headline schedule actually visits, then the headline diverges through a path
//! the forward pass never touches, and no amount of determinism in `exp` can
//! close it.
//!
//! HOW
//!
//! By calling `trios_trainer::train_loop::cosine_lr`. Not a copy of it. A copy
//! would make "the dump matches the trainer" an assumption, and this repository
//! has been burned before by a replica that drifted from the thing it modelled.
//!
//! The parameters are the headline run's, and they are not configurable: no
//! arguments, no environment reads, no file I/O. `run_single` sets
//! `warmup = args.steps / 10` and iterates `for step in 1..=args.steps`, so
//! that is exactly the range dumped here - 12000 lines, step 1 through step
//! 12000. Step 0 is not dumped because the trainer never evaluates it.
//!
//! OUTPUT
//!
//! One line per step on stdout:
//!
//!     LR <step> <8 hex digits, the u32 bit pattern of the f32 lr>
//!
//! Bits, not a decimal rendering, because the question is byte-identity and a
//! decimal rendering rounds two different floats onto one string.

use trios_trainer::train_loop::cosine_lr;

/// The headline run: `--steps 12000 --lr 0.003`, warmup = steps / 10.
const MAX_STEPS: usize = 12_000;
const BASE_LR: f32 = 0.003;

fn main() {
    let warmup = MAX_STEPS / 10;
    println!("# lr_schedule_dump max_steps={MAX_STEPS} warmup={warmup} base_lr={BASE_LR}");
    for step in 1..=MAX_STEPS {
        let lr = cosine_lr(step, MAX_STEPS, BASE_LR, warmup);
        println!("LR {} {:08x}", step, lr.to_bits());
    }
}
