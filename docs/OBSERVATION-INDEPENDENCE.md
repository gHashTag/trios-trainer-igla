# Observation independence

**The invariant.** A declared observation parameter may change *what is
measured* and never *what is measured on*.

Stated as the property the harness actually checks:

> The weights at step *k* must not depend on ANY observation parameter.

Anything that only decides how often to look, how hard to look, what to call
the run, or how often to save may move the numbers in the log and the set of
files on disk. It may not move one bit of a checkpoint at a step both settings
emit.

## Why this is a procedure and not another schema field

Eight generations of the record schema exist on disk -- 92 sidecars across
seven of them. Every generation was believed complete when it was written, and
every generation was falsified by exactly one more field that turned out to be
part of the recipe rather than part of the report. Adding field N+1 closes the
instance, never the class.

The motivating defect was found by accident, after 1,851 runs. `gf16_floor`
mutates `embed`, `proj`, `lm_head` and every `ctx` slab in place, and it was
gated on `step % args.eval_every == 0` for the last 30% of training. So
`--eval-every` -- a knob whose entire documented purpose is deciding how often
to take a reading -- decided the artifact. Two seed-47 runs differing in that
flag and nothing else produced different weights and `val_bpb` **2.6141 vs
2.6169**. Runs with different eval cadence were not comparable, and nothing
said so.

Note where the defect was *not*. Rust's type system already guarantees
`evaluate(&model, ..)` cannot mutate the model; the evaluator was never the
problem. The coupling lived one level up, in a training loop keyed on an
observation knob. No amount of care inside the measurement function would have
caught it. A harness that varies one knob at a time and demands byte-identical
checkpoints would have caught it on day one.

That harness is `tests/observation_parameter_independence.rs`. It runs the real
release binary twice per parameter -- same seed, same steps, same corpus, one
knob different -- and compares the `sha256` of every `*.bin` at every step both
runs emitted.

## Parameters currently under test

| Parameter | Values compared | What it is |
|---|---|---|
| `--eval-every` | `50` vs `100` | reading cadence (the motivating case) |
| `TRIOS_EVAL_CHUNKS` | `40` vs `0` (full coverage) | how much of val one reading averages: 40 windows / 5.16% against all 775 |
| `TRIOS_CANON_NAME` | two names | the run's label: checkpoint directory and ledger identity |
| `TRIOS_CHECKPOINT_EVERY` | `50` vs `100` | artifact cadence -- the step *sets* differ by design, so the comparison is over their intersection |
| *(control)* | same settings twice | same knobs must give the same bytes |
| *(negative control)* | `--seed 47` vs `--seed 89` | a RECIPE parameter, asserted **unequal** |

Neither control is ceremony. Without the first, a green file could equally mean
"this trainer is deterministic in nothing and every pair happened to match".
Without the second -- the only `assert_ne!` in the file -- it could mean the
checkpoint bytes do not depend on the run at all, which is not a hypothetical:
`checkpoint::save` was a stub returning `Ok(())` for the whole 1,851-run
campaign. Under a stub, every equality in the table above passes and proves
nothing.

Each variant also has to *prove it took effect*: the run must print the value it
was given (`eval_every=50`, `eval_chunks_target=0`, `checkpoint_every=100`, the
canon name in the `[ckpt]` path) or the case fails as unproven. A knob the
binary silently ignores would otherwise produce two identical runs and a
passing assertion -- which is the exact failure mode that once turned
`TRIOS_CHECKPOINT_EVERY=1_000` into `0` with no warning.

## Result

All five cases pass as of this writing, on aarch64 macOS, at
`--seed 47 --steps 200 --hidden 64 --attn-layers 1`. `--eval-every` is
independent because `gf16_floor` now has its own cadence knob
(`TRIOS_GF16_FLOOR_EVERY`), and `TRIOS_CHECKPOINT_EVERY` is independent because
the artifact cadence was lifted out of the eval guard it used to sit inside.
Both of those were defects; both are now under a test that would have found
them.

## What this does NOT prove

**The list above is not proven complete.** Nothing here establishes that the
four parameters are all the observation parameters the trainer has. This
document does not claim otherwise, and no future version of it should: the
whole history of this file's subject is people believing a list was complete.

What the harness is, precisely:

- a **discovery procedure**, cheap enough to run on every commit, that turns a
  new coupling into a named failure instead of an accident 1,851 runs later;
- a **regression barrier** for the two couplings already found;
- **not** a completeness proof, and **not** a substitute for declaring, in
  every artifact, which parameters were in force.

Two things follow. First, when a new knob is added to the trainer, it must be
classified as *recipe* or *observation*, and if it is observation it must be
added to the table above. Second, a failure in this harness is the procedure
succeeding: it has found a new observation-coupled parameter. The response is
to decouple the training loop from it, exactly as `gf16_floor` was decoupled --
never to relax the assertion.

`phi^2 + phi^-2 = 3 | TRINITY`
