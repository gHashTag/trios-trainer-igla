# The fleet count

**Status of the question `[RESOLVED 2026-08-06, with one figure withdrawn]`.**

Three numbers have circulated in this programme's own artifacts as if they were
the same quantity: **1,851**, **1,878** and **7,927**. Earlier versions of
[EVAL-UNCERTAINTY.md](EVAL-UNCERTAINTY.md) and
[IGLA_V6_FINAL_RESULTS.md](../IGLA_V6_FINAL_RESULTS.md) called them "three
different figures for overlapping populations ... Do not quote any of them as
the fleet size."

That instruction was right to refuse the number and **wrong about the reason**.
The populations are not overlapping and the figures are not in conflict. They
count **three different things on two different machines**, and only one of them
was ever a fleet size at all. The correction matters because "our own sources
disagree" is what a counterparty hears as evasion, whereas "these count three
different things, here is which" is a decomposition.

| Figure | What it actually counts | Where | Standing |
|---|---|---|---|
| **1,878** | Rows in the Railway `strategy_queue` table | Railway cloud, audited 2026-05-02 | **The fleet size**, on a stated method. Not re-derived this pass. |
| **8,037** of **8,195** `*.log` (was 7,927) | **v6-wave sweep log files** -- the `v6_*.log` family only, not every log in the directory | Local Mac, v6 sweep 2026-05-25/26 | Correct as a file count **of one wave**, **never a fleet size**. Re-derive it with its denominator, do not quote it bare. |
| **1,851** | -- | -- | **WITHDRAWN.** No derivation exists anywhere. |

The two populations are **disjoint**: different hardware (Railway cloud vs one
local Mac), different dates (2026-05-02 vs 2026-05-25/26), different recording
mechanism (a Postgres table vs stdout redirected to files). Neither is a subset
of the other, and nothing establishes a run appearing in both.

---

## 1,878 -- the Railway fleet, and the only one of the three that is a fleet size

**Method, in the source's own words:** "Railway logs + strategy_queue audit."

**Decomposition, which closes exactly:**

| Category | Count | Corpus |
|---|---:|---|
| Explicit corpus tag | 3 | `tiny_shakespeare` (3 MEGA-ASHA-R2 runs) |
| No corpus tag (entrypoint default applied) | 1,875 | `tiny_shakespeare` |
| Explicit `corpus="fineweb"` | **0** | -- no fineweb run ever executed |
| **All runs** | **1,878** | |

3 + 1,875 + 0 = 1,878. An internally consistent decomposition into named,
mutually exclusive categories is materially stronger evidence than a bare total,
because a transcription error in a bare total is undetectable and here it is not.

**Sources**, both dated 2026-05-02, in `/Users/playra/trios-railway`:

- `golden_sunflowers_crosslinks/issues/P0_scarab_corpus_catastrophe.md:29` --
  the inventory table above; corroborated at that file's line 192.
- `golden_sunflowers_crosslinks/issues/trios442_withdrawal_addendum.md:11` and
  `:77` -- "Full forensic inventory of all 1878 `strategy_queue` runs".

### The caveat, stated plainly

**This figure was NOT re-derived in this pass, and cannot be from here.** The
Postgres instance it counted is unreachable from this machine, and **no dump of
`strategy_queue` is committed anywhere**. What `trios-railway` commits is the
*schema* and the *queries* -- `migrations/0005_strategy_queue_final_metrics.sql`,
`scripts/forensic_corpus_inventory.sql` -- not the rows they ran against.

So 1,878 is defensible as **a figure produced by a stated method, with an
internally consistent decomposition, recorded contemporaneously in two
independent documents**. It is *not* defensible as a figure this repository can
reproduce on demand. Say it that way. The distinction is the entire product.

---

## 8,037 -- the v6-wave log files on one workstation, and never a fleet size

The number long quoted as **7,927** was a `ls | wc -l` of `.trinity/results/v6_*.log`
taken on one workstation on one day in May 2026. It reads **8,037** today. It was
never a count of experiments; it is a count of files, and it drifted the moment
anything else wrote to that directory.

**And it is not the count of log files in that directory.** `v6_*.log` is one
family among several. The directory holds **8,195** `*.log` files. A reader who
runs the obvious `ls .trinity/results/*.log | wc -l` gets **8,195**, finds
**8,037** in the table above, and has just been handed a fourth disagreeing
figure by the document whose entire purpose is to remove disagreement. That is
why the label is now "v6-wave sweep log files" and why the denominator travels
with the number everywhere it appears -- in this file and in the script's own
output.

### The decomposition, re-derived here, and it closes

Run in `.trinity/results/` on 2026-08-06. `find`, not `ls`, because the argument
list runs to over 8,000 entries; `-maxdepth 1 -type f` so no subdirectory or
directory-name match can enter the count:

```
find . -maxdepth 1 -type f -name '*.log'              | wc -l   # 8195
find . -maxdepth 1 -type f -name 'v6_*.log'           | wc -l   # 8037
find . -maxdepth 1 -type f -name 'v2_*.log'           | wc -l   #   30
find . -maxdepth 1 -type f -name 'v3_*.log'           | wc -l   #   36
find . -maxdepth 1 -type f -name 'v4_*.log'           | wc -l   #    9
find . -maxdepth 1 -type f -name 'v5_*.log'           | wc -l   #   11
find . -maxdepth 1 -type f -name 'format_sweep_*.log' | wc -l   #   19

find . -maxdepth 1 -type f -name '*.log' \
  ! -name 'v6_*.log' ! -name 'v2_*.log' ! -name 'v3_*.log' \
  ! -name 'v4_*.log' ! -name 'v5_*.log' ! -name 'format_sweep_*.log' \
  | wc -l                                                       #   53
```

| Family | Count | In the 8,037? |
|---|---:|---|
| `v6_*.log` | **8,037** | **counted** |
| `v2_*.log` | 30 | excluded |
| `v3_*.log` | 36 | excluded |
| `v4_*.log` | 9 | excluded |
| `v5_*.log` | 11 | excluded |
| `format_sweep_*.log` | 19 | excluded |
| residual `other` | 53 | excluded |
| **all `*.log`** | **8,195** | |

8,037 + 30 + 36 + 9 + 11 + 19 + 53 = **8,195**. The residual is computed by
subtraction-free enumeration (the last command above lists the files rather than
deducting a number), and it agrees with 8,195 - 8,142. The 53 are earlier,
differently-named waves: `long_*` 28, `ultra_*` 12, `sweep_*` 6, `mega_*` 6,
`champion_*` 1.

The directory also holds **39 non-`.log` files**: 38 `*.json` result records
(`cpu_train_*`, `igla_*`, `trinity_pr*`, `gf16_*gram*`), written by
`src/bin/cpu_train.rs` and its siblings, plus one stray `RETRACTION.md`. They
are outside the `*.log` denominator and outside every count on this page.
8,195 + 39 = 8,234 entries, and `find . -maxdepth 1 -mindepth 1 -type d | wc -l`
returns 0, so nothing in the directory is a directory and no glob here can be
inflated by one.

**Do not transcribe any of it. Re-derive it:**

```
python3 scripts/fleet_census.py
```

Observed output, this machine, 2026-08-07 (the header block and the closing
notes are elided; the counts, the population line and the shape block are
verbatim):

```
LOCAL SWEEP CENSUS (re-derived, not transcribed)
as-of:          2026-08-07 02:45:31 +0700
host:           playras-MacBook-Pro.local
glob:           /Users/playra/trios-trainer-igla/.trinity/results/v6_*.log

v6-wave log files                8037   the counted population: v6_*.log only, 8037 of 8195 *.log.
                                        NOT a fleet size.
files with a 'DONE:' line           5   RAN TO COMPLETION -- 5 of 8037. The completion count.
files containing 'val_bpb'       7274   REACHED AN EVAL -- 7274 of 8037 emitted at least one
                                        reading. NOT a completion count (that is the 5 above)
                                        and NOT a fleet size: a run killed after its first
                                        eval line is counted here.

population:     v6_*.log 8037 of 8195 *.log; excluded: v2 30, v3 36, v4 9, v5 11, format_sweep 19, other 53

filename shapes (every counted file must match one, or this script REFUSES):
  7 components, opt-terminated      165   v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}_{opt}
                                          COMMITTED: 85f33fd scripts/auto_launch.sh:72, redirect >
                                          at :84; also the template at HEAD. Truncation MEASURED.
  6 components                     7742   v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}
                                          NO COMMITTED TEMPLATE writes this shape (git log --all -S
                                          returns nothing, and the string is absent from the working
                                          tree). Truncation ASSUMED, not measured.
  4 components                      130   v6_{fmt}_seed{seed}_{step}
                                          COMMITTED: cfbe605 scripts/auto_launch.sh:42, redirect >
                                          at :54. Truncation MEASURED.
  classified                       8037   == the counted population above
                                          truncation MEASURED for 295 of 8037 counted files (4%);
                                          ASSUMED for the other 7742. That is the SAME condition
                                          docs/FLEET-COUNT.md gives for EXCLUDING v2_..v5_.
```

Each count carries its meaning **inline**, on the same line as the digits. That
is deliberate and it is the fix for the defect in the section below: the previous
version of this script printed three bare counts (`7274` among them) and left the
reader to supply the labels, so a number with no definition in any document sat
in the output looking exactly like the two that had one.

The `population:` line is not decoration. It is asserted by case 6 of
`python3 scripts/fleet_census.py --self-test` (8 cases, all of which must
report), which builds a fixture holding every one of those families and fails --
naming the families that went missing -- if the printed report omits the glob,
the denominator, or any excluded family with its count. The **shape block** is
asserted the same way by cases 7 and 8: case 7 puts a filename matching none of
the three shapes into a counted population and requires a **nonzero exit with
no count printed**, and case 8 requires the printed block to name every shape
with its count and to mark, per shape, whether its writer is committed.

### What a log file is, and is not

The file count is a **lower bound on processes launched**, and -- subject to the
decomposition below -- an **exact count of distinct configurations that produced
a log**. It is not a count of runs, because the writer of the counted family
**truncates**: the name is a pure function of the configuration and the redirect
is `> "$log"`, so re-running a configuration **replaces** its log rather than
adding one. Every relaunch after a kill -- and this sweep peaked at ~110
concurrent processes under a load average near 500 -- is invisible to the file
count.

The previous version of this page grounded that on a single citation,
`scripts/auto_launch.sh:72,84`. **That citation covers 165 of the 8,037 counted
files.** The rest is below.

#### The three filename shapes, and which of them has a committed writer

Re-derived on this tree on 2026-08-07 by grouping every counted filename on its
underscore-component count. Nothing here is transcribed from an earlier pass:

| Shape | Count | Template | mtime range (local, +0700) |
|---|---:|---|---|
| 7 components, optimizer-terminated | **165** | `v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}_{opt}` | 2026-05-26 17:11:07 -- 17:36:53 |
| 6 components | **7,742** | `v6_{fmt}_h{hidden}_lr{lr}_seed{seed}_{step}` | 2026-05-25 13:16:46 -- 2026-05-26 17:31:03 |
| 4 components | **130** | `v6_{fmt}_seed{seed}_{step}` | 2026-05-25 11:02:22 -- 12:38:07 |
| **classified** | **8,037** | | |

165 + 7,742 + 130 = **8,037**, which is the counted population exactly, out of
**8,195** `*.log` in the directory. `mtime` is last-write, not creation; for
truncated logs it is when the run stopped writing, not when it started.

Now the writers. Two commits have ever touched `scripts/auto_launch.sh`
(`git log --all --follow`), and each carries one template:

| Commit | Date (author, +0700) | Template at `scripts/auto_launch.sh` | Shape |
|---|---|---|---|
| `cfbe605` | 2026-05-25 10:37:00 | `:42` `name="${wave}_${fmt}_seed${seed}_${step}"`, redirect `> "$log"` at `:54` | 4 components |
| `85f33fd` | 2026-05-31 14:36:03 | `:72` `name="${wave}_${fmt}_h${hidden}_lr${lr}_seed${seed}_${step}_${opt}"`, redirect at `:84` | 7 components |

`85f33fd` is also the template at `HEAD`, where those line numbers still hold,
and `scripts/auto_launch.sh:35` is still the hardcoded `local wave="v6"` that is
why the census glob is `v6_*` and not something wider.

**The 6-component shape -- 7,742 files, 96% of the count -- matches no template
committed anywhere in this repository.**

```
# no commit anywhere carries the 6-component template:
git log --all -S'${wave}_${fmt}_h${hidden}_lr${lr}_seed${seed}_${step}"'

# and the working tree carries only the 7-component one.
# NOTE THE -F. Without it this grep matches NOTHING and exits 1, because
# ugrep (and GNU grep) read the unescaped {} as an interval operator. Exit 1
# from a broken pattern is indistinguishable from exit 1 meaning "no such
# writer exists" -- the false negative was hit while writing this section,
# and it would have "confirmed" the paragraph below for the wrong reason.
grep -rnF 'lr${lr}_seed${seed}_${step}' scripts/ src/ .github/
#   scripts/auto_launch.sh:72:  local name="${wave}_${fmt}_h${hidden}_lr${lr}_seed${seed}_${step}_${opt}"
#   scripts/fleet_census.py:48:  (this census quoting that same template)
```

`git log -S` is a literal-string pickaxe, not a regex, so it is not subject to
the brace problem. On the 6-component template it returns nothing across all
refs, and the fixed-string grep shows the string is absent from the **working
tree** as well, tracked or untracked. The searches that do return something
return only the two templates in the table above.

**So: the truncation behaviour of the writer that produced the 7,742-file
majority is ASSUMED, not measured, because no such writer is committed.** That
is the identical condition this page gives, below, as its reason for
**excluding** the `v2_`..`v5_`, `long_`, `ultra_`, `mega_`, `sweep_` and
`champion_` families -- "no writer is committed in this tree", therefore
"unknown, not assumed". The exclusion rule was applied to those **139** files
(30 + 36 + 9 + 11 + 53, i.e. everything excluded except the 19
`format_sweep_*`, whose writer *is* committed) and not to the 7,742 inside the
count that meet the same condition. Stating that is the correction; the count
itself does not move.

Two further facts that a reader should have rather than reconstruct:

- **The 7-component files predate the commit of the template they match.** Their
  mtimes run 2026-05-26 17:11:07 -- 17:36:53; `85f33fd` was authored
  **2026-05-31 14:36:03**, five days later. The template was in the working tree
  before it was in a commit. This is not a discrepancy in the count -- the shape
  and the template agree exactly -- but "matches a committed template" is a
  weaker statement than "was written by a committed template", and only the
  first is established. The 4-component group is the other way round: mtimes
  2026-05-25 11:02:22 -- 12:38:07 against `cfbe605` at **2026-05-25 10:37:00**,
  25 minutes earlier, which is consistent with that commit's script having
  written them.
- **The 6- and 7-component groups overlap in time**, 17:11:07 to 17:31:03 on
  2026-05-26. Whatever wrote the majority shape was still writing while the
  7-component shape appeared, so the two are not cleanly sequential waves.

#### The exactness claim: measured, not argued

The exactness half -- "an exact count of distinct configurations" -- needs the
filename to be **injective** over configurations. It is not injective by
construction here: two of the three shapes drop fields. The 6-component shape
drops `${opt}`; the 4-component shape drops `${opt}`, `h${hidden}` and
`lr${lr}`. Two runs differing only in a dropped field would land on one
filename and one would overwrite the other, and the count would silently be
short.

An earlier version of this page settled that by reasoning. **This one measures
it.** Every `v6_*.log` opens with the trainer's own line

```
[trios-train] parsed seed=43 steps=10000000000 hidden=256 lr=0.0003 ctx=None optimizer=adamw neon=None
```

so the dropped fields are recoverable from the file's contents. Read out of all
8,037 logs on 2026-08-07:

| Shape | Files | `optimizer` recorded | `hidden` | `lr` |
|---|---:|---|---|---|
| 7 components | 165 | `adamw` 55, `muon` 55, `muon-cwd` 55 | in the name | in the name |
| 6 components | 7,742 | **`adamw` 7,742 -- one value** | in the name | in the name |
| 4 components | 130 | **`adamw` 130 -- one value** | **`384` 130** | **`0.003` 130** |

Every file carried the line; none was missing it. **Every field that a shape
drops from its filename was constant across every surviving log of that shape.**
The 7-component shape, which drops nothing, is the only one where the optimizer
varies -- and it varies exactly three ways, which is why that shape exists at
all.

**The limit of that measurement, stated rather than glossed:** an overwrite
destroys its own evidence, so this cannot by itself exclude a collision in which
the `adamw` run happened to write last. For the 7,742 six-component files that
would have to have happened 7,742 times out of 7,742, with no survivor left
recording anything else. What is established is the observable half: not one
surviving log of either shorter shape records a value its filename could not
have carried. That is what a sweep which never varied the dropped fields looks
like; it is not what a sweep which varied them looks like, and the 7-component
shape is on the same page to show the contrast.

The exactness claim therefore **survives on that measurement, and not on the
argument this page previously made**. The distinction matters: the argument was
about a template, and for 96% of the population there is no template to argue
from.

The same pass checked the other direction. For each of the 8,037 files, the
configuration recorded inside the log was reassembled into a filename under that
file's own shape and compared to the actual filename: **0 mismatches in 8,037**.

This is now enforced rather than narrated. `scripts/fleet_census.py` classifies
every counted file into one of the three shapes and **refuses -- nonzero exit,
no count printed -- if any counted file matches none of them**, naming the file.
A new shape appearing in that directory can no longer be absorbed silently into
a total whose published meaning is not about it.

**The truncation argument covers `auto_launch.sh` and nothing else.**
`scripts/format_sweep.sh:44` also truncates -- `tee ".trinity/results/format_sweep_${fmt}_seed${SEED}.log"`
-- but it writes the `format_sweep_*` family, which the 8,037 **excludes**. An
earlier version of this page and of the script cited that `tee` as part of the
justification for reading 8,037 as an exact configuration-slot count, which
argued for a population the number omits. It is recorded here as a property of
an excluded family, not as evidence about the counted one.

For the remaining excluded families -- `v2_`..`v5_`, `long_`, `ultra_`,
`mega_`, `sweep_`, `champion_`, **139 files** -- **no writer is committed in
this tree**: `grep -rln 'trinity/results' scripts/` names only
`auto_launch.sh`, `format_sweep.sh`, two monitors that read (`monitor_all.sh`,
`monitor_v2_sweep.sh`) and `fleet_census.py`. Whether those families truncate is
**unknown, not assumed**, which is the second reason the glob is not widened to
`*.log`.

That reason is now weaker than it reads, and the honest form of it is this: the
counted population **already contains** 7,742 files in the same condition, so
the glob is not the line between "writer known" and "writer unknown" and never
was. What it is instead is the line between one wave and several, and the shape
block above is what carries the writer question across it -- per shape, with the
counts, in the script's own output. Widening the glob is now refused outright
rather than answered: run on 2026-08-07,

```
python3 scripts/fleet_census.py --glob '*.log'
```

exits **3**, prints no count, and reports "158 counted file(s) match NONE of the
3 known filename shapes", naming them.

**5 DONE out of 8,037 v6-wave logs is the load-bearing number here**, and it is
unchanged from the original report. 7,274 files carry a `val_bpb` line, so most runs
produced *some* reading; only 5 ran to completion. That is the fact that
disqualifies the v6 sweep as a format comparison, and it does not depend on
which of the three figures is the fleet size.

### The third row: what 7,274 counts, and what it does not

The census prints three counts, and only two of them have ever been explained.
The third -- **7,274** -- had no definition anywhere in this programme's
documents, which is how an unlabelled number becomes a fourth fleet size.

**7,274 is the number of the 8,037 `v6_*.log` files that contain at least one
`val_bpb` line: logs whose run reached an eval.** It is a per-FILE count -- a log
with two hundred eval lines is counted once -- and the test is a substring match
for `val_bpb` anywhere in the file, so it fires on the initialisation reading
that `cpu_train` prints before the first optimizer step as readily as on a
converged one.

**It is not a completion count.** Completion is the line-initial `DONE:` test,
and that count is **5 of 8,037**, unchanged since the original report. The gap --
7,269 logs that produced a reading and never finished -- is the sweep's actual
shape: 110 concurrent processes on 8 cores at a load average near 500, most of
them killed long before their step budget ran out. A log holding one `val_bpb`
line establishes that a forward pass emitted a number, and nothing whatsoever
about convergence.

**It is not a fleet size either**, for the same two reasons 8,037 is not: it is a
count of files on one gitignored directory on one workstation, and its writer
truncates, so it is a count of distinct *configurations that reached an eval* and
a lower bound on launches that did.

**And it is not a superset relation anyone should assume.** All 5 `DONE:` logs
also carry a `val_bpb` line in the current population, but that is an observation
about these files, not a property of the format: nothing in `auto_launch.sh`
requires a completed run to have printed the string, and the two tests are
independent scans. Read the three rows as three separate measurements of the same
8,037 files, which is what they are, and which is why the script now prints each
one with its meaning attached rather than leaving the reader to supply it.

### This population exists on one machine

`.trinity/results/` is gitignored (`.gitignore:54`). The 8,037 logs are not in
any clone and cannot be recovered from one. A checkout on another machine will
get a refusal from `fleet_census.py`, which is correct behaviour: it has nothing
to count and must not answer "0".

### A measured defect in the obvious way of counting this

The shell one-liner `grep -l 'val_bpb' .trinity/results/v6_*.log | wc -l` returns
**7,271**, three short of the true 7,274. Three real logs --
`v6_f32_seed44_20000.log`, `v6_fp16_seed49_20000.log`, `v6_gf8_seed46_20000.log`
-- contain NUL bytes from an interrupted write (213 NULs in the first). macOS
`grep -l` classifies them as binary and **skips them silently**: exit 1, no
output, no warning. `grep -al` finds 3 matches in each.

This was found by two of our own methods disagreeing, and it is the reason the
census is a script with a self-test rather than a one-liner. The undercount is
small and completely invisible -- a failure that reports success. The self-test
now carries a NUL-byte fixture as a regression guard.

---

## 1,851 -- WITHDRAWN

**No derivation of this figure exists in any source examined.**

Its only primary occurrence is
`/Users/playra/skills/_state/backup-2026-08-03-dead-pointers/memory/trios-igla-checkpoint-chain.md:11`,
which asserts flatly that the repository "ran 1,851 experiments for OpenAI
Parameter Golf (PR #2003, openai/parameter-golf)" and states no method, no
query, no ledger and no date.

Its upstream is named at `P0_scarab_corpus_catastrophe.md:86`, which lists the
row `PR #2003 "1851 experiments fleet"` among claims invalidated by the corpus
audit. **The source of the number is therefore the TEXT of a submission PR** --
a figure written into a narrative, not read off an instrument.

That is sufficient to withdraw it. A number whose only provenance is a sentence
in a PR body cannot be cited as a fleet size, whatever it happens to equal.

**Do not assert a cause.** There is an available hypothesis -- that `1851` is a
transcription of something else, a PR number among them -- and it is consistent
with every observation made here. **It was not established, and it must not be
written down as if it were.** "No derivation exists" is a finding. "It came from
a typo" is a guess, and this programme's cost of confident unmeasured sentences
is precisely what it is selling against.

---

## What to say when asked "how many experiments?"

> 1,878 runs on the Railway fleet, audited 2026-05-02 against the
> `strategy_queue` table, decomposing as 1,875 untagged + 3 tagged + 0 fineweb.
> That instance is no longer reachable and no dump is committed, so the figure
> rests on a contemporaneous audit by a stated method, not on a re-derivation.
> Separately, the v6 sweep on one workstation left 8,037 `v6_*.log` files -- out
> of 8,195 `*.log` in that directory, the rest being earlier waves -- of which 5
> completed. That is a file count of one wave on one machine, not a fleet size,
> and it is re-derived, with that denominator and every excluded family, by
> `scripts/fleet_census.py` on every read. Those 8,037 have three filename
> shapes, and only 295 of them match a template committed anywhere, so for the
> other 7,742 the "one file per configuration" reading rests on an assumed
> writer -- measured, though, is that no two configurations could have shared a
> filename, because every field the shorter shapes drop was constant. A third
> figure, 1,851, circulated in a submission PR and is withdrawn: no derivation
> of it exists.

Both halves of that answer name their method and their limits. Neither requires
the word "unreconciled".

---

## Follow-up, out of scope for this pass

**`1,851` is still restated in Rust sources**, which were owned by another agent
during this pass and were deliberately not touched. The site list below was
**re-run on 2026-08-06, not copied**; the previous version of this section listed
`src/train_loop.rs:174,201,922,2576,2832,3148,5424` and omitted `src/race/neon.rs`
entirely, so every line number in it was wrong and one file was missing. Line
numbers in a live file go stale by construction -- **the command is the
authority, not this list**:

```
grep -rn '1,851\|1851' src/ tests/
```

```
src/train_loop.rs:310:/// was 1,851 experiments and no artifact. `TRIOS_CHECKPOINT_DISABLE=1` is the
src/train_loop.rs:337:/// substitution as "1,851 experiments, zero artifacts", one layer up.
src/train_loop.rs:1079:/// substitution as "1,851 experiments, zero artifacts", one knob over from
src/train_loop.rs:2784:    // exited 0 - the exact shape of "1,851 experiments, zero artifacts".
src/train_loop.rs:3093:        // "1,851 experiments, zero artifacts".
src/train_loop.rs:3409:    // exited 0 - the exact shape of "1,851 experiments, zero artifacts".
src/train_loop.rs:5685:    /// "1,851 experiments, zero artifacts".
src/race/neon.rs:6://! shape as the `checkpoint::save` stub that returned `Ok(())` for 1,851
tests/observation_parameter_independence.rs:6://! part of the report. The flagship case was found by accident after 1,851
tests/observation_parameter_independence.rs:408:/// 1,851 experiments and zero artifacts -- every case in this file would pass
```

Ten sites, in three files. **Every one of them is prose describing a failure
SHAPE, not a count being asserted as data** -- "the exact shape of '1,851
experiments, zero artifacts'" is a name for the defect that `checkpoint::save`
was a stub returning `Ok(())` for an entire campaign, so every equality
assertion in the harness passed while proving nothing. No control flow, no
assertion and no output depends on the number. A reviewer who greps for a
withdrawn figure will hit these ten lines, and should not read them as ten
surviving claims.

They should still be corrected the same way `docs/OBSERVATION-INDEPENDENCE.md`
was: the anecdote is intact and load-bearing **without a run count attached to
it**. Keep the anecdote, drop the figure.

The companion assertion this section used to make -- that
`grep -rn '1,851\|1851' docs/ IGLA_V6_FINAL_RESULTS.md` returns "only this
file's withdrawal section" -- **was also false, and is withdrawn**. Re-run on
2026-08-06 it returns, besides this file, `docs/EVAL-UNCERTAINTY.md:818`,
`docs/OBSERVATION-INDEPENDENCE.md:103` and `IGLA_V6_FINAL_RESULTS.md:53`. All
three are withdrawal notices, which is the correct content -- but "returns only
this file" was a statement about a command's output that nobody had run, which
is the defect class this repository exists to remove, appearing at the bottom of
the page that removes it.
