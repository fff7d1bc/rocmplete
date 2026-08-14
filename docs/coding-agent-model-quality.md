# Coding-agent model quality baseline

This record interprets the current coding-agent acceptance results with solution
quality as the first concern and runtime cost as the second. Token speed, model
size, and successful raw tool calls do not establish coding quality. Exact
artifacts, measurements, runtime conditions, and retained result paths remain
in the
[same-host hardware record](hardware-acceptance.md#coding-agent-comparison-2026-08-11).

## Evidence boundary

The shared easy screen used `rocmplete-coding-v4`, fingerprint
`8825f9235c854fdf693c4881faa035c5efe99a545f4111372ca95e6f2def1160`,
Pi 0.84.1, ROCm, high thinking, 131072 context, one fresh fixture, and one
repetition on the Fedora 44 Strix Halo host. A screen pass means the model
completed the easy `re-align` implementation with ordinary tests, hidden
tests, and the build passing, without a dependency change, retained build
artifact, or network attempt. That version 4 screen remains the direct
same-task comparison across all six models.

Passing one easy task establishes basic coding-agent competence. It does not
establish safety, review accuracy, or performance on hard changes. Models are
not treated as equivalent merely because their grading result says `solved`.
The structure of the retained patch, convergence behavior, and broader suite
evidence also matter.

The current frozen definition is `rocmplete-coding-v5`, fingerprint
`9da456c1820080d032896fe0e69fafbf3722addc39008068ba62daff84b5aad7`.
Version 5 retains the version 4 tasks and adds ssh-host-proxy late-connection
ownership, ROCmplete SELinux verification ordering, and nonet process
lifecycle work. Qwen3.6 35B-A3B MTP completed the full version 5 suite. The
other five screened models then received the hard `re-source-race` promotion
gate with a 20-minute practical ceiling. A remote SIGINT preserved an
interrupted checkpoint and let ROCmplete remove the server cleanly.

To distinguish practical completion from eventual capability, Qwen3.6 27B
MTP and Muse Glimmer 30B DFlash later received one additional `re-source-race`
attempt with a 60-minute outer ceiling. The suite fingerprint, Pi version,
model presets, 131072 context, and grading contract were unchanged. These
single attempts are capability probes, not replacements for the 20-minute
promotion gate.

Those runs calibrate the policy for future candidates at a 45-minute ceiling
per attempt. This does not retroactively change the recorded 20-minute
practical screen or the two 60-minute capability probes.

## Current reasoning-default calibration

The 2026-08-14 native-control follow-up compared the three retained practical
models at 256K through Pi on the same Strix Halo host. Qwen3.6 used its binary
thinking-on mode, Qwen3.8 used medium effort, and Muse used high strength. The
exact conditions and measurements are in the
[same-host reasoning-default record](hardware-acceptance.md#reasoning-default-calibration-2026-08-14).
These are operational defaults, not a claim that the models received equal
reasoning effort.

All three solved the version 5 `re-align` task. Qwen3.8 medium was the best
attempt in this narrow screen: it finished in 609.8 seconds with 16 tool calls
and 7,789 output tokens, and its patch used one named width constant for the
header and rows. Qwen3.6 on took 636.5 seconds, 21 calls, and 8,554 output
tokens. Muse high generated fastest but took 768.0 seconds, 49 calls, and
13,559 output tokens; it spent substantial work rechecking the same formatting
and test details before producing the same minimal literal patch as Qwen3.6.

This single easy repetition is enough to validate the three current paths and
to reject raw token speed as a ranking proxy. It is not enough to replace the
broader quality groups below or to change the user-selected Muse default.
Before changing reasoning defaults, compare Qwen3.8 low with medium and Muse
medium with high on the same frozen task, then carry the winner into the hard
promotion gate. The official managed Qwen3.8 template showed no matching or
tool-continuation failure here, so the broader Froggeric community template is
not the next variable to introduce.

## Focused Muse M, Muse XL, and Qwen 27B comparison

A 2026-08-13 follow-up separates the two currently managed Muse targets and
compares them with Qwen3.6 27B MTP. The controlled runs used source commit
`5d170046250616cc88f67fce09177fe39c2a3afb`, the version 5 suite fingerprint,
Pi 0.84.1, ROCm, high thinking, 131072 context, one fresh fixture, and image
`localhost/rocmplete:llama-cpp-ubuntu26.04-rocm7.14-62bf73d-r19` with image ID
`d4b7065b465a85efbfc5ff0aa10895283bdc5e79b2aae5b528f9f1b6e9647147`.
The Muse presets used their accepted ROCm DFlash depth 15. Each result is one
stochastic attempt, so this is role-positioning evidence rather than a new
general ranking.

The read-only evidence splits by task. On the frozen fzr concurrency review,
Qwen was the only model to explain correctly that an older accepted entry
snapshot can be applied and followed by a refresh. It still missed that Enter
can select immediately from the currently visible older snapshot while that
refresh is pending. Muse M described the version behavior in one section but
contradicted it elsewhere. Muse XL produced the cleanest answer but made the
material error that an older result is discarded. Qwen therefore gave the
best analysis of this particularly subtle mechanism, while none was fully
trustworthy. The attempts took 391.6, 428.4, and 369.9 seconds respectively
for Qwen, M, and XL.

A naturalistic archaeology question against an older private checkout asked
the models to explain a nontrivial data path end to end. Muse XL's 949-word
answer was the best-balanced explanation, grounded in ten tool calls. Muse M
was equally grounded and the most exhaustive at 940 words and twelve calls,
but was less economical. Qwen used ten calls and gave a concise, usefully broad
653-word answer, but made one material error about a size-selection threshold.
The interactive Muse runs used the 256K variants and were not frozen evaluation
attempts, so this observation supports a human-guided code-reading preference,
not a strict score.

The implementation evidence reverses the Muse order. On medium-difficulty
`re-cancel`, both Muse variants passed ordinary tests, hidden tests, the build,
artifact checks, and network isolation. Muse M solved it in 890.2 seconds and
75 tool calls; Muse XL solved it in 1,014.4 seconds and 67 calls. Qwen's source
patch passed every functional gate in 1,028.1 seconds and 38 calls, but its
own `go build ./...` left a forbidden `reencode` executable, making the strict
result unsolved. Qwen also consumed much more live context: 89,659 tokens,
versus 57,702 for M and 66,026 for XL.

No candidate solved hard `re-source-race`. Muse M exited in 1,429.3 seconds,
Muse XL in 1,463.2, and Qwen in 1,527.6; ordinary tests and builds passed, but
the hidden suite rejected all three helper contracts. M's candidate was the
best of the two Muse implementations: it threaded per-input snapshots and
used quarantine with deferred restoration, although restoration could still
replace a new occupant and the helper API was incompatible. XL introduced
global mutable snapshot state and explicitly removed an occupant before
restoration, which violated the no-replace requirement. Qwen carefully traced
the error paths but then independently chose incompatible snapshot,
verification, and quarantine APIs. Peak live context was 69,686 for M, 66,640
for XL, and 97,818 for Qwen. More wall time did not turn any candidate into a
safe solution.

The practical positioning from this evidence is:

1. **Human-guided unfamiliar-code archaeology: Muse XL.** It gave the best
   naturalistic explanation and reached a polished answer economically. Ask
   it to cite exact files and challenge concurrency or boundary claims because
   the frozen review exposed a confident factual error.
2. **Long autonomous implementation among these three: Muse M, provisionally.**
   It owns the quickest strict medium-task solve, while Qwen's otherwise-
   passing attempt failed workspace hygiene, and its hard-task design was
   safer than XL's. It is deliberate and tool-heavy, and one medium success
   does not make it a trusted unattended engineer.
3. **Broad second opinion and difficult control-flow analysis: Qwen3.6 27B.**
   It was strongest on the subtle frozen review and often used fewer tool
   calls, but consumed substantially more context, made one archaeology error,
   failed medium-task workspace hygiene, and did not solve the hard task.

This does not establish that either Muse quantization is globally more capable
than Qwen. It establishes a useful role split under Pi on this host. If a task
may run overnight, staged verification still matters more than the extra wall
time: require cited analysis, run the repository tests, inspect the working
tree for artifacts, and review destructive or concurrent paths independently.

## Current quality groups

These groups put demonstrated repository-level quality before token speed.
They do not turn a hidden-test failure into a success merely because the patch
looked plausible or the model exited normally.

1. **Demonstrated baseline: Qwen3.6 35B-A3B MTP.** It is the only model with a
   complete version 5 run, remains the fastest, and solved 3/9 implementations.
   Six implementation misses and two flawed reviews mean “baseline,” not
   “trusted autonomous engineer.”
2. **Completed but failed the hard gate: Gemma4 31B MTP, Ornith 1.0 35B, and
   KAT-Coder v2.5 Dev.** All three produced coherent snapshot-oriented patches,
   passed the repository tests, and failed the withheld source-race contract.
   Gemma left the cleanest fixture and used the fewest tool calls. Ornith and
   KAT also retained generated executables. None qualifies for a full-suite
   promotion.
3. **Correct easy patch, slow and incorrect hard-task completion: the then-
   managed Muse Glimmer dynamic DFlash target and Qwen3.6 27B MTP.** Neither
   completed the practical gate within 20 minutes. Both later completed a
   candidate inside a 60-minute capability probe, passed ordinary tests and
   the build, and failed the withheld safety contract. Qwen 27B still owns the
   best-designed easy patch, but that task-local result cannot support an
   overall first-place claim.

Laguna XS, Laguna S, and DwarfStar DeepSeek V4 Flash are outside this order.
They did not complete the bounded autonomous coding task in the tested
configuration.

## Model assessments

These sections preserve the evidence for each model; their order is not a
second ranking separate from the quality groups above.

### Qwen3.6 27B MTP

**Verdict:** strongest easy-task patch, but too slow to complete the hard
promotion gate within the practical ceiling.

The `qwen3.6-27b-mtp-q8-0` result introduced one named width-10 constant and
used it for the header and every rendered row. That was the clearest expression
of the requested shared formatting policy. It removed the opportunity for the
two paths to drift and included a comment explaining the constraint. Ordinary
tests, hidden tests, and the build all passed.

The version 5 source-race gate confirmed that limitation. Qwen inspected the
repository broadly, began a substantial file-identity patch after roughly
13.5 minutes, and was still wiring the change when the 20-minute ceiling sent
SIGINT. It had made 31 tool calls and had not reached tests. The checkpoint is
therefore `interrupted`, not a hidden-test result. This is a practical failure
for bounded autonomous work even though the partial design was sensible.

The extended capability probe exited normally after 1,674.5 seconds and 49
tool calls. Ordinary tests and the build passed, but the hidden suite did not
compile against the patch: the required `snapshotSource` helper was absent,
and the verification and quarantine helpers had incompatible contracts. Live
context peaked at 98,943 tokens without compaction. More time let Qwen produce
a coherent candidate, but did not make it solve the task, and the run was not
limited by the 131072-token context.

The run took 766.7 seconds, 3.34 times the Qwen 35B reference time. It used 22
tool calls, generated 8,699 tokens at 15.57 tokens per second, and processed
prompt tokens at 48.43 tokens per second. It was the second-slowest successful
screen, ahead of Gemma only. Choose it when the observed patch quality matters
more than turnaround time, then review the result as unproven on harder work.

### Qwen3.6 35B-A3B MTP

**Verdict:** best-supported and fastest current baseline, with a low full-suite
solve rate and material safety limitations.

The `qwen3.6-35b-a3b-mtp-ud-q8-k-xl` result is now backed by the complete
version 5 suite. It solved formatting, EINTR handling, and symlink behavior,
for 3/9 implementations. The run completed in 52 minutes 28 seconds elapsed,
with 3,105.2 seconds of agent wall time, 286 tool calls, and 114,087 generated
tokens. Its three successes are unchanged from version 4.

The six misses were bounded completions rather than hangs. `re-cancel` passed
all tests but retained a generated executable. Both inherited hard tasks
failed their hidden contracts. The new proxy task failed to close a successful
connection that returned after timeout. The SELinux attempt invoked real
recursive `chcon` work at the wrong boundary, broke 14 ordinary tests, and
missed the required per-file no-dereference command. The nonet attempt passed
ordinary tests but changed a helper contract incompatibly with the hidden
lifecycle test.

Both review answers were detailed but materially inaccurate. The reencode
review placed core functions in the wrong file and overstated cancellation
safety after output validation. The fzr review said older entry snapshots are
discarded, although the picker deliberately accepts one and schedules a
follow-up, then gave an internally inconsistent explanation of Enter while
that stale refresh is active.

On `re-align`, it used matching width-11 literals for the header and rows. The
patch was functionally correct and passed all grading, but the longest relevant
value only requires width 10. The extra column was unnecessary and the shared
policy remained duplicated. This was weaker than Qwen 27B's bounded design,
even though the model has much stronger evidence across the suite.

The comparable `re-align` attempt took 229.9 seconds with 23 tool calls. It
generated 12,596 tokens at 70.03 tokens per second and processed prompt tokens
at 111.30 tokens per second. It is the practical baseline when predictable
runtime and known limitations matter. Its 3/6 suite result was the version 4
threshold. The current threshold is the 3/9 version 5 result, not proof that
the model is safe without review.

### KAT-Coder v2.5 Dev

**Verdict:** correct and efficient on the easy task, but expensive and
incorrect on the hard safety gate.

The `kat-coder-v2.5-dev-q8-0` patch made the required width-10 changes to the
header, rows, and expectations. It was minimal and all grading passed. Its only
design weakness relative to Qwen 27B was representing the shared width as two
matching literals instead of one named policy.

The easy run took 319.3 seconds, 1.39 times the Qwen 35B reference, with 25
tool calls. On `re-source-race`, KAT spent a long initial period investigating,
then produced a snapshot-oriented patch and exited after 1,139.7 seconds and
90 tool calls. Ordinary tests and the build passed, but the hidden contract did
not compile against its incompatible helper design, and KAT left a generated
`reencode` executable. This is a completed 0/1 safety-gate result, not a reason
to spend a full-suite run.

### Muse Glimmer 30B dynamic DFlash (current XL target)

**Verdict:** capable of a correct minimal patch, but unable to finish the hard
gate inside the practical autonomous-work budget.

The historical `muse-glimmer-30b-kquant-dynamic-dflash` result used the tensor-
equivalent target now exposed as
`muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash`. It made the same width-10
header, row, and expectation changes as KAT. Ordinary tests, hidden tests, and
the build passed. The retained patch itself gives no quality reason to place
Muse below KAT.

The operational behavior does. Muse took 670.2 seconds on the easy task and
made 40 tool calls. On the source-race gate it made 67 tool calls, built a
substantial patch, and reached ordinary tests, but did not exit before the
20-minute ceiling. Earlier testing also showed substantial harness
sensitivity. Use Muse where its behavior has been accepted with the chosen
harness, and keep a human close enough to notice loss of convergence.

The extended capability probe exited normally after 1,929.2 seconds and 82
tool calls. Ordinary tests and the build passed. The hidden suite still did
not compile because `snapshotSource` was absent and `quarantineSource` had an
incompatible signature. Live context peaked at 80,558 tokens without
compaction. Muse can eventually turn this prompt into a plausible tested
patch, but it did not demonstrate the required safety behavior, and more
context would not have addressed the observed failure.

### Gemma4 31B MTP

**Verdict:** cleanest completed challenger on the hard gate, but still a hidden
safety failure and too slow for routine first choice.

The `gemma4-31b-it-q8-0-mtp` patch was the same correct width-10 implementation
as KAT and Muse. It changed the header, rows, and expectations without adding
unrelated work, and all grading passed. Its patch quality therefore belongs in
the same task-local group as those two models.

The easy run took 902.5 seconds, making it the slowest completed screen. On the
source-race gate, Gemma produced a coherent snapshot design, passed ordinary
tests and the build, left no generated artifact, and exited after 1,000.2
seconds with only 27 tool calls. Its hidden contract still failed to compile
because the required snapshot and quarantine interfaces were absent. This is
the best operational failure among the challengers, not a safety pass.

### Ornith 1.0 35B

**Verdict:** functional alternate that converged on the hard task but failed
its withheld safety contract.

The `ornith-1.0-35b-q8-0` result used matching width-11 literals, just like
Qwen 35B. It passed ordinary tests, hidden tests, and the build, but used one
more column than necessary and left the policy duplicated. This is a small
design issue rather than a correctness failure. It is enough to rank Ornith
below the models that produced the exact minimal behavior on the only shared
task.

The easy run took 484.2 seconds, 2.11 times the Qwen 35B reference, with 18
tool calls. On the hard gate Ornith reached repeated full test runs and exited
after 1,000.6 seconds and 54 tool calls. Ordinary tests passed, but its hidden
contract did not compile and it retained a generated executable. Ornith is a
reasonable second opinion, not a safe autonomous choice.

## Not promoted under the current configuration

### Laguna XS 2.1

`laguna-xs-2.1-q4-k-m` was interrupted after more than 21 minutes without
making an edit. It could use the raw tool protocol and answer short prompts,
but it did not converge on the version 4 easy coding screen. For an autonomous
agent, failure to turn valid tool use into a completed change is a quality
failure. Keep it to short-prompt and runtime experiments unless a material
model, template, runtime, or harness change justifies retesting.

### Laguna S 2.1

The older version 2 screen and the first version 5 `re-align` follow-up did not
preserve Laguna's interleaved reasoning. The preset also advertised no
reasoning-effort support, so Pi's requested `high` level was ignored. The
server explicitly warned that the template supported reasoning preservation.
Treat the resulting 27-minute and 45-minute no-edit runs as useful controls,
not the final Laguna quality result.

Commit `a2256eb` corrected both boundaries. The server now preserves reasoning
between tool turns, while managed agent clients expose the project's bounded
low, medium, and high levels. A first corrected trial reached the right
width-10 implementation and was beginning an edit tool call when it was
manually interrupted after about 40 minutes. That operator-interrupted trial
is evidence of capability but is not a pass.

A fresh corrected trial then received the full 45-minute ceiling. Laguna used
11 completed read or shell calls, exhausted one high-effort turn at 8,329
decoded tokens, and repeatedly derived the correct width-10 policy. It still
never edited the worktree. The retained reasoning grew the live history to
about 29K tokens; generation fell from 16.53 tokens per second on the first
turn to 5.65 on the bounded turn and 3.93 near timeout. The final 3,906-token
reasoning turn was still incomplete when the normal timeout checkpointed the
attempt. Its result is
`apps/agent-evaluation/results/v5-45m-laguna-s-preserved-re-align-r2.json`.
The hard gate was not started.

The integration is now behaving as intended, but Laguna S is not promoted as
a Pi repository agent on this host. Faster hardware could convert the
near-edit trajectory into a pass and would make retained context cheaper; it
would not remove the model's excessive reasoning-token use. The official
hybrid Q4_K_M quantization may contribute, but this test does not isolate
quantization and the official Q8_0 artifact does not fit this 128 GiB host
with runtime state. Retest after materially faster hardware, a practical
higher-precision control, or a model/template/harness change. A bounded
medium-effort run is also a useful follow-up now that the selector is wired.

### DwarfStar DeepSeek V4 Flash

DwarfStar's then-public `deepseek-v4-flash` identity derived the likely fix in
the older version 2 screen, then entered a repetitive loop and was interrupted.
The runtime and Pi tool protocol worked. The failure was the model's inability
to turn its reasoning into a finished patch. That loop disqualifies it from
dependable autonomous coding in the current configuration, while leaving it
useful for DeepSeek runtime research and tightly bounded prompting.

## Decision rule for a new model

A newly integrated Qwen or other candidate should progress through:

```bash
timeout --foreground --signal=INT --kill-after=90s 45m \
  ./rocmplete benchmark agent --preset NEW_PRESET --task re-align
timeout --foreground --signal=INT --kill-after=90s 45m \
  ./rocmplete benchmark agent --preset NEW_PRESET --task re-source-race
./rocmplete benchmark agent --preset NEW_PRESET
```

The 45-minute ceiling applies to each active attempt, including an attempt in
the complete-suite promotion. It does not apply once to the aggregate suite.
A clear loop may be terminated earlier. A model that reaches the ceiling while
still making forward progress does not earn promotion on the tested hardware,
but remains an explicit retest candidate rather than being classified as
incapable.

Use these quality gates in order:

1. It must converge and solve the easy screen. Speed is secondary at this
   stage.
2. It must complete the destructive source-race safety task within 45 minutes
   to earn promotion. A loop, plausible explanation without a patch, or
   hidden-test failure rejects the candidate under the tested configuration. A
   progressing timeout stops promotion but is recorded as performance-bounded
   and inconclusive, with a retest trigger tied to faster hardware or runtime.
3. It must complete the frozen suite. Compare implementation solve rate and
   the nature of failures, not only the aggregate score.
4. Human-grade both review answers for factual correctness. A polished answer
   containing a material concurrency or lifecycle error does not pass.
5. Repeat a finalist three times before changing the recommended default.

To be a demonstrated quality improvement over the current evidence, a new
model should retain a clean shared-policy implementation on the easy screen,
solve `re-source-race` inside the declared practical ceiling, exceed the 3/9
version 5 implementation baseline, avoid the inherited hard safety failures,
and produce factually sound reviews. Report the three new tasks separately.
Merely beating the 152.6-second version 5 easy-task time would make the model
faster, not better. Conversely, exceeding the ceiling while still progressing
makes a model impractical on the tested host, not necessarily incapable on a
future host.
