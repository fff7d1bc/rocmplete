# Coding-agent model quality baseline

This record interprets the current coding-agent acceptance results with solution
quality as the first concern and runtime cost as the second. Token speed, model
size, and successful raw tool calls do not establish coding quality. Exact
artifacts, measurements, runtime conditions, and retained result paths remain
in the
[same-host hardware record](hardware-acceptance.md#coding-agent-comparison-2026-08-11).

## Evidence boundary

The current comparable screen used `rocmplete-coding-v4`, fingerprint
`8825f9235c854fdf693c4881faa035c5efe99a545f4111372ca95e6f2def1160`,
Pi 0.84.1, ROCm, high thinking, 131072 context, one fresh fixture, and one
repetition on the Fedora 44 Strix Halo host. A screen pass means the model
completed the easy `re-align` implementation with ordinary tests, hidden
tests, and the build passing, without a dependency change, retained build
artifact, or network attempt.

Passing one easy task establishes basic coding-agent competence. It does not
establish safety, review accuracy, or performance on hard changes. Models are
not treated as equivalent merely because their grading result says `solved`.
The structure of the retained patch, convergence behavior, and broader suite
evidence also matter.

The current frozen definition is version 5. It retains these tasks and adds
ssh-host-proxy late-connection ownership, ROCmplete SELinux verification
ordering, and nonet process-lifecycle work. No model has version 5 acceptance
evidence in this record yet, so the ranking below remains explicitly version 4
evidence rather than silently mixing suite fingerprints.

## Provisional quality order

This is the most useful current order for choosing a coding model. It is
provisional because only Qwen 35B has completed the full suite and the other
placements rely on one shared easy task.

1. **Qwen3.6 27B MTP** produced the best-designed bounded patch. It expressed
   the requested alignment as one named policy used by both output paths. It
   was also the second-slowest completed screen, so this is a quality-first
   placement rather than a speed ranking.
2. **Qwen3.6 35B-A3B MTP** has the broadest and therefore most trustworthy
   evidence. It was the fastest completed screen and solved three of six full
   suite implementations, but real hard-safety failures and one materially
   wrong review prevent calling it the quality winner.
3. **KAT-Coder v2.5 Dev** produced a correct minimal patch at a much lower
   cost than Qwen 27B. It is the strongest next candidate for a full-suite run,
   but its broader quality is still unknown.
4. **Muse Glimmer 30B DFlash** produced the same correct minimal change as
   KAT. Repeated probing, 40 tool calls, and known harness sensitivity make it
   less dependable as an autonomous default.
5. **Gemma4 31B MTP** also produced the same correct minimal change. It was
   the slowest completed screen, and there is no broader evidence to justify
   paying that cost for normal coding work yet.
6. **Ornith 1.0 35B** completed the task correctly, but its patch allocated an
   unnecessary extra column and duplicated the policy as literals. It remains
   a viable alternate, not a demonstrated leader.

Laguna XS, Laguna S, and DwarfStar DeepSeek V4 Flash are outside this order.
They did not complete the bounded autonomous coding task in the tested
configuration.

## Model assessments

### 1. Qwen3.6 27B MTP

**Verdict:** strongest patch on the comparable bounded task, with insufficient
evidence to declare it the best general coding model.

The `qwen3.6-27b-mtp-q8-0` result introduced one named width-10 constant and
used it for the header and every rendered row. That was the clearest expression
of the requested shared formatting policy. It removed the opportunity for the
two paths to drift and included a comment explaining the constraint. Ordinary
tests, hidden tests, and the build all passed.

This design is why Qwen 27B ranks first despite its speed. The conclusion is
strictly task-local. It has not yet faced the hard source-race task, the other
full-suite implementations, or the two review tasks. A clean abstraction on
one easy task cannot prove safe repository-wide behavior.

The run took 766.7 seconds, 3.34 times the Qwen 35B reference time. It used 22
tool calls, generated 8,699 tokens at 15.57 tokens per second, and processed
prompt tokens at 48.43 tokens per second. It was the second-slowest successful
screen, ahead of Gemma only. Choose it when the observed patch quality matters
more than turnaround time, then review the result as unproven on harder work.

### 2. Qwen3.6 35B-A3B MTP

**Verdict:** best-supported and fastest model in the current evidence, but not
a demonstrated overall quality winner.

The `qwen3.6-35b-a3b-mtp-ud-q8-k-xl` result is the only one backed by the
complete version 4 suite. It solved three of six implementation tasks. The
successful work covered formatting, EINTR handling, and symlink behavior. One
otherwise passing cancellation attempt was invalidated because it left a
generated executable in the fixture. Both hard safety implementations had
real behavioral failures. One review also made a material concurrency error.

On `re-align`, it used matching width-11 literals for the header and rows. The
patch was functionally correct and passed all grading, but the longest relevant
value only requires width 10. The extra column was unnecessary and the shared
policy remained duplicated. This was weaker than Qwen 27B's bounded design,
even though the model has much stronger evidence across the suite.

The comparable `re-align` attempt took 229.9 seconds with 23 tool calls. It
generated 12,596 tokens at 70.03 tokens per second and processed prompt tokens
at 111.30 tokens per second. It is the practical baseline when predictable
runtime and known limitations matter. Its 3/6 suite result is the threshold a
new candidate should exceed, not proof that it is safe without review.

### 3. KAT-Coder v2.5 Dev

**Verdict:** correct, efficient, and the strongest candidate not yet run
through the hard safety screen and complete suite.

The `kat-coder-v2.5-dev-q8-0` patch made the required width-10 changes to the
header, rows, and expectations. It was minimal and all grading passed. Its only
design weakness relative to Qwen 27B was representing the shared width as two
matching literals instead of one named policy.

The run took 319.3 seconds, 1.39 times the Qwen 35B reference, with 25 tool
calls. It generated 12,321 tokens at 41.42 tokens per second and processed
prompt tokens at 493.52 tokens per second. This is an attractive balance on
the easy task, but it has no hard-task or review evidence. KAT should be the
next non-Qwen model promoted through the source-race screen before its result
is interpreted as more than a promising bounded success.

### 4. Muse Glimmer 30B DFlash

**Verdict:** capable of a correct minimal patch, but currently better treated
as a supervised alternate than an autonomous default.

The `muse-glimmer-30b-kquant-dynamic-dflash` result made the same width-10
header, row, and expectation changes as KAT. Ordinary tests, hidden tests, and
the build passed. The retained patch itself gives no quality reason to place
Muse below KAT.

The operational behavior does. Muse took 670.2 seconds, 2.92 times the Qwen
35B reference, and made 40 tool calls, the most in the completed screen. It
generated 14,080 tokens at 18.69 tokens per second and read 958,355 cached
tokens. Earlier testing also showed substantial harness sensitivity. A model
that performs well through Pi or Maki can be markedly less effective through
a different client even when the underlying server is unchanged. Use Muse
where its behavior has been accepted with the chosen harness, and keep a human
close enough to notice repeated probing or loss of convergence.

### 5. Gemma4 31B MTP

**Verdict:** correct bounded result, but presently too slow and too lightly
tested to recommend ahead of the main candidates.

The `gemma4-31b-it-q8-0-mtp` patch was the same correct width-10 implementation
as KAT and Muse. It changed the header, rows, and expectations without adding
unrelated work, and all grading passed. Its patch quality therefore belongs in
the same task-local group as those two models.

The run took 902.5 seconds, 3.93 times the Qwen 35B reference, making it the
slowest completed screen. It used 17 tool calls, generated 6,389 tokens at
11.38 tokens per second, and processed prompt tokens at 70.87 tokens per
second. Gemma remains useful as a model-family diversity check or second
opinion. Broader acceptance would be needed before the runtime cost is
justified for routine autonomous coding.

### 6. Ornith 1.0 35B

**Verdict:** functional alternate with a weaker bounded patch and no broader
quality evidence.

The `ornith-1.0-35b-q8-0` result used matching width-11 literals, just like
Qwen 35B. It passed ordinary tests, hidden tests, and the build, but used one
more column than necessary and left the policy duplicated. This is a small
design issue rather than a correctness failure. It is enough to rank Ornith
below the models that produced the exact minimal behavior on the only shared
task.

The run took 484.2 seconds, 2.11 times the Qwen 35B reference, with 18 tool
calls. It generated 15,049 tokens at 37.34 tokens per second and processed
prompt tokens at 358.97 tokens per second. It consumed substantially more
input and cached context than KAT. Ornith is reasonable as a second opinion,
but needs hard-task evidence before it can be trusted for safety-sensitive
repository changes.

## Disqualified models

### Laguna XS 2.1

`laguna-xs-2.1-q4-k-m` was interrupted after more than 21 minutes without
making an edit. It could use the raw tool protocol and answer short prompts,
but it did not converge on the version 4 easy coding screen. For an autonomous
agent, failure to turn valid tool use into a completed change is a quality
failure. Keep it to short-prompt and runtime experiments unless a material
model, template, runtime, or harness change justifies retesting.

### Laguna S 2.1

`laguna-s-2.1-q4-k-m` was interrupted after about 27 minutes without making an
edit in the older version 2 screen. This is not comparable version 4
performance data, but it is sufficient negative evidence for the current use
case. It remains an experimental inference model rather than a dependable
repository agent.

### DwarfStar DeepSeek V4 Flash

DwarfStar `deepseek-v4-flash` derived the likely fix in the older version 2
screen, then entered a repetitive loop and was interrupted. The runtime and Pi
tool protocol worked. The failure was the model's inability to turn its
reasoning into a finished patch. That loop disqualifies it from dependable
autonomous coding in the current configuration, while leaving it useful for
DeepSeek runtime research and tightly bounded prompting.

## Decision rule for a new model

A newly integrated Qwen or other candidate should progress through:

```bash
./rocmplete benchmark agent --preset NEW_PRESET --task re-align
./rocmplete benchmark agent --preset NEW_PRESET --task re-source-race
./rocmplete benchmark agent --preset NEW_PRESET
```

Use these quality gates in order:

1. It must converge and solve the easy screen. Speed is secondary at this
   stage.
2. It must complete the destructive source-race safety task. A loop, plausible
   explanation without a patch, or hidden-test failure is a rejection.
3. It must complete the frozen suite. Compare implementation solve rate and
   the nature of failures, not only the aggregate score.
4. Human-grade both review answers for factual correctness. A polished answer
   containing a material concurrency or lifecycle error does not pass.
5. Repeat a finalist three times before changing the recommended default.

To be a demonstrated quality improvement over the current evidence, a new
model should retain a clean shared-policy implementation on the easy screen,
exceed the inherited 3/6 implementation baseline, avoid both inherited hard
safety failures, and produce factually sound reviews. A complete version 5 run
must also report the three new tasks separately until there is a repeated
version 5 baseline. Merely beating the 229.9-second easy-task time would make
the model faster, not better.
