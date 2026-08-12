# Coding-agent evaluation maintenance

The frozen coding-agent suite answers a different question from GPU smoke
acceptance and native llama.cpp performance tests: can one exact managed model,
through one fixed harness, understand and safely change real code?

## Ownership and boundaries

The authoritative inputs are:

- `evaluations/coding/tasks.json`: suite identity, pinned public repositories,
  base and reference commits, expected Git trees, prompts, and task metadata;
- `evaluations/coding/hidden/`: post-run Go or Python tests with exact SHA-256
  hashes;
- `src/rocmplete/agent_evaluation.py`: definition validation, source mirrors,
  single-commit fixtures, Pi execution, transcript capture, grading,
  checkpointing, and Markdown reporting;
- `src/rocmplete/pi_agent.py`: the normal Pi provider policy and the shared
  sandbox extension point used to mount a prepared Go module cache read-only
  when the selected task requires it;
  and
- `tests/test_agent_evaluation.py`: schema, isolation, audit, metrics, review,
  dry-run, and report behavior.

Raw machine results belong below
`StorageLayout.agent_evaluations`, normally
`apps/agent-evaluation/` in the configured data directory. Do not commit raw
transcripts, generated patches, timing, or host-specific result JSON to the
source tree. A curated result summary may be added to an appropriate research
or hardware-acceptance record when it states the exact suite fingerprint,
model artifact, runtime, harness, context, repetitions, and hardware.
The [quality-oriented model grouping](coding-agent-model-quality.md) interprets
the current evidence. Exact same-host measurements and the full-suite
reference are recorded in the
[Fedora 44 Strix Halo coding-agent comparison](hardware-acceptance.md#coding-agent-comparison-2026-08-11).

## Frozen version 5 task set

Version 5 contains nine implementation tasks and two review tasks:

Implementation tasks:

- `re-align` uses Go in reencode, from `d8b1874` to `7f1f600`, to test shared
  table-width behavior.
- `fz-eintr` uses Go in fzr, from `8301ead` to `a349035`, to test exact EINTR
  retry semantics.
- `fz-symlink` uses Go in fzr, from `f0e691e` to `745bfbc`, to test symlink
  identity and lazy metadata.
- `re-cancel` uses Go in reencode, from `5f4881b` to `46f6c20`, to test context
  propagation and fallback safety.
- `fz-sort-cancel` uses Go in fzr, from `ecab57a` to `f955966`, to test
  cancellation inside stable sorting.
- `re-source-race` uses Go in reencode, from `34ae314` to `4919870`, to test
  destructive source-replacement safety.
- `proxy-late-probe` uses Go in ssh-host-proxy, from `b8513df` to `03d7624`, to
  test ownership of late asynchronous connections.
- `rc-selinux-verify` uses standard-library Python in ROCmplete, from
  `6f866b3` to `a333de2`, to test labeling before hashing and receipts.
- `nonet-lifecycle` uses Go and cgo in nonet, from `3f63875` to `374e8be`, to
  test signal relay and parent-death safety.

Review tasks:

- `review-reencode-lifecycle` reviews the destructive lifecycle at `7f1f600`.
- `review-fzr-concurrency` reviews scan, filter, selection, and terminal
  concurrency at `a349035`.

The implementation tasks intentionally span easy, medium, hard, and
safety-critical work. Reference commits are maintainer evidence and a way to
validate hidden tests; the grader never compares an agent patch byte-for-byte
with the reference implementation.

## Fixture and grading lifecycle

For every task and repetition the runner:

1. Clones a fixed allowlisted public repository into a managed bare mirror.
2. Resolves the pinned base commit and fails unless its Git tree matches the
   recorded tree hash.
3. Uses `git archive` and a traversal-safe extractor to create a clean source
   tree. Symlinks and special archive members are rejected.
4. Adds controlled `AGENTS.md` instructions, initializes a new repository,
   and creates exactly one synthetic commit with no remote. The allowlisted
   ROCmplete source has its existing root instructions replaced so the model
   receives the same evaluation policy as every other task. An unexpected
   instruction file in any other source still fails closed. Later public
   history is not available inside the fixture.
5. Selects the fixed controller-owned adapter declared for the reviewed
   repository. Go tasks download only `go.mod`/`go.sum`-pinned modules at the
   controller boundary and mount the module cache read-only for Pi. The Python
   task uses the repository's standard-library test suite and receives no
   dependency environment or package installer preparation.
6. Runs Pi noninteractively with no saved session, extensions, skills, or
   prompt templates. The normal ROCmplete model catalog still owns endpoint,
   context metadata, output allowance, and sampling policy.
7. Preserves the complete worktree diff and structured Pi transcript.
8. Copies the worktree into a grading directory, restores pinned dependency
   and fixture-instruction files, runs ordinary tests, adds the hash-verified
   hidden test, reruns tests, and runs the adapter's build or compilation
   check.

Agent-written test changes remain in the grading copy. Some historical tasks
legitimately update an existing expectation, and hidden tests remove any
benefit from merely weakening an old assertion. Dependency changes remain an
invalidating policy violation and are restored before grading. A root-level
executable named `reencode` or `fzr` is retained in the captured patch but also
invalidates the attempt as a generated Go build artifact. The same rule covers
the repository-named outputs of ssh-host-proxy and nonet.

The worktree is validated before copying. Links, special files, excessive
file counts, and oversized files or patches are rejected. Hidden tests and
protected snapshots are siblings of the sandboxed fixture and are not mounted
into Pi.

Review tasks require only `ROCMLETE_EVAL_ANSWER.md`. Unexpected source changes,
network-command evidence, missing evidence, or an implausibly short or long
answer invalidate capture. Valid answers remain `review-pending` for human
factual grading and never increase the implementation solve rate.

## Fair comparison policy

Use one machine, backend, image, context, harness version, thinking level,
task selection, repetition count, and runtime policy for a model comparison.
Pi remains the fixed harness for version 5. OpenCode, Maki, and OMP have
different tool prompts and context behavior and belong in a separately
labelled harness comparison.

The default 131072-token context fits every maintained candidate used for the
initial screening. Start with one attempt per task to validate a broad model
set, then run three fresh repetitions for finalists. Do not tune the prompt or
hidden grader after seeing a new model's answer. A speculative preset is a
practical runtime configuration, not proof that speculation preserved task
quality; retain the matching non-speculative control when that distinction
matters.

Use a 45-minute wall-clock ceiling for each future model-evaluation attempt.
Apply it at the operator boundary so ordinary benchmark execution remains
intentionally unbounded:

```bash
timeout --foreground --signal=INT --kill-after=90s 45m \
  ./rocmplete benchmark agent --preset PRESET --task TASK
```

The runner checkpoints an interrupted attempt and removes its model container.
A clear repetitive loop may be stopped earlier and recorded as
non-convergence. Do not wrap a complete multi-task suite in one 45-minute
timeout: the ceiling belongs to each active attempt, not to the aggregate
suite. If an attempt in a full-suite promotion reaches the ceiling, interrupt
the suite and reject that promotion rather than allowing the model unlimited
time. Preserve older 20-minute and 60-minute records under their actual
policies instead of relabeling them.

The model process must reach a loopback HTTP server, so the shared Pi sandbox
retains the host network. Pi startup is offline, Go's proxy is disabled for Go
tasks, the Python task has no prepared third-party environment, and every task
explicitly prohibits network use. Recognized network commands in the structured
transcript invalidate the attempt. This detects ordinary agent behavior but is
not adversarial syscall enforcement. Do not describe the public historical
suite as contamination-proof. A private unpublished holdout remains the
stronger final check.

## Adding or changing a task

Changing any prompt, pin, tree, hidden test, or fixture instruction changes
the suite fingerprint and requires a new named suite version. Do not silently
rewrite `rocmplete-coding-v5` after results exist. Version 1 was superseded
during initial calibration because asking the agent to run `go build ./...`
left an untracked executable in a single-main-package fixture. Version 2 moved
that build check to the controller. Version 3 makes the grader's existing
200-to-2,000-word review-answer bound visible in both review prompts; version 2
results remain valid under their recorded fingerprint and must not be relabeled.
Version 4 makes the fixture's existing no-generated-build-artifacts instruction
an invalidating grading rule after version 3 calibration captured an ignored
root-level Go executable. Earlier results remain evidence under their exact
fingerprint and project revision and are not relabeled under later suite
versions. Version 5 adds the ssh-host-proxy, ROCmplete, and nonet implementation
tasks plus a fixed Python standard-library adapter. It does not relabel
version 4 measurements as version 5 results.

For a new implementation task:

1. Choose a base that builds and passes its existing tests.
2. Record the full base, tree, and reference hashes.
3. Write a behavior-focused prompt that does not disclose the patch.
4. Add a standalone hidden Go or Python test in the destination form permitted
   by the fixed repository toolchain.
5. Confirm the hidden test fails on the base and passes on the reference.
6. Confirm the complete reference diff, including legitimate test updates,
   is graded `solved` without dependency changes.
7. Run the focused Python tests and the complete Tier 1 suite.
8. Dry-run the public command before target-hardware execution.

Repository-to-toolchain ownership is an allowlist in
`src/rocmplete/agent_evaluation.py`. Go and Python commands are selected by
controller code, not task JSON. Adding another language requires a reviewed
fixed adapter with explicit dependency preparation, sandbox environment,
ordinary-test, hidden-test, and build behavior. Never make the frozen task
definition an arbitrary command runner.

Keep unrelated repositories separate even if their tasks have a similar use
case. A task source is executable input to a tool-using model, so repository
allowlisting, exact pins, archive validation, and sandbox boundaries must fail
closed.
