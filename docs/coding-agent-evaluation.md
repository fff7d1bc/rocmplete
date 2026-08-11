# Coding-agent evaluation maintenance

The frozen coding-agent suite answers a different question from GPU smoke
acceptance and native llama.cpp performance tests: can one exact managed model,
through one fixed harness, understand and safely change real code?

## Ownership and boundaries

The authoritative inputs are:

- `evaluations/coding/tasks.json`: suite identity, pinned public repositories,
  base and reference commits, expected Git trees, prompts, and task metadata;
- `evaluations/coding/hidden/`: post-run Go tests with exact SHA-256 hashes;
- `src/rocmplete/agent_evaluation.py`: definition validation, source mirrors,
  single-commit fixtures, Pi execution, transcript capture, grading,
  checkpointing, and Markdown reporting;
- `src/rocmplete/pi_agent.py`: the normal Pi provider policy and the shared
  sandbox extension point used to mount a prepared Go module cache read-only;
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

## Frozen version 4 task set

Version 4 contains six implementation tasks and two review tasks:

| Task | Repository | Base | Reference | Purpose |
| --- | --- | --- | --- | --- |
| `re-align` | reencode | `d8b1874` | `7f1f600` | shared table-width behavior |
| `fz-eintr` | fzr | `8301ead` | `a349035` | exact EINTR retry semantics |
| `fz-symlink` | fzr | `f0e691e` | `745bfbc` | symlink identity and lazy metadata |
| `re-cancel` | reencode | `5f4881b` | `46f6c20` | context propagation and fallback safety |
| `fz-sort-cancel` | fzr | `ecab57a` | `f955966` | cancellation inside stable sorting |
| `re-source-race` | reencode | `34ae314` | `4919870` | destructive source-replacement safety |
| `review-reencode-lifecycle` | reencode | `7f1f600` | same | destructive lifecycle review |
| `review-fzr-concurrency` | fzr | `a349035` | same | scan, filter, selection, and terminal review |

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
   and creates exactly one synthetic commit with no remote. Later public
   history is not available inside the fixture.
5. Downloads only `go.mod`/`go.sum`-pinned modules at the controller boundary,
   warms the baseline, and mounts the module cache read-only for Pi. The agent
   receives `GOPROXY=off` and a temporary build cache.
6. Runs Pi noninteractively with no saved session, extensions, skills, or
   prompt templates. The normal ROCmplete model catalog still owns endpoint,
   context metadata, output allowance, and sampling policy.
7. Preserves the complete worktree diff and structured Pi transcript.
8. Copies the worktree into a grading directory, restores pinned dependency
   and fixture-instruction files, runs ordinary tests, adds the hash-verified
   hidden test, reruns tests, and builds the project.

Agent-written test changes remain in the grading copy. Some historical tasks
legitimately update an existing expectation, and hidden tests remove any
benefit from merely weakening an old assertion. Dependency changes remain an
invalidating policy violation and are restored before grading. A root-level
executable named `reencode` or `fzr` is retained in the captured patch but also
invalidates the attempt as a generated Go build artifact.

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
Pi is the version 4 fixed harness. OpenCode, Maki, and OMP have different tool
prompts and context behavior and belong in a separately labelled harness
comparison.

The default 131072-token context fits every maintained candidate used for the
initial screening. Start with one attempt per task to validate a broad model
set, then run three fresh repetitions for finalists. Do not tune the prompt or
hidden grader after seeing a new model's answer. A speculative preset is a
practical runtime configuration, not proof that speculation preserved task
quality; retain the matching non-speculative control when that distinction
matters.

The model process must reach a loopback HTTP server, so the shared Pi sandbox
retains the host network. Pi startup is offline, Go's proxy is disabled, the
task explicitly prohibits network use, and recognized network commands in the
structured transcript invalidate the attempt. This detects ordinary agent
behavior but is not adversarial syscall enforcement. Do not describe the
public historical suite as contamination-proof. A private unpublished holdout
remains the stronger final check.

## Adding or changing a task

Changing any prompt, pin, tree, hidden test, or fixture instruction changes
the suite fingerprint and requires a new named suite version. Do not silently
rewrite `rocmplete-coding-v4` after results exist. Version 1 was superseded
during initial calibration because asking the agent to run `go build ./...`
left an untracked executable in a single-main-package fixture. Version 2 moved
that build check to the controller. Version 3 makes the grader's existing
200-to-2,000-word review-answer bound visible in both review prompts; version 2
results remain valid under their recorded fingerprint and must not be relabeled.
Version 4 makes the fixture's existing no-generated-build-artifacts instruction
an invalidating grading rule after version 3 calibration captured an ignored
root-level Go executable. Earlier results remain evidence under their exact
fingerprint and project revision, not version 4 results.

For a new implementation task:

1. Choose a base that builds and passes its existing tests.
2. Record the full base, tree, and reference hashes.
3. Write a behavior-focused prompt that does not disclose the patch.
4. Add a standalone hidden `_test.go` file with a unique test-name prefix.
5. Confirm the hidden test fails on the base and passes on the reference.
6. Confirm the complete reference diff, including legitimate test updates,
   is graded `solved` without dependency changes.
7. Run the focused Python tests and the complete Tier 1 suite.
8. Dry-run the public command before target-hardware execution.

Keep unrelated repositories separate even if their tasks have a similar use
case. A task source is executable input to a tool-using model, so repository
allowlisting, exact pins, archive validation, and sandbox boundaries must fail
closed.
