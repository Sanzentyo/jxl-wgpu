# Repository guidance

This is maintainer-directed GPU codec implementation and conformance work, not a
public contribution program or a production-ready release. Follow the
[project phase](README.md#project-phase); do not add contributor onboarding,
release promises, or publication workflows as incidental documentation cleanup.

## Boundaries

- Image-codec work is GPU-required. CPU image codecs and native oracles belong in
  development support, never a production fallback. Bounded host parsing,
  scheduling, container assembly, and explicit readback are allowed.
- Preserve typed rejection, validation-before-authority, and byte-budgeted ownership
  through GPU completion, cancellation, and retained output. An explicitly
  unvalidated handoff is not a validated frame.
- Capability claims require the roadmap's evidence. Do not weaken numerical bounds,
  negative tests, or oracle independence to make a change pass.

## Read for the task

| Task | Relevant context |
|---|---|
| API or behavior | Affected crate README and source; [topic index](docs/README.md) for the relevant contract. |
| Capability change | Affected items and acceptance gates in [the roadmap](docs/FULL_JPEG_XL_ROADMAP.md). |
| Shader, binding, scheduling, or resource lifetime | Relevant sections of [GPU architecture](docs/GPU_ARCHITECTURE.md) and [WGSL memory](docs/WGSL_MEMORY.md). |
| Tests, fixtures, or oracles | [Test support](tools/jxl_test_support/README.md) and the affected family in [the corpus](docs/CONFORMANCE_CORPUS.md). |
| Validation or performance evidence | [Internal development](docs/DEVELOPMENT.md); [benchmarks](docs/GPU_BENCHMARKS.md) for measured claims. |

These are entry points, not a mandatory reading sequence. Keep current contracts in
their owning documents; remove duplicated status reports rather than archiving them
inside another always-discoverable README.

## Completion

Carry the authorized change through implementation, documentation, and the
[applicable checks](docs/DEVELOPMENT.md#validation-by-change), fixing regressions it
introduces without seeking approval between local iterations. Run tests with exactly two
libtest threads (`--test-threads=2`), including GPU tests. Keep separate GPU test processes
sequential; the two threads are within one test executable.
Fixture/reference replacement, enabling CI, publishing, and remote mutations must
belong to the authorized task, not incidental cleanup.

Follow the roadmap's same-commit documentation and acceptance gates for capability
changes. Report actual checks, failures, and unavailable toolchains, adapters, or
oracles; a skipped check is not a pass. Do not claim measured agent or runtime
improvements from documentation size alone.

## Waiting for final validation

- Start final validation once in a persistent process/session, retaining its handle,
  process identity, source snapshot, logs, and final receipt. Keep the tested source
  fixed until completion and reuse the repository's default Cargo output directory.
- Wait for command completion or 25 minutes, whichever comes first. At every such
  wakeup, run the [compact status script](tools/validation_status.rb), following the
  [recording procedure](docs/DEVELOPMENT.md#waiting-for-final-validation). It returns
  one line: `RUNNING`, `DONE exit=… stable=…`, or `UNKNOWN …`. Do not inspect intermediate
  logs, tail test output, reanalyse source, or report individual passing cases.
- On `RUNNING`, wait on the same process/session for another completion event or
  25-minute interval. Elapsed time, a timeout, or a previous 76-minute measurement
  never establishes completion; repeat for as long as the actual run needs.
- On `DONE`, read the final receipt, gate results, and completed logs. Check every
  gate and inspect failure details before fixing only the affected cases. `DONE`
  means execution ended, not that validation passed. On `UNKNOWN`, inspect the
  process/session and missing or invalid evidence; never assume success or start a
  duplicate run while the original job or its children may still be active.
- After a fix, run the applicable checks and retain unaffected evidence only when
  its source and runtime inputs are unchanged. Follow the development document's
  whitespace/prose rules; such cleanup alone does not require another full GPU run.
