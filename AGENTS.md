# Repository guidance

## Boundaries

- Production image-codec work is GPU-required. CPU codecs and native oracles belong
  in development support, never in a production fallback. Bounded host parsing,
  scheduling, container assembly, and explicit readback are allowed.
- Preserve typed rejection, validation-before-authority, and byte-budgeted ownership
  through GPU completion, cancellation, and retained output. Do not treat an
  explicitly unvalidated handoff as a validated frame.
- Capability claims require the roadmap's acceptance evidence. Do not weaken
  numerical bounds, negative tests, or oracle independence to make a change pass.

## Read for the task

| Task | Relevant context |
|---|---|
| Public API or behavior | The affected crate README and source; [documentation index](docs/README.md) for domain-specific contracts. |
| Capability change | Relevant items and acceptance gates in [the roadmap](docs/FULL_JPEG_XL_ROADMAP.md). |
| Shader, binding, scheduling, or resource lifetime | Relevant sections of [GPU architecture](docs/GPU_ARCHITECTURE.md) and [WGSL memory](docs/WGSL_MEMORY.md). |
| Tests, fixtures, or oracles | [Test support](tools/jxl_test_support/README.md) and the affected family in [the corpus](docs/CONFORMANCE_CORPUS.md). |
| Validation or performance evidence | [CONTRIBUTING.md](CONTRIBUTING.md); [benchmarks](docs/GPU_BENCHMARKS.md) for measured claims. |

These are task-specific entry points, not a prerequisite reading sequence. Large
reference documents and `README.legacy.md` do not need to be loaded for every edit.

## Working boundary and completion

For the requested change, run relevant local checks, fix regressions it introduces,
and rerun affected checks without seeking approval between iterations. Choose the
scope from [validation by change](CONTRIBUTING.md#validation-by-change); GPU tests
run serially. Keep generated scratch output in build or temporary directories.
Fixture/reference replacement, enabling CI, publishing crates, and remote changes
need to be part of the authorized task, not incidental cleanup.

Finish the implementation, affected documentation, and proportionate validation
rather than stopping at a first draft. For capability changes, follow the roadmap's
same-commit documentation requirements and acceptance gates. Report the actual
checks and their results, including unavailable toolchains, adapters, or oracles;
a skipped check is not a pass or evidence of conformance.
