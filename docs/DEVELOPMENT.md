# Internal development and validation

These notes support maintainer-directed implementation and authorized agent work.
The [project phase](../README.md#project-phase) defines the current scope; this is
not an external-contributor onboarding guide.

## Environment

Use Rust 1.98 or later, as declared in [Cargo.toml](../Cargo.toml). Codec and
GPU-kernel execution requires a compatible adapter. WebAssembly checks require the
`wasm32-unknown-unknown` target; compilation alone does not verify browser execution.
CPU/native image-codec oracles remain development-only under the
[upstream boundary](UPSTREAM_BOUNDARY.md).

Use the repository's default `target/` directory for Cargo builds and validation. Do not create
per-task Cargo output directories under `.git` or elsewhere. When disk space runs low, stop the
active builds and validation jobs, run `cargo clean` from the repository root, then restart the
required checks. Never clean an output directory while a build or test still uses it.

Use exactly two libtest threads for tests, including GPU tests, as directed by the maintainer
on 2026-09-21. [Cargo configuration](../.cargo/config.toml) sets `RUST_TEST_THREADS=2`; explicit
test commands use `--test-threads=2`. Run separate GPU test executables or Cargo test jobs
sequentially, so process-level parallelism does not multiply the two-thread workload.

## Validation by change

| Change | Required evidence |
|---|---|
| Prose, links, or navigation only | Check Markdown, paths/anchors, and claims against current owning documents/source. Check for stale references after moving or deleting files. No full GPU suite is needed for prose-only changes. |
| Executable documentation examples | Compile/run affected examples when the toolchain and hardware are available; shell syntax alone is not runtime validation. |
| Test setup or helper changes only | Run affected test targets with their required adapters/oracles and unchanged case matrices, tolerances, and ownership checks. Use the same workload before and after for runtime comparisons. |
| Host parsing, metadata, or format logic | Affected crate/target tests, malformed-input and limit cases, plus downstream checks for affected codec contracts. |
| WGSL, codec/output behavior, or GPU ownership | Tests on an actual adapter with two libtest threads, applicable independent oracles, precision/invalid-input checks, budget admission, cancellation, and retained-output lifetime. |
| Advertised capability or cross-workspace contract | The full gates below plus the relevant roadmap acceptance evidence. Focused passing tests do not replace these gates. |

Integration targets use `tests/<target>/main.rs` with ordinary child modules. Shared
helpers belong in [jxl_test_support](../tools/jxl_test_support/README.md), not
cross-target source includes. Remove unused/dead code; necessary platform or
ownership exceptions use narrowly scoped, reason-bearing `expect` attributes.

## Capability-change gates

Run from the repository root while CI is disabled:

```console
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets --all-features --no-fail-fast -- --test-threads=2
cargo test --locked --workspace --all-features --doc -- --test-threads=2
cargo doc --locked --workspace --no-deps
cargo check --locked --target wasm32-unknown-unknown \
  -p jxl_gpu_bitstream -p jxl_gpu_protocol -p jxl_gpu_formats \
  -p jxl_wgpu -p jxl_wgpu_decode -p jxl_wgpu_encode
cargo run --locked -p jxl_gpu_harness -- verify --backend reference
cargo run --locked -p jxl_gpu_harness -- verify --backend wgpu
cargo run --locked -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```

The [disabled workflow](../.github/workflows/ci.yml.disabled) also records the
portable Linux/macOS and minimum-Rust-version matrix; retain applicable Metal GPU
evidence. `verify --backend reference` validates a development reference backend,
not production GPU execution. GPU verification and codec readback are distinct
checks, and neither alone establishes full JPEG XL conformance.

Record unavailable toolchains, targets, adapters, or oracles as missing evidence,
not successful checks. Do not promote a feature to Done without its acceptance
evidence or widen tolerances to compensate for a missing oracle. The absence of
CI runs is not a successful CI result.

## Updating evidence

The [roadmap](FULL_JPEG_XL_ROADMAP.md) owns capability status and same-commit
documentation requirements. Capability changes update its affected item, the root
summary, and the affected crate README. Update [WGSL memory](WGSL_MEMORY.md) for
shader ABI/memory changes and [the corpus](CONFORMANCE_CORPUS.md) for coverage.
Use the [topic index](README.md) for the owning API and color/render contracts.

Offline fixture generators can rewrite checked-in data and launch native tools;
they are not ordinary read-only test runs. When regeneration is part of the task,
use the corpus recipe, retain pinned oracle versions/provenance, and review the
fixture and reference diff. Keep scratch outputs in temporary/build directories.

Update [benchmarks](GPU_BENCHMARKS.md) only for collected measurements, recording
adapter/backend, dimensions/format, output path, workload/concurrency,
warmup/iterations, and validation contract. Distinguish codec GPU batching from
host-thread fan-out and aggregate readback. Historical corpus counts and timings
are evidence for their recorded scope, not a current all-platform guarantee.

[CI remains intentionally disabled](../.github/workflows/README.md). Enabling it
requires a separate runner/resource-policy decision; it is not part of routine
cleanup. Change summaries state scope, actual validation, and remaining limits.
