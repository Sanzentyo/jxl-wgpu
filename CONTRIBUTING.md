# Contributing

## Environment and scope

Use Rust 1.98 or later; [Cargo.toml](Cargo.toml) declares the supported minimum.
Codec and GPU-kernel tests require a compatible adapter. The
[harness](tools/jxl_gpu_harness/README.md) can enumerate adapters, but individual
operations still validate their device requirements.

Production codec execution never selects a CPU image codec. Independent CPU/native
oracles belong in development dependencies and offline tools. See the
[upstream boundary](docs/UPSTREAM_BOUNDARY.md) when changing dependencies or moving
code between those layers.

## Validation by change

| Change | Validation to select |
|---|---|
| Prose, links, or document navigation only | Check Markdown, relative paths/anchors, and claims against their source. A prose-only change does not require the complete GPU suite. Changed executable examples need the relevant compile/run check when the environment permits. |
| Bounded host parsing, metadata, or format logic | Run the affected crate/target tests, including malformed-input and boundary cases; add downstream checks where behavior crosses into a codec path. |
| WGSL, codec execution, output, or GPU resource ownership | Run affected tests serially on an actual adapter, with the applicable oracle, precision, invalid-input, budget, and cancellation cases. Expand scope when shared contracts change. |
| Advertised capability or cross-workspace contract | Apply the capability-change gates below and the specific acceptance evidence in the roadmap. A narrow passing test is not a substitute for those gates. |

Focused examples, to use only for the corresponding change:

```console
cargo test --locked -p jxl_gpu_bitstream --lib --all-features
cargo test --locked -p jxl_wgpu_decode --test preview --all-features -- --test-threads=1
```

Use the affected integration target instead of `preview` for unrelated work.
Integration targets use `tests/<target>/main.rs`; shared helpers belong in
[jxl_test_support](tools/jxl_test_support/README.md), not cross-target source includes.
The workspace denies unused/dead code. Necessary platform or ownership exceptions
use narrowly scoped, reason-bearing `expect` attributes.

## Capability-change gates

Capability changes retain the full local validation boundary while CI is disabled.
Run from the repository root; the WebAssembly check also requires the
`wasm32-unknown-unknown` Rust target to be installed.

```console
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets --all-features -- --test-threads=1
cargo test --locked --workspace --all-features --doc
cargo doc --locked --workspace --no-deps
cargo check --locked --target wasm32-unknown-unknown \
  -p jxl_gpu_bitstream -p jxl_gpu_protocol -p jxl_gpu_formats \
  -p jxl_wgpu -p jxl_wgpu_decode -p jxl_wgpu_encode
cargo run --locked -p jxl_gpu_harness -- verify --backend reference
cargo run --locked -p jxl_gpu_harness -- verify --backend wgpu
cargo run --locked -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl \
  --format u8 --output-target cpu-readback
```

The disabled workflow also describes portable Linux/macOS and minimum-Rust-version
checks. Retain its Metal-specific GPU evidence where applicable. WebAssembly
compilation is not evidence of browser runtime correctness.

`verify --backend reference` checks a development reference backend, not the
production GPU codec. `verify --backend wgpu` and codec readback provide distinct
GPU evidence; neither by itself proves full JPEG XL conformance.

When a toolchain, target, adapter, or external oracle is unavailable, report the
missing gate and its impact. Do not label it passed, substitute a CPU production
path, broaden tolerances, or promote a roadmap item to Done without its required
evidence. A scoped change can still be submitted with those limitations explicit.

## Evidence and documentation

[The roadmap](docs/FULL_JPEG_XL_ROADMAP.md) owns capability status and the same-commit
documentation requirements. Update its affected item, the root capability summary,
and the affected crate README when capabilities change. Update
[WGSL memory](docs/WGSL_MEMORY.md) for shader ABI or memory-contract changes and
[the conformance corpus](docs/CONFORMANCE_CORPUS.md) when coverage changes.
Use the [documentation index](docs/README.md) for other domain-specific contracts.

Fixture generation is separate from running tests: offline tools can rewrite
checked-in data and launch native processes. Use the affected corpus recipe,
record pinned oracle versions and provenance, and review the resulting diff.
Regenerate or replace references only when the task includes that change.

Change [benchmark records](docs/GPU_BENCHMARKS.md) only for measurements actually
collected. Record the adapter/backend, input dimensions and format, output target,
workload/concurrency, warmup/iterations, and validation contract. Keep codec GPU
batching distinct from host-thread fan-out and aggregate readback.

## Task requests for contributors and agents

Describe the outcome, important boundaries, and evidence of completion. These are
examples, not mandatory phases or additional skills:

> Implement `<roadmap item or bounded variant>` through the production GPU path.
> Preserve typed rejection outside that scope and the resource/validation contract.
> Complete the applicable acceptance checks and documentation updates; report any
> unavailable gate rather than stopping after the first implementation.

> Improve `<measured bottleneck>` for `<input, adapter, and output path>` without
> changing correctness, numerical bounds, or resource ownership. Compare before
> and after under the same workload, fix regressions introduced by the change,
> and report both measured results and remaining limits.

> Simplify `<documentation area>` without changing runtime behavior or capability
> claims. Preserve useful references, check affected links and examples, and leave
> unrelated implementation and fixture files unchanged.

## CI and review

[GitHub Actions is intentionally disabled](.github/workflows/README.md). Do not
rename the disabled workflow or enable CI as part of an unrelated change. Include
the scope, actual validation results, and known limitations in the change summary;
absence of a CI run is not a successful CI result.
