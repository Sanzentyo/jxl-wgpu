# JPEG XL test support

This unpublished workspace crate owns helpers shared by codec integration tests and offline
fixture generators. The decoder selects it only as a development dependency. Its CPU references
and native oracle processes are absent from the production codec dependency graph.

The module tree follows the source tree:

- `corpus` reads checked-in image inputs and comparison data.
- `fixtures` describes reproducible families and assembles their bounded codestream metadata.
- `oracles` provides independent CPU, scalar and offline native comparisons.
- `gpu` drives whole or fragmented test input and reads explicitly requested test output.
- `offline` writes fixture files and manages native generator processes and hexadecimal formats.

Each decoder integration target has a `tests/<target>/main.rs` entry point. Its private modules
live below that directory and use ordinary `mod` declarations. Examples with private helpers
follow the same `examples/<target>/main.rs` layout. Shared helpers are imported from this crate;
source files are never included through `#[path]` or compiled repeatedly under different names.
Existing `cargo test --test <target>` and `cargo run --example <target>` commands stay the same.

The workspace denies `dead_code` and `unused`. Remove obsolete helpers and imports instead of
silencing the diagnostics. Platform or ownership exceptions belong on the smallest applicable
item as `expect(..., reason = "...")`; a stale expectation must produce a warning. A resource
retained until GPU completion can be necessary even when it is never read explicitly. Such
ownership should be expressed in its containing type and explained at that boundary.

Run the workspace checks from the repository root with Rust 1.98 or later:

```sh
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features -- --test-threads=1
```

GPU tests run serially. Fixture generators use the same manifests and helpers as the tests;
their commands, pinned oracle versions and observed errors are recorded in
[`CONFORMANCE_CORPUS.md`](../../docs/CONFORMANCE_CORPUS.md).
