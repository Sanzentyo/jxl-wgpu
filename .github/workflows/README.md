# Disabled workflows

GitHub Actions is intentionally disabled while the GPU codec is under active
bring-up. `ci.yml.disabled` is retained as a reference; it is not an active workflow.
This documentation does not enable Actions or change the runner/resource policy.

## Local validation

Use [validation by change](../../CONTRIBUTING.md#validation-by-change) to select
checks for the task. Prose/link-only edits do not require the full GPU gate set.
Capability changes still require the
[capability-change gates](../../CONTRIBUTING.md#capability-change-gates), including
formatting, workspace checks, Clippy, tests/documentation, portable WebAssembly,
reference-harness, applicable Metal-harness, and codec-readback evidence.

The retained workflow records the portable Linux/macOS and minimum-Rust-version
matrix. Missing local hardware or tools must be reported as unverified gates,
not as successful CI. Do not enable the workflow as incidental cleanup.

Re-enabling CI is a separate change: decide the runner/resource policy, reconcile
the workflow with the current manifest and validation commands, then rename
`ci.yml.disabled` to `ci.yml` to enable its push, pull-request, and manual triggers.
