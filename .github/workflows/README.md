# Disabled workflows

GitHub Actions is intentionally disabled while the GPU codec is under active bring-up. The
portable workflow is retained as `ci.yml.disabled`, which GitHub does not recognize as a workflow
definition. Rename it to `ci.yml` to re-enable push, pull-request, and manual runs after the CI
resource policy is decided.

Until then, capability commits must run the equivalent formatting, workspace check, Clippy, test,
documentation, WebAssembly, reference-harness, Metal-harness, and codec-readback gates locally.
