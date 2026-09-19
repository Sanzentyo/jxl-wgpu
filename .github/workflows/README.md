# Disabled workflows

GitHub Actions is intentionally disabled during GPU codec implementation and
conformance development. `ci.yml.disabled` is a reference, not an active workflow.

Use [internal validation by change](../../docs/DEVELOPMENT.md#validation-by-change)
for task-specific checks. Capability changes retain the full
[local gates](../../docs/DEVELOPMENT.md#capability-change-gates), including the
portable/minimum-Rust matrix and applicable actual-GPU/Metal evidence.
Missing local tools or hardware are unverified gates, not successful CI.

Re-enabling CI is a separate maintainer decision: select the runner/resource policy,
reconcile the retained workflow with current manifests and validation commands,
then rename `ci.yml.disabled` to `ci.yml`. Do not enable it as incidental cleanup.
