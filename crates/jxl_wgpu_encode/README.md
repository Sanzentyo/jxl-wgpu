# jxl_wgpu_encode

GPU-required JPEG XL encoding orchestration for `wgpu`. This crate does not contain a CPU pixel
encoder or a CPU fallback. The concrete `LosslessGray8Encoder` reads an 8-bit grayscale
pitch-linear storage buffer directly on the GPU and emits a standards-compatible lossless Modular
codestream or `jxlc` container.

## Lossless Gray8 profile

- Extents are `1..2^30` on each axis, further bounded by the selected WebGPU device's storage
  binding, buffer, and dispatch limits.
- The frame is split into standard 256x256 PassGroups. Edge groups may be one pixel wide or high.
- One GPU invocation handles each PassGroup. All invocations are submitted as one workgroup grid,
  then all artifacts are copied to one staging buffer and completed by one runtime-neutral future,
  one map callback, and one bounded poll slot.
- Every group produces independent Gradient-predictor residuals, LZ77/raw token events, and
  histograms. The host validates every artifact, combines only the histograms, creates the shared
  prefix code, and serializes groups in standard row-major JPEG XL TOC order.
- LF global carries the shared Modular tree and entropy code; LF groups and HF global are empty;
  each PassGroup carries its own group header and token stream.

`LosslessGray8Encoder::memory_plan` reports the exact source binding range, parameter-storage
bytes, artifact-storage bytes, mapped readback bytes, total encoder-owned live bytes, and the group
grid before submission. A single shared `MemoryBudget` reservation covers the whole frame. The
exclusive three-buffer pool lease and reservation survive until the map callback and mapped-range
consumer are both finished, including when the returned future is abandoned.

The returned `LosslessGray8Submission` implements `Future` without depending on an async runtime;
native callers may instead use `wait`. `group_grid` and `ordered_groups` expose the exact dispatch
rectangles and normative PassGroup order before completion.

```rust,no_run
# use jxl_wgpu_encode::{BufferImageSource, LosslessGray8Encoder, WgpuContext};
# fn submit(
#     context: WgpuContext,
#     source: BufferImageSource,
# ) -> Result<(), jxl_wgpu_encode::EncodeError> {
let encoder = LosslessGray8Encoder::new(context);
let plan = encoder.memory_plan(&source)?;
assert_eq!(plan.group_grid.groups, plan.group_grid.columns * plan.group_grid.rows);

let submission = encoder.submit_container(source)?;
for group in submission.ordered_groups() {
    // `group.index` is also its standard row-major PassGroup/TOC index.
    let _rectangle = (group.x, group.y, group.width, group.height);
}
let jxl_container = submission.wait()?;
# let _ = jxl_container;
# Ok(())
# }
```

Single-group containers additionally carry the private `jwgp` acceleration index understood by
the stock GPU decoder. Its current schema represents one contiguous token span, so multi-group
containers intentionally omit that private box; they remain ordinary interoperable JPEG XL
containers. The encoder's conformance tests decode all boundary cases exactly with the published
Rust `jxl` decoder and, when installed, reference `djxl`.
