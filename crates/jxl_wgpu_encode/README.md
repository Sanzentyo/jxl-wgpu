# jxl_wgpu_encode

GPU-required JPEG XL encoding orchestration for `wgpu`. This crate does not contain a CPU pixel
encoder or a CPU fallback. `LosslessModular8Encoder` reads Gray8, packed RGB8, or packed RGBA8
pitch-linear storage directly on the GPU and emits a standards-compatible lossless Modular
codestream or `jxlc` container.

## Lossless Modular8 profile

- Extents are `1..2^30` on each axis, further bounded by the selected WebGPU device's storage
  binding, buffer, and dispatch limits.
- Gray8 uses one unsigned native-endian `X` plane. RGB8 and RGBA8 use one unsigned native-endian
  interleaved plane in canonical RGB/RGBA order. Row pitch and plane offset may contain padding.
  Planar RGB, BGR/BGRA, and explicitly defined non-sRGB color specifications are rejected.
- `Default` and `Undefined` RGB color specifications are interpreted as sRGB, matching the compact
  all-default JPEG XL color header. RGBA is written as one default 8-bit unassociated alpha extra
  channel.
- RGB(A) is converted to JPEG XL reversible color transform type 0 (YCoCg) in WGSL. No transformed
  image or source pixels are read by the CPU.
- The frame is split into standard 256x256 PassGroups. Edge groups may be one pixel wide or high.
- One GPU invocation handles each PassGroup/channel pair. Dispatch parameters and artifacts use
  group-major, channel-major order. All artifacts are copied to one staging buffer and completed by
  one runtime-neutral future, one map callback, and one bounded poll slot.
- Every group/channel produces independent Gradient-predictor residuals, LZ77/raw token events, and
  histograms. The host validates every artifact, combines histograms per channel, creates the four
  JPEG XL context prefix codes, and serializes channels inside standard row-major TOC groups.
- LF global carries the shared Modular tree and entropy code; LF groups and HF global are empty;
  each PassGroup carries its own group header and token stream.

`LosslessModular8Encoder::memory_plan` reports the exact source binding range, parameter-storage
bytes, artifact-storage bytes, mapped readback bytes, total encoder-owned live bytes, and the group
grid before submission. A single shared `MemoryBudget` reservation covers the whole frame. The
exclusive three-buffer pool lease and reservation survive until the map callback and mapped-range
consumer are both finished, including when the returned future is abandoned.

The returned `LosslessModular8Submission` implements `Future` without depending on an async runtime;
native callers may instead use `wait`. `group_grid` and `ordered_groups` expose the exact dispatch
rectangles and normative PassGroup order before completion.

```rust,no_run
# use jxl_wgpu_encode::{BufferImageSource, LosslessModular8Encoder, WgpuContext};
# fn submit(
#     context: WgpuContext,
#     source: BufferImageSource,
# ) -> Result<(), jxl_wgpu_encode::EncodeError> {
let encoder = LosslessModular8Encoder::new(context);
let plan = encoder.memory_plan(&source)?;
assert_eq!(plan.group_grid.groups, plan.group_grid.columns * plan.group_grid.rows);

let submission = encoder.submit_container(source)?;
let source_format = submission.format();
for group in submission.ordered_groups() {
    // `group.index` is also its standard row-major PassGroup/TOC index.
    let _rectangle = (group.x, group.y, group.width, group.height);
}
let jxl_container = submission.wait()?;
# let _ = (jxl_container, source_format);
# Ok(())
# }
```

Single-group Gray8 containers additionally carry the private `jwgp` acceleration index understood
by the stock GPU decoder. Its current schema represents one contiguous single-channel token span,
so RGB(A) and multi-group containers intentionally omit that private box; they remain ordinary
interoperable JPEG XL containers. Conformance tests decode odd sizes and extreme aspect ratios
exactly with the published Rust `jxl` decoder and reference `djxl`.
