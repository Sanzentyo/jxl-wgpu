# jxl_wgpu_encode

GPU-required JPEG XL encoding orchestration for `wgpu`. This crate does not contain a CPU pixel
encoder or a CPU fallback. `LosslessModularEncoder` reads packed Gray, RGB, or RGBA unsigned
integer pitch-linear storage directly on the GPU and emits a standards-compatible lossless
Modular codestream or `jxlc` container.

The complete encoder backlog, dependencies, and acceptance gates are tracked in
[`FULL_JPEG_XL_ROADMAP.md`](../../docs/FULL_JPEG_XL_ROADMAP.md). This README describes only the
currently executable profiles.

## Lossless Modular profile

- Extents are `1..2^30` on each axis, further bounded by the selected WebGPU device's storage
  binding, buffer, and dispatch limits.
- Valid sample depths are every integer in `1..=16`. `1..=8` use one native `u8` word per
  component; `9..=16` use one native `u16` word per component. The valid sample occupies the low
  bits and high padding bits are ignored. `LosslessModularFormat::pixel_format` constructs this
  explicit storage/valid-bits contract, including native-U16 10- and 12-bit layouts.
- Gray uses one unsigned `X` plane. RGB and RGBA use one unsigned interleaved plane in canonical
  RGB/RGBA order. Row pitch and plane offset may contain arbitrary padding. Planar RGB, BGR/BGRA,
  MSB-aligned sub-16-bit words, and explicitly defined non-sRGB color specifications are rejected.
- `Default` and `Undefined` RGB color specifications are interpreted as sRGB, matching the compact
  all-default JPEG XL color header. RGBA is written as one unassociated alpha extra channel at the
  same declared integer depth as RGB.
- RGB(A) is converted to JPEG XL reversible color transform type 0 (YCoCg) in WGSL. No transformed
  image or source pixels are read by the CPU.
- The frame is split into standard 256x256 PassGroups. Edge groups may be one pixel wide or high.
- One GPU invocation handles each PassGroup/channel pair. Dispatch parameters and artifacts use
  group-major, channel-major order. Small jobs use one mapped artifact allocation. Larger jobs use
  complete-channel-group batches bounded by storage-binding and dispatch limits.
- Multi-batch jobs first run a histogram pass to derive one stream-wide prefix code; a second pass
  validates and serializes each batch immediately, then releases its mapped artifact storage.
  Native builds drive the sequence with one runtime-neutral worker. Browser WebGPU drives the same
  two-pass sequence from map callbacks and the returned `Future`: each callback wakes the caller,
  and the next poll records exactly one next batch without requiring a Web Worker or a particular
  async runtime. Peak GPU memory is therefore bounded independently of total image area even though
  the final standard codestream remains contiguous.
- Every group/channel produces independent Gradient-predictor residuals, LZ77/raw token events, and
  histograms. The host validates every artifact, combines histograms per channel, creates the four
  JPEG XL context prefix codes, and serializes channels inside standard row-major TOC groups.
- LF global carries the shared Modular tree and entropy code; LF groups and HF global are empty;
  each PassGroup carries its own group header and token stream.

`LosslessModularEncoder::memory_plan` reports the detected valid bits, component storage bytes,
full and peak source binding ranges, peak parameter/artifact/readback bytes, diagnostic total
artifact bytes, batch count, exact GPU submission count, streaming mode, total encoder-owned live
bytes, and the group grid before submission. Streamed jobs report exactly twice the batch count:
one histogram and one serialization submission per batch. Every live batch uses the same shared
`MemoryBudget`. Its exclusive buffer-pool lease and reservation survive until the map callback and
mapped-range consumer are both finished, including when the returned future is abandoned.

The returned `LosslessModularSubmission` implements `Future` without depending on an async runtime;
native callers may instead use `wait`. Browser builds intentionally reject blocking `wait`, because
WebGPU completion is delivered by the browser event loop. Dropping an in-progress browser future
keeps the active batch's lease and shared byte-budget reservation alive through its map callback,
then releases them without submitting another batch. `group_grid` and `ordered_groups` expose the
exact dispatch rectangles and normative PassGroup order before completion.

```rust,no_run
# use jxl_wgpu_encode::{
#     BufferImageSource, LosslessModularEncoder, LosslessModularFormat, WgpuContext,
# };
# fn submit(
#     context: WgpuContext,
#     source: BufferImageSource,
# ) -> Result<(), jxl_wgpu_encode::EncodeError> {
let encoder = LosslessModularEncoder::new(context);
let plan = encoder.memory_plan(&source)?;
assert_eq!(plan.group_grid.groups, plan.group_grid.columns * plan.group_grid.rows);
assert!((1..=16).contains(&plan.bits_per_sample));

// Use this descriptor when constructing a packed native-U16 RGB10 source layout.
let _rgb10 = LosslessModularFormat::Rgb.pixel_format(10)?;

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

Single-group Gray8 containers additionally carry the optional private `jwgp` acceleration index.
Its current schema represents one contiguous 8-bit single-channel token span, so other depths,
RGB(A), and multi-group containers intentionally omit that private box; all remain ordinary
interoperable JPEG XL containers. Conformance tests cover every depth `1..=16`, the
1/255/256/257 group boundaries, and extreme aspect ratios. A streamed 16,384×1 RGB8 case is exact
through both the published Rust `jxl` decoder and reference `djxl`, with identical blocking and
runtime-neutral Future codestreams. Browser/WASM compilation covers that same multi-batch state
machine; browser execution still requires a WebGPU-capable page and executor/event-loop integration
provided by the application.

## Experimental VarDCT profile

`VarDctEncoder::new` takes an explicit `VarDctStrategy` and accepts one padded, interleaved sRGB8
image whose extent equals that transform. All 27 standard strategies are executable end to end,
from the 8×8-footprint strategies through the regular 16/32/64/128/256 square and rectangular
families. `VarDctStrategy::EXECUTABLE` is the authoritative inventory, and every entry emits its
exact standard identifier rather than being relabeled or lowered to DCT8.

The GPU executes sRGB linearization, XYB conversion, LF quantization, the per-8×8 clamped-gradient
DC predictor, signed tokenization, prefix packing, histogramming, and construction of the standard
strategy map. Through 32×32, one workgroup also records the complete diagnostic forward transform.
Its built-in workgroup is 256 lanes with 16 KiB of fixed shared XYB storage. Larger strategies
dispatch one 64-lane workgroup per 8×8 block (1,024 bytes of workgroup storage)
and then end that compute pass before a one-invocation control pass, making all DC writes visible
before deterministic prediction and entropy serialization. The host validates the complete typed
artifact, including status, live counts, orientation, every section offset/length, fragment length,
histogram, tokens, strategy map, and zero padding, before serializing control metadata. This LF-first
distance-25 profile deliberately quantizes every AC coefficient to zero, so it is interoperable but
is not yet a general quality or rate-control implementation.

`TiledVarDctEncoder` extends that same honest LF-only contract to a grid of independent regular
DCT8 transforms. Width and height may reach 2,048, with at least one axis above 256; partial edge
blocks are padded by GPU-side edge replication. The current bound is deliberately one standard
LF/DC group and at least two AC groups. A one-AC-group frame has a normatively fused one-packet
layout that cannot identify this tiled subset, so it returns typed
`UnsupportedFeature::TiledVarDctSingleAcGroup` instead of emitting an ambiguous stream.
Within it, the codestream uses the full `ceil(width / 256) * ceil(height / 256)` AC/pass-group grid,
so 257-pixel and larger axes exercise real multi-packet TOC topology rather than pretending that
the image is one transform. Each 8×8 block is marked as a first DCT8 transform. GPU workgroups
produce the block DC values, one LF-group-local Gradient residual stream, prefix bits, histogram,
and strategy map; the CPU validates and packetizes those artifacts but does not pad pixels,
transform, quantize, predict, or entropy-code them. Multiple LF groups (an axis above 2,048) return
the typed `UnsupportedFeature::TiledVarDctLfGroups` error. The block-product dispatch and all
storage allocations remain independently bounded by the selected device limits.

`VarDctMemoryPlan::kernel_layout` distinguishes fixed, scalable single-transform, and tiled-DCT8
artifacts. Fixed submissions
reserve exactly 51,456 encoder-owned bytes: one 256-byte parameter record, one 25,600-byte artifact,
and one equal-size readback. Scalable artifacts are computed from the live block/sample count and
the maximum fragment bits derived from the actual prefix entries; they range from 2,560 bytes for
64×32 or 32×64 to 50,944 bytes for 256×256, so the largest encoder-owned reservation is 102,144
bytes.
Every section starts on a 256-byte boundary. Parameter/header records are `bytemuck::Pod`, all
arithmetic is checked, and source binding, buffer, workgroup-storage, invocation, and dispatch
limits are validated before submission. Completion supports blocking native use and a
runtime-neutral `Future`, and is deterministic on one device. Actual-GPU tests run every strategy
through the published Rust `jxl` decoder; black is exact and solid-red and gradient fixtures carry
explicit quality guards. `djxl` verifies emitted streams, while `cjxl` serves as a development
quality oracle rather than a strategy-selection oracle. There is no CPU transform, quantization,
residual, entropy, pixel-codec fallback, or compatibility alias.

Contexts created with `WgpuContext::from_backend` inherit that backend's adapter-validated
`KernelPolicy`. Autotune keys `vardct_encode_bounded` and `vardct_encode_quantize` accept
`Scalar`, `Lanes32`, `Lanes64`, `Lanes128`, and `Lanes256`; actual-GPU tests require every choice to
emit the same codestream as the built-in variant. The fixed `serialize_control` pass is deliberately
not tunable because its DC predictor and bit offset are sequential. The lossless Modular token
kernel remains fixed for the same correctness reason until it is replaced by a parallel scan and
compaction algorithm.

```rust,no_run
# use jxl_wgpu_encode::{BufferImageSource, VarDctEncoder, VarDctStrategy, WgpuContext};
# fn encode(
#     context: WgpuContext,
#     source_16_by_8: BufferImageSource,
# ) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let encoder = VarDctEncoder::new(context, VarDctStrategy::Dct8x16)?;
assert_eq!(encoder.strategy().block_extent(), (16, 8));
encoder.encode(source_16_by_8)
# }
```

The tiled API has the same blocking, container, and executor-neutral `Future` completion forms:

```rust,no_run
# use jxl_wgpu_encode::{BufferImageSource, TiledVarDctEncoder, WgpuContext};
# fn encode_tiled(
#     context: WgpuContext,
#     source_768_by_513: BufferImageSource,
# ) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let encoder = TiledVarDctEncoder::new(context)?;
let plan = encoder.memory_plan(&source_768_by_513)?;
let grid = encoder.grid(&source_768_by_513)?;
assert_eq!(plan.kernel_layout, jxl_wgpu_encode::VarDctKernelLayout::TiledDct8);
assert_eq!(grid.ac_group_count()?, 3 * 3);
encoder.encode(source_768_by_513)
# }
```

Actual-GPU conformance covers odd 257×17, asymmetric 513×259, and larger 768×513 inputs. Black is
exact, solid colors and LF gradients have explicit PSNR floors, and both Rust `jxl` and `djxl`
decode the emitted multi-group streams with at most one byte of mutual output disagreement.
`cjxl` provides a separately decoded distance-25 development-quality reference for the same edge
and larger fixtures.

## Animation sessions

`LosslessModularEncoder::begin_animation` writes one standard stream-wide animation header and
keeps a reusable GPU session open for multiple frames. The descriptor fixes the canvas, format,
integer depth, tick rate, loop count, and timecode presence. Each frame supplies an exact duration,
optional timecode, optional signed crop rectangle, color blend contract, one contract per extra
channel, and the two-bit source/destination reference slots. RGBA animation continues to carry
alpha as the standard unassociated extra channel; alpha-weighted `Blend` and `MultiplyAdd` name
that extra channel instead of treating alpha as a fourth color component.

Frame submissions own their GPU work and therefore do not borrow the session. Callers may keep
multiple frames in flight, complete each with blocking `wait` or await the same runtime-neutral
`Future`, and insert completed artifacts in any order. Final assembly restores normative frame
order and rejects duplicates, gaps, or an invalid final-frame flag. All live frame jobs share the
same byte-weighted `MemoryBudget` and buffer pool as still encoding.

```rust,no_run
# use std::num::NonZeroU32;
# use jxl_wgpu_encode::{
#     AnimationHeader, BufferImageSource, FrameBlend, FrameCrop, FrameOptions, FrameTiming,
#     LosslessModularAnimationDescriptor, LosslessModularEncoder, LosslessModularFormat,
#     ReferenceSlot, WgpuContext,
# };
# fn encode_animation(
#     context: WgpuContext,
#     full_frame: BufferImageSource,
#     crop_pixels: BufferImageSource,
# ) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let encoder = LosslessModularEncoder::new(context);
let timing = AnimationHeader::Animation {
    ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
    ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
    num_loops: 0,
    have_timecodes: false,
};
let mut animation = encoder.begin_animation(LosslessModularAnimationDescriptor::new(
    1920,
    1080,
    LosslessModularFormat::Rgba,
    10,
    timing,
)?)?;
let reference = ReferenceSlot::new(1)?;
let first = animation.submit_frame(
    full_frame,
    FrameOptions {
        timing: FrameTiming { duration_ticks: 4, timecode: None },
        save_as_reference: reference,
        ..FrameOptions::default()
    },
)?;
let crop = animation.submit_last_frame(
    crop_pixels,
    FrameOptions {
        timing: FrameTiming { duration_ticks: 4, timecode: None },
        crop: Some(FrameCrop::new(320, 180, 640, 360)?),
        color_blend: FrameBlend { source_reference: reference, ..FrameBlend::default() },
        ..FrameOptions::default()
    },
)?;
animation.insert(first.wait()?)?;
animation.insert(crop.wait()?)?;
animation.finish_container()
# }
```

The conformance suite exercises full-frame Replace, cropped Add, reference-slot persistence, RGBA
alpha-weighted Blend, mixed blocking/Future completion, and out-of-order completion. Every
displayed frame is compared exactly with both published Rust `jxl` and reference `djxl`.
