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
- LF global always carries a valid shared Modular tree and entropy code; LF groups and HF global
  are empty. `LosslessModularTreeMode::SharedGlobal` makes each PassGroup select that descriptor.
  `LocalPerGroup` instead writes a complete standards-compliant MA/entropy configuration after
  every pass-group header. The current local mode repeats the frame-trained codes, providing a
  deterministic interoperable policy and decoder/conformance input without pretending to perform
  independent per-group tree learning. A streamed 16K×1 RGB8 test exercises this mode across
  multiple bounded artifact batches through blocking and runtime-neutral completion.

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
#     BufferImageSource, LosslessModularEncoder, LosslessModularFormat,
#     LosslessModularTreeMode, WgpuContext,
# };
# fn submit(
#     context: WgpuContext,
#     source: BufferImageSource,
# ) -> Result<(), jxl_wgpu_encode::EncodeError> {
let encoder = LosslessModularEncoder::with_tree_mode(
    context,
    LosslessModularTreeMode::LocalPerGroup,
);
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
families. `VarDctStrategy` re-exports the shared protocol `TransformKind`; `ALL` enumerates the
standard alphabet and `pixel_extent()` supplies its spatial geometry. Every entry emits its
exact standard identifier and real AC coefficients.

`VarDctEncoder::new_with_strategy_map` accepts a `VarDctStrategyMap` for the entire image.
Its `VarDctTransform` placements use 8×8-block coordinates. Construction sorts placements into
raster order and rejects holes, overlaps, out-of-grid rectangles and AC-group crossings.
Any standard strategy may appear at an unaligned block origin when its rectangle fits one
256-pixel AC group. The padded grid is `ceil(width/8) × ceil(height/8)`; the GPU replicates
partial source edges for every strategy. Dimensions share the checked 16K per-axis bound.
The map is caller-selected metadata; content-adaptive strategy selection remains unimplemented.

`VarDctEncoder::new_with_config`, `new_with_strategy_map`, and
`TiledVarDctEncoder::new_with_config` accept a `VarDctConfig`. Its
`lf_metadata` field holds validated `VarDctLfMetadata`. Its LF dequantization and base-correlation fields retain exact finite
binary16 values, while the colour factor and signed LF factors use their normative integer
domains. Construction rejects dequantized coefficients below libjxl's `1e-8` threshold, colour
factors outside `2..=65793`, and base correlations outside `[-4, 4]` with typed `EncodeError`
variants. Default and explicit bundles share one serializer, and both single-transform and tiled GPU
kernels subtract the selected LF chroma-from-luma slopes and quantize with the selected channel
dequantization multipliers. Generated explicit-metadata streams are parsed back by the stock
frontend and agree across Rust `jxl`, the stock GPU decoder, and optional `djxl` within one RGB8
code; blocking and runtime-neutral Future assembly are identical.

The GPU executes sRGB linearization, XYB conversion, forward transforms, LF/AC quantization, the per-8×8
clamped-Gradient DC predictor, signed tokenization, prefix packing, histogramming, and the
standard strategy map. All 27 strategies and `TiledVarDctEncoder` use default dequantization
matrices and natural coefficient orders, with one prefix distribution
for all 495 coefficient contexts and no LZ77. `VarDctQuantization` validates exact global scale
`1..=73728`, LF quantizer `1..=65536`, and a default `VarDctHfMultiplier` in `1..=256`.
`VarDctTransform::with_hf_multiplier` overrides the default for that transform; sorting a map
preserves its associated multiplier. GPU quantization and serialized LF/HF metadata use these
same values. Defaults are `(8813, 10, 6)` and carry no perceptual-distance claim. The former
`PerceptualDistance` API was removed; general distance/quality guarantees, adaptive selection,
and rate control remain unimplemented.

The LF and AC streams use 33-symbol raw prefix alphabets, covering every signed 32-bit value.
The global MA tree is one Gradient leaf with no LZ77. Prefix bits retain all 15 canonical bits;
Modular lossless retains its separate raw-plus-LZ77 policy. Quantization multiplies scalar
controls in floating point before conversion and reports `BackendError::VarDctQuantizationOverflow`
instead of saturating or clipping coefficients. Effective HF multipliers are limited to 256:
JPEG XL decoders clamp each signed HF metadata sample to `0..=255` before adding one. The GPU
decoder now performs the same clamp for negative and oversized source samples.

Single transforms and image-wide maps share the `ForwardVarDctPipeline` batch path: regular DCTs run separable horizontal
and vertical passes, while special 8×8 transforms evaluate a constant basis on GPU. A final pass
extracts LF from the transform's lowest-frequency rectangle using the normative resampling
factors and inverse small DCT. This replaces the old large-transform approximation by independent
8×8 means. Raw coefficients, LF and quantized coefficients remain GPU-resident. The host expands
only validated placement metadata and bounded strategy constants, dequantization matrices and coefficient orders, shared with the
decoder; it never evaluates image samples.
The decoder's direct dependencies on `jxl-vardct`, `jxl-threadpool` and `jxl-oxide-common` now
serve development oracles. Production defaults no longer instantiate a CPU decoder's matrix
parser; common metadata dependencies may still use the latter two transitively.

`TiledVarDctEncoder` accepts nonzero RGB8 dimensions through the checked 16,384-pixel per-axis
bound. Partial edge blocks replicate the final source row/column on GPU. A single AC group uses
the standard fused packet, including tiny and odd images; larger images carry every
`ceil(width / 256) * ceil(height / 256)` AC group and
`ceil(width / 2048) * ceil(height / 2048)` LF group. Each block is an independent DCT8 transform.
The first pass dispatches a two-dimensional block grid, with 64 lanes by default. Each workgroup
uses 2,048 bytes for 64 XYB pixels and 64 quantized AC vectors, plus a four-byte quantization error flag. Coefficients stay in shared
memory and are immediately packed into one word-aligned block fragment; adjacent workgroups never
write the same storage word. The second pass predicts and packs DC, resetting Gradient at LF-group
boundaries and writing a checked descriptor per LF group. Ending the first compute pass is the
global visibility boundary before the control pass publishes the completed artifact.

The host validates status, every layout field, live counts, DC residuals/histogram, AC counts and
coefficient ranges, exact fragment consumption, and zero padding. It appends the GPU-owned block
bits in AC-group raster order (Y, X, B inside each block), with byte alignment only at packet ends.
There is no host transform, quantization, source padding, coefficient re-encoding, or pixel-codec
fallback. The independently concatenable block format relies on the single-distribution prefix
policy; future contextual or ANS encoders must maintain their state on GPU.

`VarDctMemoryPlan::kernel_layout` distinguishes `SingleTransform`, `StrategyMap` and `TiledDct8`. All use
768-byte parameters and a runtime-sized artifact with a 272-byte header. LF descriptors
follow the header; the subsequent strategy, sample and entropy sections align to 256 bytes. Single-transform plans additionally report exact XYB, raw coefficient,
LF, quantized coefficient, matrix/order, transform-task and forward scratch allocations in `transform`.
Mapped plans report their aggregate allocation sizes: basis/uniform storage is shared per strategy,
while all transforms share image-wide XYB/coefficient/LF/quantized arenas. Each strategy batch
owns one horizontal scratch allocation and a 20-byte forward task per transform; encoder tasks
occupy 44 bytes per transform. No GPU allocation is created per individual transform.
Sources use channel origins within one complete XYB binding, so small maps also work on devices
requiring 1024-byte storage offsets without padding each channel allocation.
An 8×8 DCT submission owns 11,668 bytes: 768 parameters, 3,072 artifact, 3,072 readback and
4,756 resident transform bytes. It no longer allocates a fixed diagnostic coefficient readback.
Single transforms reserve one AC slot for three counts and at most `area - area / 64` coefficients
per channel; the largest 256×256 slot has 217,730 words. Mixed maps reserve the exact
strategy-specific bound per transform, with one length word each and no maximum-size slot
for smaller transforms. Tiled artifacts add one length word and a
214-word AC slot per block, with each section aligned to 256 bytes. The slot bound comes from
the actual prefix lengths for three counts and at most 63 signed coefficients per channel.
The complete parameter + artifact + readback + resident transform reservation remains live through validation or
abandoned-job cleanup; caller-owned source bytes are reported separately. Source binding,
artifact and transform bindings, buffer size, workgroup storage, invocation count, and per-axis dispatch limits
are checked before submission. A full 16K square therefore also depends on adapter and budget
capacity.

Actual-GPU tests compare emitted streams with Rust `jxl`, installed `djxl`, and the stock GPU
decoder. All 27 strategies run textured RGB8 inputs with default and custom correlation: each AC
coefficient is checked against independent f64 transforms and pinned native basis/matrix/order
data, and all 54 streams agree across the three decoders within one RGB8 code. The shared
forward primitive separately checks 667 native coefficient/LF cases, including complete impulse
bases for all ten strategies with an 8×8 footprint. See the
[native fixture generator](../jxl_wgpu/test-data/forward_vardct_generator/README.md).
Procedural checkerboards, stripes, impulses, gradients, and colour patterns also exercise
single-packet images, AC/LF boundaries, custom correlation, and a 2057×2057 four-LF-group image.
Mixed-map cases cover all 27 strategies in one 512×512 image, a 2057×17 LF-boundary image,
and 13×21 non-DCT8 edge replication, including native coefficient checks and all three decoders.
The batched forward primitive separately checks disjoint/reordered source and output ranges for
all 27 strategies, with poisoned gaps and two transforms per batch under all five variants.
An independent f64 cosine-sum reference checks AC values within one integer quantizer step;
this is a numerical regression bound, not ISO precision or perceptual-quality certification.
Blocking/Future assembly and all supported linear workgroup variants produce identical bytes.
The suite also rejects malformed or missing GPU AC output, checks an insufficient device binding,
and tests exact budgets, one-byte backpressure, abandoned completion, and successful reuse.

Contexts created with `WgpuContext::from_backend` inherit that backend's adapter-validated
`KernelPolicy`. Autotune keys `vardct_encode_forward` and `vardct_encode_quantize` accept
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
assert_eq!(encoder.strategy_map().extent().width, 16);
assert_eq!(encoder.strategy_map().extent().height, 8);
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

Actual-GPU conformance covers odd 257×17, asymmetric 513×259 and 768×513, horizontal 2056×256 and
vertical 256×2056 two-LF-group inputs, plus exact-black 16384×1 and 1×16384 panoramas. Rust `jxl`
and `djxl` decode the emitted multi-group streams with at most one byte of mutual output
disagreement. The two-LF-group streams also execute through the stock GPU decoder and explicit
readback within one code of Rust `jxl`. `cjxl` provides a separately decoded distance-25
development-quality reference for the edge and two-LF-group fixtures.

```rust,no_run
# use jxl_wgpu_encode::{BufferImageSource, VarDctEncoder, VarDctStrategy, VarDctStrategyMap, VarDctTransform, WgpuContext};
# fn encode(context: WgpuContext, source_13_by_21: BufferImageSource) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let map = VarDctStrategyMap::new(13, 21, vec![
    VarDctTransform::new(0, 0, VarDctStrategy::Dct16x16),
    VarDctTransform::new(0, 2, VarDctStrategy::Dct8x16),
])?;
let encoder = VarDctEncoder::new_with_strategy_map(context, map, Default::default())?;
encoder.encode(source_13_by_21)
# }
```

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
