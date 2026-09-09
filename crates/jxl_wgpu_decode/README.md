# jxl_wgpu_decode

GPU-required JPEG XL decode orchestration. Production execution uses the stock WGSL engines and has
no dependency on the published `jxl` decoder. The complete-format backlog and acceptance gates are
tracked in [`FULL_JPEG_XL_ROADMAP.md`](../../docs/FULL_JPEG_XL_ROADMAP.md).

## Executable profile

`GpuDecoder::wgpu` supports independent full-canvas Replace animations and layered stills, including
mixed Modular/VarDCT presentations and recursive progressive DC. It uses the shared frame executor
described below.

`GpuOutputRequest::with_image_selection(ImageSelection::Preview)` selects the embedded preview.
The default is `ImageSelection::Main`. Both use the ordinary GPU session and output APIs:

```rust,no_run
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, ImageSelection, vardct_rgb8_format};

# fn example(backend: jxl_wgpu::WgpuBackend, encoded: &[u8]) -> jxl_wgpu_decode::Result<()> {
let decoder = GpuDecoder::wgpu(backend)?;
let request = GpuOutputRequest::color(vardct_rgb8_format())?
    .with_image_selection(ImageSelection::Preview);
let mut preview = decoder.open(encoded, request)?;
// preview.metadata().extent is the preview's own oriented extent.
let frame = preview.next_frame()?.expect("one preview presentation");
# drop(frame);
# Ok(())
# }
```

The frontend creates a `SelectedImageInventory` once, before the `GpuSubmissionEngine::open`
boundary. `source_inventory()` returns `ImageSourceInventory::Complete` or `PreviewPrefix`,
preserving original metadata and distinguishing complete transport/header inventory from a completely
received preview. `complete_inventory()` returns `Some` only for the former.
`reconstruction_inventory()` contains only the selected image domain. A preview becomes one final still with its own
dimensions and no intrinsic-size override, duration or animation timecode. An encoded
`is_last=false` preview still ends after exactly one physical frame. Main selection excludes
that frame and retains all main LF/reference dependencies and presentation timing.
`FrameExecutionPlan` consumes the reconstruction inventory and rejects an unselected preview
inventory with `FramePlanError::ImageNotSelected`.

Physical IDs, entropy ranges and noise seeds are never renumbered during selection or producer
projection. `CodestreamInventory::frame_position` resolves IDs to local vector positions.
The preview uses noise seed `[0, 1]`; leading nonvisible main frames continue that counter,
and the first visible main frame advances to `[1, 0]`. Reference and LF image state remain
separate across the boundary. This matters for a preview followed by noisy progressive DC.

All eight orientation values, Apply/Keep, supported color and scalar-extra delivery, and leased
output ownership apply to both selections. A missing preview returns
`Error::ImageSelection(ImageSelectionError::MissingPreview)` before engine admission.
Whole `open` and fragmented `stream(...).finish()` both require complete transport and frame
inventory; a preview alone does not validate an incomplete main image.

For early delivery, feed the same stream and call `take_preview` after `is_preview_ready()`:

```rust,no_run
use jxl_gpu_bitstream::ContainerStreamEvent;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, vardct_rgb8_format};

# async fn example(
#     backend: jxl_wgpu::WgpuBackend,
#     events: impl IntoIterator<Item = ContainerStreamEvent>,
# ) -> jxl_wgpu_decode::Result<()> {
let decoder = GpuDecoder::wgpu(backend)?;
let output = GpuOutputRequest::color(vardct_rgb8_format())?;
let mut stream = decoder.stream(output.clone())?;
let mut displayed_preview = None;
for event in events {
    stream.push_transport_event(&event)?;
    if stream.is_preview_ready() {
        let mut preview = stream.take_preview(output.clone())?.expect("ready preview");
        displayed_preview = preview.next_frame_async().await?;
        // The GPU frame can be displayed while the same stream receives main input.
    }
}
let mut main = stream.finish()?; // Requires the transport scanner's authoritative End.
let main_frame = main.next_frame_async().await?;
# drop((main_frame, displayed_preview));
# Ok(())
# }
```

`take_preview` returns `None` until its complete header/TOC and every section byte have arrived.
A known absent preview returns `MissingPreview`; a second successful take returns
`PreviewAlreadyTaken`. Opening errors leave the stream and take state unchanged. The method
selects Preview and leaves the original stream request intact, so orientation, color, numeric
extra-channel selection and frame leases are independent. Entropy validation occurs in the GPU
session; malformed later main input or transport cannot invalidate a previously validated preview.

A prefix keeps original byte offsets and ends exactly after the preview's final section.
`GpuCodestream::is_container()` is `None` for that prefix and `Some(bool)` after authoritative
transport completion. Each admitted transport range owns one shared immutable budget token;
preview and main never double-charge shared ranges, and preview never retains subsequently
received ranges. A final range crossing into main keeps its entire original charge while shared.
The budget counts logical retained ranges, not allocator capacity or unrelated bytes within a
caller allocation. `IncrementalInputBudget::with_limits` bounds both bytes and span count;
`new` defaults to 1,048,576 spans. Either admission failure leaves its input event retryable.

`tests/preview.rs` checks 48 reproducible streams against native libjxl's preview API and
independent main-image controls, with byte-identical whole/bounded output. It includes all
16 dimension encodings in both modes, alpha, JPEG sampling, floating samples, resampling,
non-final preview headers, main animations and recursive LF, syntax/entropy failures,
one-byte inventory delivery, admission retry and cancellation. All 48 streams also produce GPU
preview before any main frame bytes arrive, then finish every main presentation with the preview
lease still live. `tests/stream_preview.rs` audits every raw/jxlc/jxlp two-chunk split, auxiliary
payload preservation, prefix clipping, delayed takes, admission retry and independent source lifetimes.
`cargo run -p jxl_wgpu_decode --example regenerate_previews` reproduces the corpus with offline
libjxl 0.12 tools. No production CPU pixel codec or new shader ABI is introduced.

### Intermediate LF and pass images

`GpuOutputRequest::with_progressive_output(true)` enables DC and intermediate AC-pass images for
VarDCT color output with optional extra channels, including deferred descriptors, animations and
composed presentations. Each presentation validates all overwritten/hidden producers before publishing
refinements from its final Regular physical frame. The compositor blends each immutable producer
snapshot against the committed reference versions and then applies output color conversion and
orientation. Only complete physical frames update reference slots. Modular color frames and
SkipProgressive frames continue to return final images. Color-only LF-dependent presentations,
including composed animations, can additionally publish their completed Modular/VarDCT LF dependencies.
Input must still be complete for `open` or `stream(...).finish()`; this is independent of early
embedded-preview delivery.

```rust,no_run
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, vardct_rgb8_format};

# async fn example(backend: jxl_wgpu::WgpuBackend, encoded: &[u8]) -> jxl_wgpu_decode::Result<()> {
let decoder = GpuDecoder::wgpu(backend)?;
let request = GpuOutputRequest::color(vardct_rgb8_format())?
    .with_progressive_output(true);
let mut session = decoder.open(encoded, request)?;
while let Some(image) = session.next_update_async().await? {
    // Present image.output(). Intermediate and final storage have the same full canvas extent.
    if let Some(progress) = image.progression() {
        // FrameProgression::Coefficients carries completed/total passes (zero means DC).
        // FrameProgression::LowFrequency identifies a complete physical LF frame and its level.
        // progress.intended_downsampling() describes detail, not buffer dimensions.
    }
    // image.is_complete() distinguishes the final reconstruction from a refinement.
}
# Ok(())
# }
```

`next_update`, `poll_next_update` and `next_update_async` retain the pending frame after a
refinement. All updates share its exact presentation metadata and one logical frame slot; each
immutable output owns its GPU byte reservation. Only the final update advances frame number,
animation time and session completion. `next_frame` and its async counterpart continue to return
only final images. Custom engines can implement `GpuPendingFrame::poll_next_update`; its default
adapts their existing final-only completion.

Pass barriers span every LF group in both whole-buffer and bounded-window execution. Each update
captures independent packet/artifact/coefficient validation evidence, validates only completed
passes, and remains valid if a later pass fails. Spectral, quantized and multiple-LF-group fixtures
match native libjxl flushes within one RGB8 code; all final GPU bytes match final-only decoding.
DC uses the image-header 8× kernel over the complete LF atlas, including MCU padding and
cross-group neighbors, followed by the normal restoration/resampling/color pipeline. Its detail
ratio is 8 and its output has the same full canvas extent as the final image. Only packet/artifact
evidence is needed at this stage. Deferred HF descriptors and raw matrices are parsed/executed
after returning DC; later errors preserve that image.

LF publication reconstructs the exact dependency version used by the presentation, after the LF
producer's restoration and frame resampling. Each level applies the image-header 8× kernel in XYB,
clips to that level's exact grid, and only then converts color and applies output orientation.
`LowFrequency { physical_frame_index, level }` carries intended detail `8^level`; it does not invent
an unfinished coefficient pass for a completed Modular or VarDCT frame. Unused and overwritten LF
versions still validate but do not become presentation updates. Reused LF slots are not decoded or
published again in later presentations. Composed LF updates use the terminal physical layer's
geometry and color encoding, then read its exact committed background reference before output
conversion and orientation. When a later hidden layer supplies that reference, the executor queues
LF planes until the hidden layer validates; a failed reference cannot produce an LF image. The
queue retains its own plane leases even if a predictor slot has expired. After each LF publication,
the next physical producer is admitted on a later poll. Native and poll/async final-only completion
discard unsubmitted LF updates and drain already submitted work. Source planes, render scratch
and packed output retain independent byte ownership through GPU completion and cancellation.

Level 1 output matches the Rust decoder's LF flush, while native libjxl validates the following
DC/AC and final images. Full recursive codestream coverage currently reaches LF2; a separate
scalar-oracle adapter test validates expansion through LF4, custom weights and odd/one-sample axes.
Seven animation families cover RGB/gray, mixed JPEG/Modular, recursive LF, negative/off-canvas
crops and reference blends with exact clocks and physical IDs. Apply/Keep orientation and whole/
40-byte fragmented input return identical immutable updates; finals equal final-only decoding.
Native libjxl cannot flush blended layers, so test-only standalone headers preserve every original
entropy byte and decoding field; independent F64 composition of its layer flushes is checked
against native coalesced finals before comparing GPU updates. Composed LF1 images additionally use
the Rust decoder's standalone LF flush and the same independent scalar composition, including a
hidden reference decoded between LF and its visible consumer. The maximum normalized linear LF1
error is below 0.000285 under the existing 0.001 composition regression bound. LF2 has renderer and
lifetime evidence without a native per-level pixel-oracle claim. No production CPU codec is used.

Extra-channel snapshots decode and validate each pass's Modular subimages before copying the
assembled channels for global inverse transforms, normalization and resampling. Future passes
continue filling the original arena. Integer and floating alpha, spots, independent channel depths
and associated-alpha composition share the ordinary final renderer. Each snapshot owns its output,
inverse arena, uniforms, normalization buffers and byte permits. Whole and 40-byte fragmented
input produce identical images; errors and cancellation preserve already returned output leases.
libjxl disables progression events with extras, so tests flush logical codestream prefixes at
physical pass boundaries; no CPU codec is added to production. Nine additional standalone header
fragments provide independent alpha-composition oracles without duplicating entropy.

The remaining progressive work includes broader LF conformance, LF previews with extra channels,
numeric/Modular refinements, selective regions, and incomplete-frame input readiness.
Native-comparison precision remains a separate conformance gate.

The low-level `WgpuSubmissionEngine` implements a standards-only Modular still profile:

- a raw codestream, ordinary `jxlc` container, or reconstructed `jxlp` container with no private
  metadata requirement;
- one final still frame with Gray or RGB Modular samples and arbitrary extra planes, each with
  independent 1–31-bit integer or legal JPEG XL floating precision, including 2×/4×/8× color/extra resampling, over a bounded
  128/256/512/1024-pixel pass-group grid and one through three passes;
- bounded DC-global, LF-group-local, or pass-group-local MA trees with all JPEG XL Modular predictors, including
  weighted self-correcting prediction, leaf offsets/multipliers/context selection, Prefix or ANS
  entropy, hybrid integers, context maps, and the standard LZ77 distance alphabet;
- shared DC-global RCT/Palette/Squeeze and per-LF/pass-subimage RCT/Palette/Squeeze stacks, including
  nonempty DC-global sample channels and group-edge geometry; straight and associated alpha are accepted,
  and restoration filters and references remain outside this low-level still engine.

Presentation normalizes all eight image orientations in the GPU writer. The source canvas and
group origins remain in codestream coordinates; output layout and changed regions use the oriented
extent. The same forward/inverse WGSL coordinate helpers serve Modular and VarDCT. Fixed-Gradient
direct Gray8 output, ordinary reconstruction, group-local inverse stacks, and the frame-wide
Palette/Squeeze finalizer all apply the mapping without an intermediate image or extra submission.
Rotated or mirrored groups use atomic byte writes because transformed group boundaries may share
a storage word. Packed 4:2:2 writes each pixel's luma and neutral-chroma bytes independently and
replicates the final odd pixel; a group no longer has to own an entire packed pair.

`GpuOutputRequest::with_orientation_policy(OrientationPolicy::Keep)` retains codestream coordinates
and unrotated extents. The default is `Apply`. This choice also controls animation metadata and
presentation buffers across Modular, VarDCT, mixed sequences and recursive DC; physical dependency
ordering and frame timing remain unchanged. `FrameExecutionPlan::negotiate_with_orientation`
exposes the matching plan metadata.

All supported 1–31-bit Gray/RGB/RGBA Modular sources can also return F32 color through
`PixelFormat::rgb_f32`, in planar or interleaved RGB/BGR/RGBA/BGRA order. Integer samples normalize
by their actual bit depth after inverse transforms; gray expands to RGB, missing alpha is one,
and decoded alpha normalizes independently of the RGB transfer. This path currently accepts
explicit full-range BT.709 primaries and sRGB/SYCC, Linear, BT.709 or BT.2020 transfer functions.
It preserves the existing native integer output contracts and uses the same resident output leases.
Gray+alpha and independent alpha precision use the same path. Floating sources decode their
declared representation before color conversion. Additional source color domains remain unsupported.

### Integer source samples

All 1–31-bit primary and extra-channel declarations are admitted by Modular and XYB/original-sRGB VarDCT.
`native_modular_pixel_format(ModularChannels::Gray, bits)` creates a canonical scalar layout for
`NumericSampleMapping::NativeUnsigned` from Modular grayscale or a selected extra in either mode;
RGB/RGBA layouts can be passed to `GpuOutputRequest::color`. General VarDCT numeric color-channel
delivery remains a separate incomplete output feature.
Valid depths 1–8, 9–16 and 17–31 use 8-, 16- and 32-bit words, respectively, with zero high padding.
The encoder's current 1–16-bit input limit is independent of this decoder delivery API.

Unfiltered integer planes retain their exact codes through entropy, prediction and inverse
transforms. Independent integer alpha precision is rescaled using a two-word product/division
on GPU. `NormalizedUnsigned` and F32 color divide by the declared maximum; wide-source F32
output agrees with the independent libjxl normalization within one ULP. Native scalar working
values outside the unsigned range return a typed error, while F32 retains negative/overshoot values.

Resampling, spots, alpha association and frame composition operate in decoded F32. Native
presentation rounds that F32 value against the exact requested integer maximum using integer
significand arithmetic, including 31-bit endpoints and half-code boundaries. This avoids an
additional rounding loss from F32 multiplication; it does not recover precision lost in earlier
filtering, blending or VarDCT reconstruction. Wide-source RGB8 and other converted color formats
use the common accounted F32 presentation surface. Non-XYB YCbCr remains an 8-bit source profile.

`tests/integer_samples.rs` checks 42 exact-source fixtures and 40 libjxl rendering cases, including
all source precisions, large predictor residuals, real RCT/Squeeze, independently coded alpha,
orientations, 2×/4×/8× reconstruction, progressive DC and animation. Whole and bounded fragmented
output must agree. `cargo run -p jxl_wgpu_decode --example regenerate_integer` reproduces the corpus;
its documented header generation covers legal precisions beyond libjxl's public encoder limit.

### Floating source samples

`DecodeProfile::{Modular, VarDct}` and `StandardVarDctProfile` expose `sample_bit_depth: SampleBitDepth`
instead of an integer-only width. All 154 legal floating declarations (2–8 exponent and 2–23 mantissa
bits) are admitted. The Modular inverse arena remains signed working words; original precision is
stored separately from transform geometry in a validated `ModularOutputPlane`.
`ModularSampleDomain::Encoded` identifies those original sample words; `DecodedF32` identifies
converted values, including filtered planes. Each extra retains its own integer or floating type.

Use `NumericSampleMapping::NativeFloat` with scalar F32 storage for Modular grayscale or a selected
floating extra in either coding mode. Integer sources continue to use `NativeUnsigned` or
`NormalizedUnsigned`; mismatched mappings are rejected. Binary16, binary32 and every custom precision
are widened using integer bit assembly on GPU. Unfiltered, uncomposed F32 samples preserve signed
zero, subnormals, infinities and NaN payloads; no-op RGB F32 delivery also copies bits directly.
Filtering, color conversion, alpha and blending operate on decoded values and make no payload or
bit-exact arithmetic guarantee. Native scalar floating output never divides by an integer maximum.

RGB integer and broader color presentation use the common planar F32 boundary, with quantization
at final output. The same domain supports mixed extra channels, first-alpha selection, associated
alpha, spot inks, Squeeze/Palette/RCT, group distribution, orientation, resampling and animation.
Floating VarDCT source precision describes the original image; it does not reinterpret reconstructed
XYB coefficients or the integer Modular words used by progressive-DC dependencies.

`tests/floating_samples.rs` checks all 154 precisions against independently decoded libjxl words,
plus 27 rendering fixtures including five nine-layer animations and a real progressive-DC dependency.
Whole and 256-byte-window fragmented async output agree exactly, and reservations return to zero.
`cargo run -p jxl_wgpu_decode --example regenerate_floating` reproduces the corpus using offline
libjxl 0.12 tools; the production crates do not link that codec. Original non-sRGB/ICC domains,
pre-transform references and source integer precision above 16 remain separate requirements.

Codestream topology is separate from native pixel formats: `DecodeProfile::Modular`
contains `ModularChannelCounts`, with `color_count()`, `extra_count()` and total `count()`.
`ModularChannels` describes native Gray/RGB/RGBA output arrangements. `AnimationMetadata::extra_channels`
preserves declaration order, names, original depths and type-specific metadata. Color output uses
the first alpha declaration regardless of its position among other extras; native RGBA rescales
alpha to the output depth, while F32 normalizes it by its own depth. Missing alpha is opaque.

`GpuOutputRequest::with_extra_channel(index)` selects a zero-based extra-channel declaration.
Use a canonical native unsigned Gray format at that channel's depth for integer codes,
or a scalar F32 descriptor with `NumericSampleMapping::NormalizedUnsigned` for codes divided by
that channel's unsigned maximum. Scalar data receives orientation but no color transfer:

```rust,ignore
let request = GpuOutputRequest::numeric(
    PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
    NumericSampleMapping::NormalizedUnsigned,
)?.with_extra_channel(2)?;
```

Frame inventory resolves each extra's effective factor, including `dimension_shift`, and validates
that it is 1, 2, 4 or 8 and at least the color factor. Both producers reconstruct channels on their
own grids. The common GPU render stage normalizes selected integer planes before the normative
5×5 interpolation, using standard or custom image-header weights. A frame arena retains neighbors
across Modular group boundaries. Color and extra factors may differ; all three factors also work
for Modular color. F32 preserves interpolation fractions and signed working ranges. Native output
rounds after resampling at the requested depth; unresampled native codes stay exact.

`WgpuDecodeMemoryStats::modular_render_bytes` and
`VarDctDecodeMemoryStats::extra_render_bytes` include output planes, shared normalization scratch,
deduplicated weights and uniforms. These transient resources are admitted before submission and
retained through cancellation callbacks. `DecodeProfile::Modular` replaces `ModularLossless` so
the coding-mode name does not imply that resampled reconstruction is lossless.


All source channels still pass through GPU entropy decoding and inverse transforms before output
selection. Selected resampled planes normalize and interpolate in resident F32 storage. Unknown
non-optional extras cannot silently be omitted from color interpretation. Spot-color data can be
selected like other extras; `with_spot_color_policy(SpotColorPolicy::Preserve)` explicitly returns
the base color. Default `Render` presents all spots in declaration order on the GPU through
`GpuDecoder::wgpu`, including single-frame images and post-transform composition. Spot rendering
uses the retained normalized plane after extra-channel resampling/blending and precedes requested
color conversion, alpha association, orientation-aware packing and target chroma subsampling.
References retain untinted color and every extra plane. Each mix is
`solidity * sample * ink_rgb + (1 - solidity * sample) * rgb`, without intermediate clamping.
Six checked-in libjxl fixtures cover eight extra-channel types, multiple alpha planes, a Gray+alpha
topology, independent depths, a one-leaf MA tree, and transformed multi-group streams. Native
planes match source codes exactly, and F32 matches Rust `jxl` and the optional libjxl C oracle.

`GpuOutputRequest::with_alpha_output_policy(AlphaOutputPolicy::Unassociated)` is the default for
color output. `Preserve` keeps the first alpha declaration's association; `Associated` requests
premultiplied color. RGB conversion runs first, then association changes, then integer rounding or
F32 packing. The same policy applies when RGB output omits the alpha component. Unpremultiplication
uses `max(alpha, 2^-26)` as its denominator; multiplication uses the same floor. Alpha itself is
unchanged, and F32 retains finite negative, extended and invisible color values. Integer color
output clips only at final packing. Numeric mappings and selected extra channels ignore this policy
and retain their existing exact-code/normalization contracts. Metadata always describes the source.

Composition keeps unrounded original-encoding RGB and every independently normalized extra plane
in one GPU allocation per reference slot. Associated source-over computes
`top + bottom * (1 - top_alpha)`; unassociated source-over
retains its alpha-weighted normalization. Conversion requested by the caller occurs only after all
layers contributing to a presentation, including when RGB and alpha use different background slots.
Fourteen associated still fixtures cover equal/independent depths, Gray+alpha, a first alpha after
other extras, zero alpha with nonzero color, one-pixel axes, shifted resampling, multi-group Squeeze
and progressive AC. Three nine-layer sequences add all five blend modes, extended reference values,
crops and slot replacement in both coding modes. Native RGB/RGBA, planar/interleaved F32, Apply/Keep,
NV12, odd-width YUYV/UYVY and BT.2020 constant-luminance P010 exercise final packing under whole and bounded input.

Each extra channel has its own blend mode, reference slot, alpha selector and clamp flag.
Multiple alpha declarations may use different association and depth; color Blend updates its
selected alpha, while presentation uses the first declared alpha. Raw F32 extra output preserves
the composed normalized result, including extended values. Native composed extra output requires
the declaration's depth and clamps/rounds the final result to that unsigned range; the exact-code
contract for unresampled stills is unchanged. Numeric spot selection is unchanged by the spot or
alpha output policy. Seven nine-layer fixtures exercise all nine extra declarations, independent
reference chains, Gray/RGB, both coding modes, distributed groups, shifted resampling and six
presentations. Admission retries and cancellation release every hidden plane and its shared lease.

The private frame surface records its RGB domain explicitly. Unreferenced XYB presentations keep
linear RGB; blending and post-transform reference storage use the original sRGB encoding. Spots
execute in that stage's domain, matching libjxl's pipeline ordering. This avoids a transfer round
trip before standalone VarDCT presentation. Ten deterministic multi-spot stills additionally cover
five inks, independent 1/4/6/10/12-bit coverage, negative/extended RGB and solidity, two alpha planes,
Gray/RGB, oriented thin axes, shifted resampling, responsive distributed transforms, native 8/12/16-bit
RGB/RGBA, F32 RGB/BGRA, NV12, packed 4:2:2 and BT.2020 constant-luminance P010. The spot table costs
32 accounted bytes per ink and exists only for rendered color output. General original color
encodings and HDR luminance mapping remain separate roadmap gates.

Image admission uses the validated inventory's color, depth, and alpha semantics rather than
reparsing a fixed header bit pattern. Enumerated D65 sRGB Gray/RGB, integer extras,
all orientations, intrinsic-size hints, and named channel declarations can use the supported
reconstruction path. Unsupported ICC/color and Modular restoration remain
rejected. Unknown image,
frame, and restoration extension selectors are typed inventory errors before any GPU work.
Twenty-three checked-in libjxl fixtures compare exact native samples with their deterministic
source and Rust jxl, plus exact Gray/RGB samples from djxl. Twelve gray fixtures cover all 30 VPI formats under both whole blocking
and 4 KiB bounded, fragmented async input. Numeric/native results are exact; transformed color
codes differ from the scalar reference by at most one. Both scheduling paths return identical
bytes and release all reservations.

Before submission the decoder inventories the standard image header, frame header, and TOC with
explicit limits. It parses only the bounded DC-global and selected LF-group/pass-group-local MA trees,
histogram descriptors, hybrid integer configuration, context maps, ordered Modular transform
metadata, and pass-group ranges needed to build typed GPU metadata. The progressive schedule maps
each non-LF transformed channel to exactly one pass through the normative
`min(hshift, vshift)`/downsampling brackets; declared empty passes are retained in the public
profile and their physical sections must contain only zero bits. RCT, Palette, and explicit or
default Squeeze are meta-applied
to an exact entropy-visible channel topology without decoding pixels. This accounts odd average/
residual dimensions, channel insertion order, shifts, delta-palette storage, and the meta-channel
prefix under bounded transform/channel/squeeze limits. A portable 32-byte `Pod` descriptor then
proves every packed channel offset fits WGSL `u32` addressing before backend allocation.
For every generalized group, a second 32-byte `Pod` record is appended to the immutable entropy metadata
for each transformed channel. It carries arena offset/stride, dimensions, cumulative decoded sample
range, and an absolute range into a flattened reference-channel list. That list contains only prior
channels whose dimensions and horizontal/vertical shifts match, newest first, as properties 16+
require. A geometry-keyed map bounds construction by channel count times the at-most-60 references
addressable by the 8-bit MA property space. Each self-contained MA/entropy descriptor has its
internal config/tree/table offsets rebased into one immutable GPU word buffer; equal local
descriptors are deduplicated. The 256-byte per-group parameter record exposes independent MA and
channel-descriptor bases. Identical transform plans share one descriptor table even when their MA
or weighted-predictor configuration differs; edge groups retain distinct checked geometry. The
resident arena stride is aligned for dynamic storage offsets.
Inverse planning walks the stack and each Squeeze parameter in reverse while retaining only the
current and immediately restored topology, rather than materializing a channel table for every
transform. The parser also charges the cumulative topology work, so a bounded channel count cannot
be combined with an adversarially quadratic transform sequence.
A standalone `ModularSqueezePipeline` executes one horizontal or vertical inverse parameter without
leaving GPU storage. Average, residual, and destination are checked non-overlapping views of one
read-write arena, which removes storage-binding alias ambiguity and permits later lifetime reuse.
Its portable WGSL emulates the required signed 64-bit smooth-tendency intermediate with two `u32`
words, then applies the specified wrapping `i32` reconstruction. Actual-adapter tests compare odd,
even, one-dimensional, and extreme-value cases to an independent scalar oracle. The stock decoder
feeds transformed entropy output directly into this arena.
Entropy reconstruction and inverse Palette share one implementation of all 14 Modular predictors.
Their averaging, self-correcting predictions, and weighted sums use portable signed 64-bit
intermediates represented by two GPU words. Persistent predictor errors retain the specified
`i32`/`u32` storage, so bounded-input and Palette continuation layouts remain unchanged. Implicit
Palette entries use a wide product for every 1–32-bit working depth; negative delta entries retain
the normative 24-bit scaling cap. Direct GPU tests cover signed extremes, binary32 bit patterns,
all implicit color components, and every predictor against a native-`i64` scalar oracle. This
working-word support also underpins the separate floating representation conversion described above.
A standalone `ModularRctPipeline` applies every one of the 42 normative operation/permutation
combinations in place to three equal-size, non-overlapping views of that same arena. Each invocation
loads all three signed words before writing any permutation, while explicit unsigned add/sub helpers
preserve wrapping `i32` behavior and an explicit bit-pattern shift fixes negative rounding. Its
64-byte, 16-byte-aligned `Pod` uniform and linear Scalar/32/64/128/256 policy variants are validated
before recording. An actual-adapter differential covers all types with odd dimensions, padded
strides, nonzero offsets, and signed extremes.
The reverse planner emits RCT and Squeeze jobs in one exact inverse-order stream. RCT jobs reuse the
three current plane views in place, while a best-fit interval allocator reserves each Squeeze
destination before its dispatch, retires that channel's average and residual immediately afterward,
and merges adjacent free spans. Thus an arbitrary valid RCT/Squeeze composition keeps only live
planes, not every transform generation.
A nested horizontal/vertical actual-adapter test records three ordered dispatches in one encoder and
maps only the final plane; the checked 45-word entropy topology uses a 90-word peak arena and reuses
its first retired range for the restored output. The real progressive-DC LF2 fixture lowers 13
parameters to 37 jobs and three final full-resolution planes within twice its entropy sample count.
An RCT/Squeeze/RCT test emits five ordered jobs and executes them in one command encoder, copying all
three noncontiguous final planes into one staging map. Production scheduling applies a concrete
group-local inverse plan and a 176-byte region-aware finalizer as soon as each group's final entropy
segment finishes when no cross-group transform is present. For DC-global Palette/Squeeze, channels
with both transformed shifts at least three are decoded by LF-group subimages first; channels with
either shift below three are decoded by pass-group subimages. Each subimage finishes its local
inverse plan, then its final planes are copied row-wise into one frame-resident transformed arena
before the scratch lane is reused. One frame-wide inverse plan and one finalizer run after the last
pass group.
Palette continuation state remains GPU-resident.
Every token range and canvas origin comes directly from standard frame sections. It does not
decode a pass-group entropy token, residual, predictor, color transform, or pixel on the CPU.
The Modular metadata reader operates on a checked shared-span bit input rather than indexing one
host slice. The same span table copies exact entropy ranges into the reusable GPU window, including
ranges that cross a chunk or entropy-word boundary, without first joining the codestream. The
contiguous `GpuDecoder::open` creates one span, while `GpuDecoder::stream` accepts borrowed
`ContainerStreamEvent` values and supplies the same engine boundary with an arbitrary checked span
table plus the incrementally produced inventory.

The compute shader reads Prefix or ANS symbols and hybrid integers from bounded codestream windows,
validates every bit and output bound, applies LZ77, walks the MA tree, reconstructs all predictors,
and reverses YCoCg for RGB(A). Up to 512 budget- and device-resolved scratch lanes decode independent
groups in one `dispatch_workgroups` wave; 64 logical group invocations are packed into each portable
compute workgroup and their canvas rectangles do not overlap. Large frames use ordered batches backed
by one reusable stream window instead of binding the full codestream. If any accepted stock Modular
group exceeds the resolved window, it resumes over ordered segments with 16-byte backward and
forward overlap. The common logical cursor, Prefix/ANS state, LZ77 copy state, decoded count, last
value, and first error remain in a 16-byte-aligned 32-byte record at the end of the same scratch
lane. Generic MA adds the Property-8 gradient history for a 48-byte record; a
Weighted/SelfCorrecting consumer also retains four true errors and twelve subprediction-error
accumulators for a 112-byte record. A resume exactly at a channel boundary resets channel-local
predictor state. State and LZ allocation use the maximum requirement across all selected group
descriptors, and mixed Prefix/ANS frames are reported explicitly. Multi-group streams also execute
the DC-global Prefix/ANS stream through the same descriptor kernel and exact termination contract,
whether it reconstructs zero or nonzero samples. One aggregate status staging buffer is mapped once
after the last batch, and its four-word record plus every LF/pass-subimage record is checked before the
frame is reported. No reconstructed sample is produced on the CPU.
Output pipeline selection is also per frame. When every plane offset and row stride is four-byte
aligned and every actual internal group edge ends on a distinct storage word, the shader uses
ordinary word RMW/store. Layouts with an odd stride, offset, or group edge use the atomic byte-safe
pipeline. The proof is performed on the host from the validated output layout and typed group
rectangles; there is no caller hint that can force the non-atomic path.
The validated MA-tree IR receives a second independent specialization proof. If every group's
resolved local/global tree has the same fixed specialization, and every decision is
channel-only and channels 0 through 3 all terminate at Gradient leaves with zero offset and unit
multiplier, the shader lowers the four cluster ids into the parameter record and runs a nested-row
Gradient loop without per-sample MA traversal, coordinate division, unused-neighbor loads, or
predictor dispatch. Any unsupported property, malformed/cyclic route, non-Gradient leaf, or
self-correcting requirement selects the complete generic MA-tree kernel instead.
For an exact U8 `NormalizedGray8` request, the same proof also permits the Gradient loop to emit
each sample directly and omit the separate group-finalization traversal. If descriptor analysis
proves that LZ history is invocation-private, the reconstruction lane retains only the current and
previous rows (512 physical words versus 65,536 logical sample words for a full 256x256 group).
Wider LZ histories, native Modular output, RGB(A), and every other VPI mapping keep
the complete logical reconstruction workspace and finalize through the generic output contract.
`DecodeProfile::Modular` reports the declared pass count and this specialization as
`ModularPredictionProfile::MetaAdaptive` with exact node/decision/leaf counts, maximum depth, and
self-correcting usage; custom synthetic engines use the distinct `Fixed` variant.

For the Modular `WgpuSubmissionEngine`, the complete transform wire grammar and resulting
channel topology are parsed, local MA trees select per-stream metadata bases, and resident
RCT/Palette/Squeeze stacks execute for single- and multi-group streams. Multi-group DC-global
Palette/Squeeze sample data is reconstructed in a frame arena. LF-group streams own exactly the
channels whose horizontal and vertical transformed shifts are both at least three; pass groups own
the remaining channels, including asymmetric shifts. LF streams execute before nonempty pass
streams in pass/group order, all use the same bounded-window executor and aggregate status map, and
one global inverse/finalizer runs after assembly. One through three passes produce a complete final
image; intermediate pass presentation is not yet exposed. Patches, splines and broader
original color profiles remain typed unsupported profiles. The public `GpuDecoder::wgpu` constructs `WgpuDecodeEngine`, inventories
the standard stream once, and selects a producer for each physical frame from
`FrameEncoding`. Callers do not choose or probe a coding mode. Both child engines retain their
mode-specific bindings and pipeline caches while sharing the backend byte budget.

### Frame execution and animation

Lossy Modular color now enters this executor after frame-wide inverse transforms. XYB words are
stored as Y/X/(B-Y); GPU normalization reorders them, restores B in the working integer domain and
applies the LF dequantization multipliers independently of source bit-depth metadata. Original
RGB/gray uses its declared integer or floating sample interpretation. Gaborish and all three EPF
passes operate before color upsampling; extra planes retain their separate normalization and
upsampling paths. Modular EPF uses a validated frame-constant inverse sigma without a sigma image.

`color_output::{ColorOutputPacker, ColorOutputConfig, ColorOutputTransform}` is the common public
RGB/XYB/JPEG color boundary, replacing the former `vardct::output` API. It preserves the stream's
inverse opsin matrix, biases, intensity target and grayscale projection. The Modular finalizer
does not reapply the transfer already performed by this packer. Unreferenced XYB presentations
retain linear RGB so spot colors are applied before transfer conversion; referenced/blended
frames retain original sRGB. All temporary color/filter/upsampling planes and uniforms participate
in `modular_render_bytes` and the backend's shared memory reservation.

`tests/lossy_modular.rs` checks 19 libjxl streams, all delivered extra planes and presentations,
whole versus 256-byte bounded fragmented input, requested RGB8/RGBA8/16-bit quantization and
reservation release. Fifteen stills and the initial Replace presentation of four animations also
match Rust `jxl`; subsequent reference chains use libjxl because of the documented Rust oracle's
clamped-Multiply defect. Enumerated D65 sRGB remains the admitted original color profile; full
color management, pre-transform references, patches and splines remain separate work.

Both XYB and original-sRGB coding modes, plus JPEG YCbCr VarDCT, parse the bounded 80-bit `NoiseModel` and
use the shared `jxl_wgpu::ResidentNoisePipeline` after restoration and frame upsampling, before
color conversion.
`FrameInventory::noise_seed` preserves the visible/nonvisible counters when a physical frame is
projected into a producer. SplitMix64/Xorshift128Plus execute as portable WGSL u32 pairs; no CPU
random image is uploaded. Three random F32 planes and one 96-byte uniform join the same frame
reservation and callback lifetime. An all-zero model skips noise allocation and dispatch.
`tests/noise.rs` covers 37 fixtures and their zero-model variants, including all four Modular
group sizes, all four ordinary JPEG sampling layouts, grayscale, 2×/4×/8× upsampling, rotated
Gray16, F32 RGB and two five-frame sequences with three presentations each. Thirty-four
use Rust jxl and optional live libjxl F32 references. Two custom base/LF correlation fixtures use
checked native linear RGB references, avoiding Rust jxl 0.6's LF-slope noise error and extended-range
sRGB approximations. A single-channel implicit palette case follows H.6.4 and Rust jxl;
libjxl 0.12 incorrectly clamps those indices. Exact whole/256-byte-window output agreement,
admission retry and cancellation cleanup cover both XYB and original color, including the
normalization allocations introduced by nonzero Modular noise. Subsampled VarDCT allocates padded
full-resolution destinations only for shifted components before noise; output conversion derives
its sampling geometry from those actual planes, avoiding a second interpolation. Zero models
retain the fused component-upsampling/output path. `tests/noise_combinations.rs` adds 27 streams:
20 JPEG streams with Gaborish/active EPF 1–3, and seven LF chains with both root encodings,
independent nested models, progressive AC and alpha/depth preservation. Every output is identical
under whole and bounded fragmented delivery; cancellation checks intermediate LF ownership.
A pinned, development-only jxl-oxide oracle and independent scalar Gaborish calculation cover
vertical-subsampling defects in the other references. LF color uses the existing Rust sRGB and
native sRGB/linear tolerances; scalar extras remain exact. See the conformance corpus for the
reference selection and observed errors. Reference-only, patch/spline combinations and
broader LF filter/resampling combinations need further coverage.

`FrameExecutionPlan` separates physical decode nodes from coalesced presentations. Nodes retain
exact earlier LF producers and their last consumers, the four reference-slot versions before each frame, save-before/after
color-transform metadata, and whether the frame needs canvas composition. Presentation metadata
retains orientation-normalized extent, rational timebase, loop count (including zero for infinite
looping), accumulated ticks, exact timecode, finality, and UTF-8 name. The plan is backend-neutral;
it does not reconstruct pixels or entropy on the CPU.

`WgpuDecodeEngine` executes full-canvas Replace sequences using Modular, VarDCT, or a mixture of
JPEG-transcode VarDCT and non-XYB Modular. Recursive progressive-DC chains may precede each
presentation. A Replace presentation completely supersedes earlier zero-duration Replace layers,
but every physical color/extra/LF producer still decodes and validates exactly once. Unused and
overwritten LF versions are included; consumers reuse the exact planned slot version.
Each overwritten output is released before the next physical producer is admitted. Only the final
producer's output is presented, retaining exact native integer codes without an F32 intermediate.
Layered stills and a final zero-duration animation frame are covered.

`DecodeProfile::FrameSequence` reports physical/presentation counts. `FrameSequenceSession` exposes
the execution plan. Independent presentations prepare one producer at a time and share immutable
inventory/source spans across the bounded prefetch window. LF or composition dependencies use
a serial physical executor; dependent prefetch reports `FrameDependency` while one presentation runs.
Blocking, polling, and futures advance the same validated physical stages. Prefetch preserves ordering,
initial byte-budget pressure leaves the first physical producer available for retry. Once a
presentation has started, a later admission failure poisons the session. Source spans remain under the shared input
budget until their last dependent submission; cancellation and output clones retain the existing
callback/lease ownership contract. Submission counts accumulate every physical producer, including
late VarDCT continuations. Frame-specific syntax/output errors surface when that producer is prepared;
a failure after the presentation starts is terminal. Unvalidated handoff reports
`UnvalidatedOutputNotSubmitted` until the final producer has been submitted, so overwritten outputs
cannot escape through the presentation API. Composition uses the physical execution path below.

Nine positive libjxl fixtures cover 8/12/16-bit Gray/RGB/RGBA, mixed coding modes, all relevant timing
fields, 17 physical frames, six orientations, a transposed one-pixel axis, and recursive
DC2. Actual GPU output is exact for Modular and within one RGB8 code for VarDCT against both Rust
`jxl` and `djxl`, with byte-identical whole and 4 KiB-window/137-byte-fragment async output. The two
formerly rejected crop/Add fixtures now execute and match both decoders.

Additional tests shorten only an overwritten entropy section while rebuilding the TOC and
preserving the complete frame plan. Hidden Modular, VarDCT AC, root LF and intermediate LF failures
poison the session before its first presentation through both blocking and fragmented async input.
Reassembled Gray31 stills with 2, 17 and 129 layers preserve checked-in independent integer words
under a GPU budget restricted to the first producer's footprint. Their submission totals include
every layer. Cancelling before validation or after advancing several hidden layers releases all
input/GPU reservations after callbacks retire.

Sequences containing crops, blends, or reference-only frames use the same ordered LF/physical
executor, including hidden zero-duration layers. The working surface is
unrounded, unrotated planar F32 RGB in the original enumerated D65 sRGB encoding, followed by
every extra plane at its own normalized depth. Up to four reference
slots retain accounted buffer leases; an overwritten slot releases its old version after any
submitted consumer completes. Empty references are zero, with opaque presentation alpha for
images without an alpha channel. Signed crops are intersected on the host with checked wide
arithmetic; all pixel copying and Replace/Add/Blend/Mul/MulAdd operations execute on the GPU.
Color and every extra may read different background slots and select different alpha declarations. Source-over also writes its selected alpha,
and Multiply clamps the foreground when requested. Native 1–31-bit Gray/RGB/RGBA packing and
the shared color-output conversion run after the full canvas has been composed; Apply/Keep
orientation never changes the coordinate system of a retained reference.

The composition executor admits one dependent presentation at a time. `prefetch` reports
`PrefetchBackpressure::FrameDependency { index }` until the oldest pending presentation completes;
the caller can retain its returned output while submitting the next. Submission never blocks,
and native waits and runtime-neutral polling advance the same physical stages. Initial byte/poll
admission failures preserve the source for retry. As with staged VarDCT, an allocation failure
after a presentation starts is terminal for that pending frame. References, uniforms, sources,
and outputs stay budgeted across callbacks and cancellation. No CPU pixels are read for blending.

Twelve libjxl fixtures cover all five blend modes, separate alpha sources, negative and oversized
crops, fully off-canvas frames, empty slots, reference overwrites, layered stills, mixed JPEG/Modular,
and recursive DC, including Gray16+Alpha5 and RGB12+Alpha5. Eleven match Rust `jxl` and `djxl`
within one native output code. F32 comparison
uses linear-light/alpha error divided by `max(1, abs(reference))`: below `3e-6` for Modular and
`1e-4` for VarDCT-containing sequences. A separate Multiply-clamp case verifies extended reference
values analytically and against `djxl`; Rust `jxl` 0.6.0 clamps the wrong operand for that condition.
Re-serialized reference-only variants independently pass both decoders and exercise slot 3.
Post-transform composition rejects pre-transform reference domains before submission. Pre-transform
patch execution, non-sRGB/ICC composition,
and non-coalesced/progressive delivery remain required for full JPEG XL.

### Bounded standard VarDCT engine


The coding-mode-neutral `GpuDecoder::wgpu` selects the VarDCT production engine for two bounded
standard packet topologies. A one-entry TOC stages LF and HF metadata before parsing its general
HF-global and AC continuation, with transforms selected from the decoded strategy map. A sectioned TOC covers one or
more independently bounded LF groups with GPU-decoded mixed maps of any of JPEG XL's 27 regular
and special strategies across one or more 256-pixel pass groups. Those pass groups may carry real HF coefficients across one through eleven spectral/refinement passes
using any of the 13 natural or entropy-coded custom coefficient-order families. Scanline and
entropy-coded center-first TOC order are both accepted: inventory retains physical section ranges
and the frontend normalizes them to logical group order before assigning pixel rectangles and
per-group scratch. The explicit section topology distinguishes combined and separately addressable
packet ranges. The sectioned form supports odd and asymmetric pixel extents across LF
group boundaries while keeping edge padding internal to GPU storage; 2056x256 is the checked
two-LF-group boundary case.
HF-global metadata is represented as shared block contexts/matrices plus `HfCoefficientPass`
records. Each pass owns its entropy descriptor, context map, order tables, coefficient shift, and
spatial packet ranges. The 160-byte pass invocation selects rebased table locations and keeps its
spatial task identity separate from its logical pass-group validation index. Each pass/group has
independent LZ77 and 464-byte resume storage, while atomic integer addition accumulates coefficients
before the single inverse-transform/restoration/output sequence. Every status must validate before
the final image becomes authoritative. Three checked-in spectral/quantized fixtures cover ordinary
and 256-byte windowed execution, 37-byte transport chunks, odd extents, center-first order, and two
LF groups; both CPU oracles agree within one RGB8 code on Apple M5.
Both forms return one final still frame in the requested color layout; a one-entry TOC has one pass, while sectioned TOCs
retain every declared pass. XYB and original-sRGB VarDCT accept all legal integer and floating source
declarations; a JPEG-reconstruction profile accepts 8-bit encoded YCbCr and the codestream's component sampling
selectors. The packet contract
accepts either adaptive LF smoothing or its standard skip flag, every 3-bit X/B frame
quant-matrix scale, every normative default or parametric custom dequantization matrix encoding,
disabled/default/custom Gaborish, disabled/default/custom
one-to-three-iteration EPF, arbitrary valid HF block-context maps, and per-pass coefficient orders, entropy descriptors, and quantized refinement shifts.
`global_scale`, `quant_lf`, LF extra precision, the quant field, per-block `hf_mul`, sharpness,
per-frequency-cell HF chroma correlation, MA
properties 0 through 15, and weighted self-correcting prediction are read from the stream. The
sectioned shared-global-tree packet form resumes across the shared bounded-window
planner without an intermediate map. Its 64/128-byte `Pod` state records the active LF/HF phase,
both decoded counts, first-block count, extra precision, ANS/LZ state, and predictor state. The
packet frontend also represents an absent LF-global tree, packs each LF-local tree independently,
executes LF image entropy on GPU, maps the aggregate end cursors, then parses and packs the following
HF-local trees without decoding host image symbols. Oversized LF-local ranges use the shared
16-byte-overlap planner, one reusable upload, and ordered queue submissions. The ABI defines
16-byte-aligned 64-byte generic and 128-byte SelfCorrecting `Pod` records that preserve ANS/LZ,
consumer, and predictor state. Current local-tree planning reserves the conservative 128-byte
capacity per group because the following HF-local tree is discovered only after the LF map; only the final segment performs entropy termination and contributes to the
single aggregate LF status map. Separate `decode_vardct_lf` and `decode_vardct_hf` entry points
preserve resident LF reconstruction across that boundary. The stock runtime-neutral pending-frame
state machine owns every LF submission, both aggregate status maps, and the initial plus dynamically
admitted metadata reservations. It is actual-GPU tested with ordinary multi-LF-group `cjxl` output
through blocking and async completion. The image header
must declare the standard sRGB/D65
RGB or grayscale presentation encoding, with supported extra channels and no ICC profile.
It receives one selected image inventory. Cropped/blended animations and layered stills enter through the frame
executor above; the low-level standalone VarDCT entry point remains an uncropped still API.

All image orientations 1–8 are normalized before target chroma subsampling and packing. `ColorOutputConfig` explicitly
separates the unrotated `extent` and typed `orientation`; `output_extent()` includes transposition.
Coefficient grids, restoration, component/frame upsampling, and progressive-DC dependencies stay
in codestream coordinates. The shared 192-byte output uniform carries geometry and orientation,
while a 160-byte source uniform describes XYB/JPEG reconstruction and independent alpha. No intermediate RGB image or
additional submission is needed. Odd 257×17 three-pass fixtures cover
every orientation; the packer also checks both one-pixel axes and zero tail padding.

Grayscale XYB uses the linear sRGB luminance projection folded into the inverse-opsin matrix before
the sRGB transfer function, producing equal RGB channels. The Modular root of a grayscale
progressive-DC chain still reconstructs three internal XYB channels. Six grayscale fixtures cover
a single-entry image, quantized passes, 4× frame resampling, two LF groups, recursive DC+AC, and a
non-XYB JPEG transcode. A 173×101 oriented 4:2:0 JPEG-transcode case verifies that both HF metadata
and AC traversal include MCU-padded edge blocks. Late host matrix uploads exclude raw matrices
already decoded on GPU, including every transposed and AFV alias. Whole-input blocking and bounded
fragmented-input async results match Rust `jxl` and `djxl` within one RGB8 code, with exact equality
between upload policies and complete budget release.

The source depth describes the original integer samples and does not rescale the normalized XYB
reconstruction or constrain the RGB8 output to that source precision. Sixteen 257×33 spectral
fixtures check every accepted depth, and four additional fixtures combine 12/16-bit input with
grayscale, orientation, resampling, multiple LF groups, recursive DC, and a single-entry TOC.
Whole and 256-byte-window async outputs match both CPU oracles within one RGB8 code. The `djxl`
oracle requests sRGB PFM to preserve floating-point reconstruction before one test-only RGB8
quantization; PNM's source-depth quantization would invalidate this comparison for low-bit inputs.
Unsupported integer depths retain both the declared depth and color transform in the typed packet
error. Floating-point declarations are also supported; integer depths above 16 remain unsupported.

Ordinary frame upsampling uses the image header's standard or custom 2×/4×/8× weights. The
profile separates encoded `width`/`height` from presented `output_width`/`output_height`; LF/HF,
coefficient and restoration work use the encoded grid. Three resident 5×5 filter dispatches then
expand that grid before XYB conversion, with mirrored boundaries, normative range clamping, and
right/bottom cropping to the exact output extent. One expanded weight buffer is shared by all
channels. Output-sized F32 planes, weights, and three 48-byte uniforms are explicitly budgeted and
retained through final validation. Checked-in libjxl fixtures cover all factors, custom nearest
neighbor weights, spectral passes with 4× resampling, odd dimensions, single-sample axes, and a
4111×17 output spanning two LF groups. Blocking and bounded-window async output matches Rust
`jxl` and `djxl` within one RGB8 code.

Every single-entry TOC now uses the staged LF, HF-metadata, and general HF-global/AC continuation,
including ordinary frames with a global tree. There is no dimension-derived transform assumption:
the GPU reads and validates the actual strategy map. This intentionally trades the old single-entry
shortcut for complete metadata handling. Each entropy stage retains bounded windows and shares one
logical pending frame and byte budget.

Adaptive LF smoothing requires equal component sampling factors. The shared frame-header parser
rejects unequal factors with `InventoryError::SubsampledAdaptiveLfSmoothing` before the TOC,
section delivery or GPU admission; public inventory negotiation reuses that validation. All four
equal-selector triples remain valid 4:4:4, including nonzero selectors with additional MCU padding.
`tests/jpeg_sampling.rs` verifies all 64 triples at both 272×32 and 257×17, with nonzero/zero noise,
independent native/Rust F32 references and exact whole/bounded input equality. Equal factors also
exercise adaptive LF smoothing. Eight additional streams with nonzero LF correlation guard
equal nonzero selectors: LF dequantization now applies correlation based on the actual channel
shifts. Native checks both noise states, while Rust checks the zero model because of its documented
noise-correlation difference. Sixty invalid smoothing combinations have whole/incremental
rejection, poisoned-stream and immediate source-release checks. Gaborish and EPF are connected:
shifted components use a fused
horizontal/vertical quarter/three-quarter resident upsample before the full-resolution restoration
cursor, while unshifted component buffers are reused directly. All destination planes and 32-byte
`Pod` uniforms are included in the shared byte budget. The interpolation primitive is actual-GPU
tested on horizontal, vertical, two-axis and odd-edge cases. Twenty additional JPEG codestreams
verify Gaborish and effective EPF before noise, including vertical subsampling with documented
native/Rust reference exceptions and an independent scalar Gaborish check. The common frame executor accepts
recursive progressive-DC dependencies, including VarDCT roots without an LF source and
LF-dependent SkipProgressive frames. These stills expose `DecodeProfile::FrameSequence` and
`WgpuDecodeSubmissionSession::Sequence`. It keeps three F32 XYB planes resident, uses shared 80-byte Modular normalization and 48-byte LF-pack `Pod` uniforms, validates every
hidden and visible status, and publishes complete presentations plus opt-in validated LF updates.
A physical LF node is decoded once, even when unused, overwritten, or shared across multiple
presentations. The plan
checks LF flags, levels, exact slot versions and sample/block extents before submission.
`MemoryPermit::split_off` transfers each plane's actual byte reservation from producer scratch
into a `GpuBufferLease` without readmission. Both Modular and VarDCT LF capture retain the final
pre-color-transform planes after restoration and frame upsampling, with that output geometry and
stride. Modular reuses the presentation reconstruction implementation and skips RGB conversion.
Both producer modes validate additional channels while retaining only XYB. LF-consuming VarDCT
frames can first decode global Modular extras through bounded GPU cursor stages, preserving the
source planes and their existing reservations until the final consumer submission. Six independent
alpha/depth fixtures cover both LF roots, optional Gaborish, and LF2→LF1→presentation chains; RGB,
alpha and depth agree with libjxl and Rust `jxl` through whole and fragmented async decoding.
Two additional 2051×33 streams distribute Squeeze extras into two LF groups. The typed
`BoundedVarDctGroupEntry` distinguishes coefficient entropy, preceding LF extras, and directly
known HF metadata. LF-extra consumers fence initial arena copies, decode each extra subimage on
GPU, then parse HF descriptors at validated cursors. They reserve conservative HF history,
predictor state and bounded upload capacity before submission, and admit exact descriptor bytes
when discovered. No placeholder LF descriptor or fabricated LF-success status is used. Every
group validates even for unselected channels; canceled stages retain their leases through callbacks.
Its default/custom Gaborish, EPF1/2/3, custom sigma, and 2×/4×/8× LF variants match both independent
decoders. `modular_render_bytes` includes the full reconstruction footprint; LF plane/uniform
counters are subsets of that total. Final planes retain their exact byte permits while intermediate
normalization/restoration allocations are released after producer validation. Four versioned LF slots retain these leases through
the last consumer; expired planes and validated scratch are released before the next admission. A single-entry intermediate frame
first executes HF metadata on GPU, maps its bounded HF-global cursor, host-parses only scalar
HF-global tables, and resumes general AC plus downstream reconstruction on the same queue.
Global-only Modular roots run their inverse/conversion and final status map in the last DC-global
submission, with zero subimage lanes. A checked three-frame DC-plus-quantized-AC fixture covers
this combination with both whole-range and 256-byte-window async execution.
Fixed scratch/status capacity is admitted before submission; descriptor/order/window bytes discovered at
the cursor are admitted through the same shared budget. `cjxl --progressive_dc=1` and
`--progressive_dc=2` actual-GPU outputs are checked through blocking and runtime-neutral async
completion against Rust `jxl` within one RGB8 code. Parametric matrix modes 0 through 6 populate the
resident resource table. For raw mode 7, the sectioned global-tree path now decodes the complete
three-channel Modular side image with the common GPU entropy executor, runs its resident
Palette/RCT/Squeeze inverse schedule, validates a 16-byte mapped status, and overlays positive
finite weights into each aliased strategy-matrix target before AC/render. A real cjpeg-to-cjxl
JPEG-transcode stream and two reproducible local-MA variants cover that primitive on an actual
adapter. Local-tree packets now complete
their LF cursor and every HF-local metadata window before entering the same repeated raw-matrix
state. Raw stages reuse one bounded upload from shared source spans and resume their entropy,
predictor and LZ state. The input cap follows caller/device limits and shrinks to fit available
shared budget, down to 40 bytes when continuation is needed. Each map validates progress and an
absolute cursor; inverse transforms and overlay wait for complete entropy. No later HF-global
suffix is uploaded once that cursor is known. Four 264x64 cjpeg fixtures cover 4:4:4, 4:2:2,
4:4:0, and 4:2:0 through the public decoder: LF/AC entropy uses exact per-component dimensions,
tasks carry three LF offsets and component destinations, and the packed output kernel applies
separable quarter/three-quarter JPEG upsampling with replicated edges followed by encoded BT.601
YCbCr-to-RGB conversion. Actual-GPU RGB8 differs from Rust `jxl` and optional `djxl` by at most one
code. Global- and local-MA raw images also match whole input byte-for-byte with 40/64/256-byte caps
and seven-byte transport chunks, including a stream with local LF/HF trees and no global MA tree.
The final aggregate validator tracks the actual HF-metadata entry point through that late raw
stage. Larger/transformed raw matrices, uncommon asymmetric component sampling, and subsampled
restoration still need conformance coverage.

`ModularSideImagePlan` and `wgpu_engine::side_image::modular` now own the shared substream
descriptor, bounded entropy/inverse recording, exact allocation size and mapped absolute cursor.
The raw-matrix layer supplies only its denominator, alias targets and overlay. Completion work
shares the initial encoder for a whole stream and a deferred encoder for windowed input. The
raw map callback owns both image and frame allocations until GPU completion, including cancellation
during the first window, a continuation, or finalization. Recording keeps the resident arena
available to a downstream consumer before the final status copy; arbitrary
one-, two-, three- or many-plane topologies share the same descriptor-based GPU executor.
The common executor retains original samples as signed integer words and performs no color
conversion. The 256-byte entropy ABI, 16-byte status and 64-byte matrix overlay ABI are unchanged.

The public VarDCT producer now executes global Modular extras as an initial GPU stage. A checked
16-byte status supplies the exact next LF bit position, or validates padding at the end of an
independent LF-global section. Only then is the frame allocation/dispatch plan constructed.
The first declared alpha plane stays in its original signed integer arena through the final
color submission; the packer uses its own bit depth and applies orientation without a plane copy.
`GpuOutputRequest::with_extra_channel` can instead select any declared extra plane as
native unsigned or normalized scalar F32 output, including non-alpha and multiple-alpha images.
The complete LF/HF/AC stream still validates before delivery. Color inverse transforms,
restoration, resampling and resident color image buffers are omitted from that frame plan.
One word-owned packing dispatch reads the selected plane in its original arena and applies the
requested orientation. Native unsigned output preserves representable codes exactly and rejects
out-of-range samples; scalar F32 preserves signed normalization without clipping or color transfer.
Explicit `SpotColorPolicy::Preserve` requests base color from the physical producer. The common
`WgpuDecodeEngine` adds spot presentation over retained all-channel surfaces; calling a physical
Modular/VarDCT engine directly with Render returns a typed routing diagnostic. Unknown non-optional
channel interpretation remains unsupported.

`VarDctDecodeSession::memory_stats()` returns `Option<VarDctDecodeMemoryStats>`: the frame
plan is absent until the global cursor validates. `global_modular_memory_stats()` reports exact
initial-stage reusable stream, arena and total buffer bytes. Both stages share the engine budget, while
`in_flight_memory_stats()` includes every live permit. Submission counts grow as stages become
known. A pending global stage cannot expose an unvalidated frame. Dropping it retains GPU
buffers and permits in the map callback until completion. Color metadata reports the actual
1–31-bit source depth and preserves all extra-channel declarations.

Six internal substream fixtures still compare every reconstructed plane to exact source codes
and Rust `jxl`/libjxl. Seven public fixtures add color and first-alpha comparisons, including a
progressive multi-entry stream, synchronous and fragmented asynchronous input, Apply/Keep,
entropy corruption, memory backpressure/retry and cancellation. The common executor retains its
256-byte entropy/16-byte status ABI; fused output uses a 160-byte source uniform and a read-only
alpha binding. The global stream now uses the common 16-byte overlap/four-byte sentinel layout
and resumes ANS, LZ77, MA and predictor state over one reusable input/parameter pair. Caps as small
as 40 bytes match whole-stream output exactly. A validated sample count and entropy terminal state
finish the stream early, without uploading the remaining LF/HF/AC suffix. Inverse transforms wait
for that completion and retain their budgeted uniforms across every continuation. Each async poll
advances at most one window; cancellation retains buffers until its map callback finishes.

Window geometry is generated on demand in constant host space, including empty single-symbol
Prefix streams. The initial-stage planner reduces the upload against total budget capacity; an
explicit cap below 40 bytes returns `StreamWindowTooSmall`, and an insufficient minimum allocation
returns `MemoryBudgetTooSmall`. Live reservations remain retryable submission backpressure.
The scalar tail reserves a 64-byte uniform, a four-byte range status and four more bytes in the
existing aggregate status map. It adds no submission or pixel readback. All 32 planes in the seven
public fixtures have exact native and dual-oracle F32 coverage under whole and bounded input;
corrupting a later AC section still fails the scalar request after the global stream succeeds.
Floating extras use `NativeFloat` and the same scalar tail. Raw matrix side images now share this
bounded executor and its deferred finalization contract.

For LF/AC distribution, Modular headers and transform topology now parse separately from MA and
image entropy. Global ownership is a leading channel prefix; an empty global subimage has no local
MA/entropy descriptor. The shared group planner clips transformed channels into LF/pass regions
and validates progressive brackets, including final-pass boundaries that preserve all remaining
resolutions. Five distributed libjxl fixtures assert that every coded
sample belongs to exactly one global, LF or pass subimage. An empty global prefix proceeds directly
to frame preparation; nonempty prefixes retain their transformed samples for later assembly.

The low-level `HfCoefficientExecutionPlan::set_stream_end` selects `Packet` or `Continuation` per
logical pass group and updates every bounded-window parameter. Packet mode retains exact
zero-padding validation. Continuation mode validates ANS terminal state and returns the next bit
without consuming the suffix, even before the final upload window. `GpuHfCoefficientStatus::validate_cursor`
checks the result against host-owned group identity and packet bounds. Callers must consume and
validate the following stream before delivering a frame. Actual-GPU tests hand all eight bit
alignments from zero-bit Prefix, nontrivial Prefix and ANS AC into a following Modular image,
check its resident samples and cursor, and reject truncated/corrupt ANS endings. This boundary
API now drives the public frame scheduler. LF cursors first resume any Modular LF subimage,
then the HF metadata parser. AC groups with nonempty extra subimages use continuation mode;
empty ones retain exact packet validation. Each subimage parses bounded local metadata, performs
GPU entropy and local inverses, and copies only its clipped rows into the frame arena. AC-side
Modular completion additionally checks terminal padding. After every group validates, the global
inverse runs once before the existing color/alpha or scalar output tail.

The frame memory plan includes `extra_arena_bytes` and `extra_inverse_uniform_bytes`. Each
subimage reserves its stream, metadata, predictor/LZ77 workspace and local inverse buffers only
when its descriptor becomes known; the shared live-budget counter includes those dynamic bytes.
A mapped completion retains both the subimage reservation and the frame arena through cancellation.
Every async poll yields at a stage boundary, and no unvalidated output is available before the
final frame tail is submitted. Resource exhaustion after decoding starts remains terminal.

The five distributed fixtures cover all 13 extra planes as exact native unsigned codes and
normalized F32, plus color and first alpha against Rust `jxl` and libjxl. Tests use whole blocking
input and 347-byte fragmented async input with 1024-byte entropy windows, including the
2051×259 LF/pass Squeeze case. Additional tests cancel a 128-byte window, retry initial admission,
reject a budget exhausted by the base frame, and reject corrupted extra entropy for both outputs.
A one-symbol code-length alphabet in the tiny second LF group also exposed and fixed a shared
Huffman descriptor bug; lengths 1–15 and all applicable skip forms have regression coverage.

A valid UTF-8 frame name is preserved in authoritative `FrameMetadata`; invalid bytes return a
typed error. Container/codestream parsing is capped at 16 MiB and 32 boxes before any fragmented
payload can be reassembled; this is an engine limit, not a late profile check after the generic
1-GiB parser ceiling.

The retained VarDCT source is a checked logically contiguous table of shared spans. Entropy-window
planning depends only on its logical length; LF, combined LF/HF, staged HF, and AC uploads copy
their exact ranges across arbitrary physical span boundaries. Whole-range kernels still require a
full GPU codestream buffer, but it is initialized while mapped directly from the span table and
zero-padded to four bytes without constructing a second full-size host `Vec`. Initial LF/HF packet
metadata, MA descriptors, block-context maps, custom coefficient-order permutations, and
cursor-dependent local-HF metadata all read the same span table without joining it. Slice-based
low-level APIs remain zero-copy through the same reader contract. The public decoder remains
runtime-neutral: `GpuDecodeStream` feeds the public transport events synchronously, then returns the
ordinary sync/async `GpuDecodeSession` after authoritative end-of-input. A separate cloneable input
budget covers the compressed spans retained by concurrent streams without conflating host storage
with the GPU allocation budget. Its permit moves into the codec source and is released at the last
submission that still needs those bytes, or immediately when the stream/session is cancelled.

The host inventories bounded scalar headers, packs the shared or local MA-tree and coefficient entropy descriptors,
and expands only the small HF coefficient-order metadata permutation. It does not decode an LF/HF
image entropy symbol or coefficient value. One GPU submission decodes and validates LF/HF metadata,
dequantizes and smooths LF, lowers every non-overlapping first block into typed HF tasks, decodes
each pass group through the common Prefix/ANS/hybrid-integer/LZ77 executor and all-order coefficient
sink. The HF metadata channels retain their logical dimensions while addressing capacity-strided
storage, so the `hf_mul` row cannot alias the packed strategy row when the actual first-block count
is below the allocation capacity. Block-context selection reads the resident quantized LF planes
and each task's `hf_mul`; its variable tables share the entropy bundle, while per-group LZ history
occupies a disjoint tail slice of its LF group's reconstruction buffer so the pass remains within eight
portable storage bindings. Each LF group owns independent reconstruction, raw-metadata,
coefficient, packet-status, artifact, occupancy, and HF-status buffers. LF values use the stream's
default or explicit channel dequantization multipliers and LF chroma-from-luma slopes before
scattering into the full-image resident atlas. HF correlation uses the same explicit base values
and colour-factor reciprocal while lowering each frequency cell. Adaptive LF smoothing runs once over
the complete block grid, so the 2048-pixel LF boundary is not treated as an image edge. Each
artifact carries global output/LF/correlation origins and creates 27 compact strategy buckets and indirect dispatch
records. The submission executes every populated bucket through the resident regular or special
VarDCT renderer using the normative default matrix for that strategy and an explicit regular/wide/
special coefficient layout, optionally applies the signaled Gaborish weights, constructs the signaled
per-block EPF inverse-sigma field, runs EPF0/EPF1/EPF2 as selected by the one-to-three iteration
contract through a shared resident ping-pong plane set, then applies inverse opsin, converts encoded
YCbCr with any remaining component upsampling, or preserves original RGB. The common output
packer applies the requested color transfer. It writes tightly packed
RGB8 without an intermediate image readback. Sectioned global-tree packet and
AC pass-group ranges that fit the resolved entropy cap retain the one-submission path. Either
oversized consumer instead uses the consumer-neutral 16-byte overlap plan, one reusable stream/
parameter pair, and ordered queue submissions. The final sectioned global-tree packet command shares the
first downstream submission. A 464-byte aligned `Pod` tail per pass group preserves bit/ANS/LZ
state, nested block/channel/order progress, coefficient-sink error, and the 96-word nonzero context
grid; only the final window validates exact ANS/padding termination. Local-tree frames first run one
or more bounded LF submissions and map one aggregate LF cursor record, host-pack the HF descriptors,
then run one or more bounded HF submissions through the same upload before the already-recorded
downstream work and one final aggregate validation map. The final HF window shares the first
downstream queue submission instead of adding an avoidable boundary. Every LF group's packet and artifact status plus one 32-byte record per pass group
share the final map; cleared downstream buffers and zeroed indirect
dispatch records make a rejected packet non-authoritative rather than an unchecked render. There
is no CPU pixel, coefficient, transform, quantization, residual, entropy, or color fallback.

`vardct_rgb8_format()` remains a convenience descriptor. The engine accepts the shared color output
families: all 20 color VPI pitch-linear layouts; planar/interleaved U8 or F32 RGB/BGR/RGBA/BGRA; Y8/Y16;
planar or semiplanar 4:4:4/4:2:2/4:2:0 YCbCr at 8/10/12/16 bits; and packed YUYV/UYVY. Primary
conversion supports D65 BT.709, BT.2020, and Display-P3; transfers are Linear, sRGB/SYCC, BT.709,
and BT.2020. Output YCbCr selects BT.601/709/2020 NCL or BT.2020 constant luminance, full/limited
range, and supported centered/cosited chroma locations. RGB requires full range; the first alpha is reconstructed at its independent depth, with explicit
Unassociated/Preserve/Associated output policy.

The codec source fragment reconstructs unclipped linear BT.709 from XYB, or encoded sRGB from
JPEG components, inside the render backend's shared word-owned output shader. Chroma sampling
therefore follows orientation and full-precision reconstruction before one final quantization.
`ColorOutputInputs` takes an explicit checked `ImageLayout`; output planning uses its exact logical
byte length and four-byte storage rounding. Separate 192-byte output and 160-byte source uniforms
cost 352 bytes in total and are checked individually against binding limits. Padded rows, unaligned
plane starts, last-row tails, opaque alpha, and unused sample/storage bits have actual-GPU coverage.
Thirty integer layout/transfer cases match both float CPU oracles within one code at 8–12 bits and at most
three codes at 16 bits on Apple M5. Dedicated Display-P3 and BT.2020 cases match requested `djxl`
color output within one RGB8 code. Non-color numeric output, arbitrary ICC output, and explicit
luminance mapping for PQ/HLG remain typed gaps; relative SDR is never relabeled as HDR.

Nine additional F32 layout/transfer cases preserve unclipped color without integer quantization.
The CPU comparisons measure reconstruction error in linear light and report encoded error as well;
the thresholds are 0.00002 against Rust `jxl` and 0.0001 against `djxl`. Linear `djxl` PFM output is
requested directly, avoiding an extra sRGB round trip. Whole and bounded fragmented output remain
byte-identical. The frame executor uses these float outputs for crop/blend/reference composition
without prematurely clipping or rounding its inputs.

These color outputs are accepted directly by `DisplayPipeline::submit_image`, which produces a
GPU-resident linear-BT.709 texture without an intermediate CPU readback; wide-gamut output requires
the float display descriptor. `VarDctDecodeMemoryStats` accounts every upload, metadata, status,
uniform, artifact, coefficient, XYB, optional three-plane restoration scratch, EPF sigma and
per-pass uniforms, transform-scratch, and
output byte. By default those bytes use
the backend budget shared by decode, encode, and readback; `VarDctSubmissionEngine::with_memory_budget`
can instead select an explicit sharing group. Transient reservations survive until the final
aggregate status map completes. For local-tree frames the initial reservation contains all LF
metadata; if the cursor-dependent HF metadata peak is larger, the exact difference is admitted
from the same shared byte budget before the second submission. The packed output reservation
survives through the final tracked
`GpuBufferLease` clone, including an early `UnvalidatedGpuImageFrame`; only the validated frame
carries authoritative metadata and changed regions. Native blocking and runtime-neutral
poll/future completion use the common decoder session API, and the engine compiles for browser
WebGPU without a Tokio or async-std dependency.

`VarDctDecodeMemoryStats` separately reports the shared packet stream peak, initial packet batch
count, reusable AC stream peak and batch count, reusable parameter bytes, LZ scratch, and
execution-state total. One typed initial window plan selects LF, HF-only, or combined packet
execution. For sectioned global-tree packets the initial count covers the complete packet; for
local trees it covers LF; for LF consumers with eager HF descriptors it covers HF. Their generic
64-byte or weighted 128-byte resume record uses the exact admitted predictor layout. HF descriptors
discovered after an LF cursor update `hf_packet_stream_batch_count()` and `submissions_per_frame()`
when that dynamic plan is installed; eager HF batches are counted from preparation. Performance
harnesses sample counts after frame completion. Applying
`WgpuDecodeEngine::with_stream_window_limit` configures both coding-mode engines and governs
sectioned global-tree packets, staged LF/HF packets, eager HF-only LF consumers, and AC pass groups.
It is a caller upper bound:
device limits and deterministic planning against the shared budget's total capacity may select a
smaller four-byte-aligned value, exposed as
`VarDctDecodeMemoryStats::resolved_stream_window_limit_bytes`. Planning does not sample live
headroom, so concurrent opens are reproducible; live jobs still receive typed non-blocking memory
backpressure at submission. If even the 40-byte overlap/sentinel layout exceeds the budget,
`MemoryBudgetTooSmall` reports both exact planned bytes and the configured limit before GPU work.
Cursor-dependent local-HF metadata remains the one dynamic addition and admits only its exact
positive difference from the same budget. Actual-adapter runs force a 40-byte staged 32x32 single-entry packet,
256-byte shared-global/local-tree/nonzero-AC paths, and an intermediate budget-resolved cap through
blocking or runtime-neutral async decode, typed corruption/backpressure, and cancellation-driven
reservation release.

Modular DC-global/group and VarDCT packet/AC window plans retain packed short streams or constant-
size geometry for each oversized stream. Counts, peaks and maximum lane occupancy are computed
without expanding windows. Packet and AC parameters are derived from one base record per coded
stream, including termination changes made before deferred submission. Preparation retains these
records and the compressed source lease; one host upload and the active parameter batch are filled
at submission. Explicit host window storage is O(coded streams + selected cap), in addition to
the existing compressed-source and entropy-table storage. The last upload releases its source
lease, while submitted GPU work keeps its existing callback-owned reservations.

GPU commands and bind groups are recorded immediately before each ordered submission, including
deferred AC after HF-global parsing. This avoids exhausting Metal's command resources when the minimum 40-byte
window splits a 1024×128 recursive DC+AC stream into thousands of submissions. Seventeen-byte
transport chunks and asynchronous completion produce exactly the same RGB8 bytes as whole input;
both independent decoders agree within one code. Cancellation before and after LF/HF transitions
releases input and producer reservations after callbacks retire.
Near-u32-bit geometry tests describe more than 134 million windows, directly inspect first/middle/
last batches and parameter records, and exercise Modular budget selection without window tables.

The actual-adapter matrix covers all nine accepted single regular transform extents plus sectioned,
odd/asymmetric multi-task and multi-pass-group frames. Lower-level GPU kernel oracles cover all 27
inverse transforms, including AFV and 64/128/256-scale strategies. It GPU-encodes each
standard packet, executes the complete resident decode, reads the result back explicitly, and,
when `djxl` is installed, compares it with that independent decoder (at most one RGB8 code of
rounding difference for the covered solid-image cases). The Dct8 case also exercises the
runtime-neutral async completion and GPU display conversion. A deterministic mutation of the
Modular header into a malformed local-tree descriptor is rejected by the bounded host metadata
parser before submission. The matrix also
verifies that readback releases its shared reservation while the last decode-output buffer clone
continues to own the exact output bytes. A separate 438x589 libjxl fixture exercises six nonempty
pass groups, 4,070 DCT8 tasks, a custom three-channel coefficient order, nonzero AC coefficients,
and a self-correcting MA tree; GPU RGB8 output differs from both Rust `jxl` and `djxl` by at most
one code.
A checked-in 257x257 libjxl effort-5 fixture exercises a mixed strategy map, LF extra precision,
three HF block clusters, custom orders 0 and 1, and a first-block count smaller than the
capacity-strided metadata allocation. The actual-adapter test executes its single-pass AC and every
populated regular/special bucket through the stock decoder, then permits at most one RGB8 code of
difference from Rust `jxl` and optional `djxl`.
A separate deterministic 438x589 fixture stores the same bounded one-pass DCT8 topology in libjxl's
center-first entropy-coded TOC order. Its physical pass-group order differs from row-major order;
the frontend test proves each logical group retains the matching physical bit range, and the
actual-adapter test matches both development-only pixel oracles within one RGB8 code.
A second deterministic 438x589 libjxl fixture enables standard Gaborish weights while disabling
EPF. The same actual-adapter test executes inverse VarDCT, fused three-plane Gaborish, and RGB8
packing in one command buffer, and differs from both development-only pixel oracles by at most one
code. It also checks the exact extra reservation of three padded F32 planes and one 80-byte
uniform.
A pair of deterministic 257x17 libjxl fixtures covers the standard EPF2 bundle and EPF3 custom
iteration count on an odd, 8-pixel-unaligned extent. The actual-adapter test executes Gaborish plus
EPF1/EPF2 or EPF0/EPF1/EPF2 without readback between stages, checks the exact shared scratch,
sigma, and uniform reservations, and permits at most one RGB8 code of difference from both Rust
`jxl` and `djxl`. A separate actual-GPU malformed-metadata test feeds sharpness 8 through the same
WGSL validation function used by the packet decoder and requires the typed `Sharpness` error.
Two deterministic 2056x256 standard fixtures contain a 2048x256 LF group followed by an 8x256
tail group and nine pass groups. One enables whole-image adaptive LF smoothing and the other uses
the standard skip flag. Their actual-adapter tests execute both groups, default Gaborish, EPF1, and
RGB8 packing in one queue submission, validate every packet/artifact/pass-group record from one
map, and match Rust `jxl` plus optional `djxl` within one RGB8 code.
An actual-GPU block-context differential test covers negative and positive LF thresholds, exact
threshold boundaries, multiple quant-field segments, all channel positions, and distinct order
IDs against the normative scalar index formula. Naga semantic validation runs even without an
adapter.
An additional actual-adapter test generates a deterministic 2056x256 RGB source and invokes
`cjxl` with distance 2, effort 7, and raw-codestream output. Its ordinary per-LF-group local trees,
including non-default X=5/B=5 quant-matrix scales, complete through the stock frame engine in two
submissions. Blocking and runtime-neutral async results differ from Rust `jxl` and optional `djxl`
by at most one RGB8 code. The test also requires a typed refusal of early unvalidated output before
the HF submission, and verifies that abandoning the LF stage releases its shared memory reservation
after GPU/map completion.
The serialized pass-1/pass-2 zero-flush values remain present in the frame inventory. libjxl 0.12
and the Rust `jxl` implementation accept those parameters but their EPF weight function does not
apply them; the GPU formula follows those executed references rather than inventing a threshold
operation.

This is not full VarDCT coverage. Broader progressive
intermediates, larger/transformed raw-matrix conformance, broader asymmetric JPEG
restoration/resampling combinations and other Modular side images,
numeric color-channel output, ICC/HDR luminance mapping,
and complete progressive presentation remain typed or unproven gaps. Crop/blend
animation and post-transform references are supported through the common frame executor. Unsupported paths return typed
errors. They are not substituted with dummy coefficients or a CPU implementation.

### Measured lossless Modular checkpoint

On an Apple M5, the 36,643,474-byte 7680x4320 Gray8 conformance codestream decoded to the exact
33,177,600-byte source hash in one codec submission. Warm sequential decode plus staged UMA
readback (`warmup=1`, `iterations=7`) selected 64 invocations per workgroup. Before the aligned
output and private distance-one history paths, 32/64/128/256 produced median latencies of
280.151/280.387/292.455/303.263 ms respectively, while 64 had the best mean and p95
(280.878/282.067 ms). Aligned output plus private distance-one history first reduced that to
209.949 ms median. The subsequent host-proven channel-fixed Gradient kernel measured 132.569 ms
median, 132.630 ms mean, 129.145 ms minimum, and 134.377 ms p95: 36.86% below the 209.949 ms
checkpoint and 11.0x faster than the earlier 64-lane, 8-MiB-window,
one-invocation-workgroup checkpoint (1.457 s). Direct normalized-Gray8 output plus the proven
two-row workspace then measured 110.366 ms median, 110.507 ms mean, 108.914 ms minimum, and
111.574 ms p95 with the same exact hash and one codec submission: 16.75% below the 132.569 ms
checkpoint. These are one-device engineering measurements, not cross-adapter performance
guarantees; `memory_stats` exposes the resolved window, lanes, workgroups, output and
reconstruction paths, logical and physical sample workspace, LZ storage, and submissions for each
device and request.

The same direct/two-row pipeline decoded the exact 15360x8640 Gray8 hash from a 146,573,715-byte
codestream in four bounded submissions: warm median 426.034 ms, mean 427.862 ms, and minimum
423.385 ms (`warmup=1`, `iterations=3`), 35.20% below the preceding 657.463 ms/six-submission
checkpoint and 13.76x faster than the earlier 5.861 s/32-submission implementation. The
132,710,400-byte output remains persistent while the stream window and proof-sized parallel
scratch are reused across waves.

## GPU output formats

`GpuOutputRequest` always carries a concrete `jxl_gpu_formats::PixelFormat`; the request is never
ignored. Construction is intentionally breaking and explicit: `GpuOutputRequest::color` accepts
only classified color formats, while `GpuOutputRequest::numeric` requires a
`NumericSampleMapping`. Canonical Gray/RGB/RGBA descriptors shared with
`jxl_wgpu_encode::LosslessModularFormat` produce exact native unsigned output: 1-8 valid bits use
one byte per component and 9-16 valid bits use a little-endian 16-bit component with the valid code
in its low bits. Gray selects `NumericSampleMapping::NativeUnsigned`; RGB/RGBA use the color
constructor. The 8-bit Gray conversion path additionally supports:

- all 10 numeric VPI 4.1 pitch-linear formats: U8, S8, U16, U32, S32, S16, 2S16, F32, F64, and
  2F32;
- all 20 color-bearing `VpiPitchLinearFormat` descriptors: limited/full Y8 and Y16; limited/full
  NV12 and NV24; limited/full UYVY and YUYV; and interleaved or planar RGB8, BGR8, RGBA8, and
  BGRA8;
- equivalent classified 8-bit planar or semiplanar YCbCr layouts at representable 4:4:4, 4:2:2,
  or 4:2:0 sampling, including NV21/NV16/NV61/NV42 and I420/I422/I444 descriptors.

Every color request requires a defined transfer and range. Linear, sRGB/sYCC, and BT.709/BT.2020
transfer conversion runs in WGSL. Gray is replicated into RGB/BGR, alpha is opaque, and YCbCr
chroma is the exact neutral code. Full/limited luma quantization, native 16-bit luma words,
four-byte YUYV/UYVY pairs, odd-width tail duplication, and all plane writes happen directly in the
GPU output buffer. PQ and other unimplemented transfers return `UnsupportedOutputFormat`.

Generic outputs use `classify_pixel_format`; the exact native Modular descriptor is recognized
separately because sub-8/sub-16 valid-bit padding is part of its contract. `NormalizedGray8` maps
code 0..255 across the full unsigned range, the nonnegative half of a signed type (0..MAX), or
float 0..1, and replicates the result for 2S16/2F32. Integer endpoints and rounding are computed
without overflow; F32 bits are formed deterministically with integer arithmetic rather than
backend-relaxed division.

F64 precision is never inferred silently. `NormalizedGray8F64` requires an `F64OutputPolicy`:
`NativeRequired` rejects devices without enabled `SHADER_F64`,
`NativeOrExactF32Widening` explicitly permits a compatibility fallback, and
`ExactF32Widening` always constructs binary64 as the exact widening of the correctly-rounded F32
normalization. The compatibility result is valid IEEE-754 F64 storage, but is not a precise F64
division. When `WgpuBackend::native_f64_enabled()` is true, the native path is lazily compiled and
evaluates `f64(gray) / 255.0` in WGSL. `WgpuDecodeSession::f64_output_path` reports the resolved
path. The returned `GpuImageFrame` owns pitch-linear GPU buffers. CPU readback occurs only when the
application explicitly requests one. On an eligible native UMA backend, the stock decoder marks
its caller-visible output `MAP_READ`, allowing `ImageReadbackPipeline::submit` to map the sole
output in place; portable and aggregate requests retain the explicit staging-copy path.

## Public flow and bounds

1. Construct `GpuDecoder::wgpu` around an application's existing `WgpuBackend`; construction is
   fallible because every mode-specific kernel policy is validated up front.
2. Call `open` with encoded bytes and a `GpuOutputRequest`.
3. Fill the ordered GPU queue with `prefetch`, `poll_prefetch`, or the runtime-neutral
   `prefetch_async` future. Prefetch submits work and never waits for frame completion.
4. Optionally borrow `pending_frames`/`front_pending_frame` and call the stock
   `WgpuDecodePendingFrame::unvalidated_gpu_frame` to enqueue same-device, same-queue display, readback,
   or custom GPU work before mapped-status validation completes. For a local-tree VarDCT frame,
   this returns typed `UnvalidatedOutputNotSubmitted` while only the LF stage is queued; call it
   after polling/waiting has validated the LF cursors and queued the dependent HF submission.
5. Consume the oldest pending frame with `next_frame` synchronously or
   `next_frame_async`/`poll_next_frame` through `std::future::Future`.
6. Retain each `GpuFrameLease` only while its GPU resource is needed. The lease holds an
   `InFlightPermit`; dropping it wakes a pending submission.

The early handoff returns `UnvalidatedGpuImageFrame`, not `GpuFrameLease<GpuImageFrame>`. It
contains only the queue token, requested layout, and permit-bearing buffer leases; authoritative
frame metadata and changed regions remain unavailable until `next_frame` succeeds. Queue ordering
removes the host synchronization point, but not validation: consumer commands already submitted
before a later validation failure cannot be rolled back, and their textures or bytes must be
discarded. Keep `GpuBufferLease` clones alive for accounted ownership; cloning a raw
`wgpu::Buffer` through `as_wgpu_buffer` is outside budget tracking.

The engine boundary is likewise split: `GpuSubmissionSession::submit_next` returns an owned
`GpuPendingFrame`, whose native `wait` or runtime-neutral `poll_complete` performs only completion
and mapped-status validation. `GpuDecodeSession` keeps these values in a `VecDeque` and therefore
returns presentation frames in submission order even when later GPU work completes first.
`PrefetchProgress` reports cumulative submitted frames, current queue depth, explicit stream end,
and typed frame-slot/memory/poller backpressure. A requested depth larger than `max_frame_slots` is
rejected instead of creating an async wait that can never complete.

`AnimationMetadata` carries the stream timebase, loop count, and timecode-presence flag.
`FrameMetadata` carries exact duration ticks, cumulative presentation-start ticks, and the
bitstream `timecode` when declared. The session rejects timebase, accumulated presentation tick,
or timecode-presence mismatches as typed errors. A cancelled async wait can be resumed through the
same session synchronously or by a later future.

The CPU/WGSL per-group Modular parameter ABI is a checked 256-byte `repr(C)` POD. Its first 12 bytes are the
shared `EntropyStreamParams`: token start/end bounds and the descriptor-derived LZ ring mask. The
same typed prefix starts the 240-byte, 16-byte-aligned VarDCT packet entropy record. Each consumer supplies its own
storage access and LZ scratch-base functions; geometry, prediction, output, and coefficient state
remain consumer-specific. Consumers whose entropy owns the complete token range also call one
shared terminator for the ANS final-state and at most seven zero-padding bits; VarDCT packet streams
followed by fixed metadata finalize ANS first and validate the enclosing section after that tail.
The VarDCT AC parameter record is a separate 160-byte aligned `Pod`. Its window suffix carries
logical/upload starts, available/full ends, the yield boundary, first/final flags, a 464-byte
execution-state offset, and the canonical status index. Its final four words select the pass's
entropy/order bases and spatial group index, with one padding word. The VarDCT packet record has its own seven
window fields, including a bounded-mode bit independent of FIRST/FINAL and a stream-base bit offset.
Combined/global-tree packets use those fields with 64-byte generic or 128-byte SelfCorrecting state;
five explicit words retain phase, LF/HF counts, first blocks, and extra precision. Staged local-tree
LF and HF packets use the same state sizes and reuse one allocation across their two sequential
stages.
The Modular suffix begins with six window fields: logical segment start, physical upload start,
full stream end, yield boundary, first/final flags, and the aligned entropy-state offset. It then
carries four plane offset/stride pairs, exact output
channel/order/depth/range/transfer codes, the resolved numeric mapping, global status index, MA
stream index, the proven fixed-leaf predictor/offset/multiplier and four channel cluster ids, the
output traversal mode, weighted-predictor header, and shader-visible logical size. The final three words are the unrotated canvas width/height and zero-based orientation. Records are a
tightly packed read-only storage array; a separate 16-byte uniform selects the global group range and local
scratch-lane stride for each wave.
Codestream segments are rounded to four bytes and include a zero sentinel word for bounded
cross-word peeks. A caller may set an additional cap with
`WgpuSubmissionEngine::with_stream_window_limit`; device limits and the shared per-slot byte budget
can only reduce it. Whole-group token offsets are rebased to their upload. Split groups instead keep
one group-relative logical cursor and map it into each upload through the segment fields. A token
may finish past a yield boundary inside the 16-byte forward overlap; the following segment includes
the same bytes as backward overlap and resumes only after the completed output value. Final ANS and
zero-padding validation runs only on the final segment. The peak window grows only when a larger
batch fits the actual per-slot shared byte budget and device storage-binding limit, allowing a large
still image to coalesce submissions without compromising concurrent small-frame admission.

Entropy metadata, bounded lane scratch including the selected 32/48/112-byte resume record, aggregate
status/readback, parameters, dispatch control,
output, and peak stream-window sizes are overflow-checked against storage, uniform, and device
buffer limits. Lane count is the minimum of the 512-lane watchdog cap, group count, device workgroup/storage
limits, and the scratch plus actual peak stream space affordable per requested frame slot. The LZ
ring itself is the next power of two above the largest reachable back-reference derived from the
distance histogram, hybrid-integer configuration, and group width; it is never sized by decoding
residuals on the host. A proven one-word ring uses invocation-private last-value state and consumes
no reconstruction storage; wider histories retain the descriptor-sized storage ring. If the requested slot count is
not affordable but one complete frame is, the prepared backend narrows it and propagates the
resolved bound into the actual session limiter and prefetch validation. `WgpuDecodeSession::memory_stats`
reports complete per-frame, output-lease, transient, peak-window, resolved-slot, logical/physical
LZ sizes, logical/physical reconstruction-sample workspace, per-lane execution-state bytes, parsed
Prefix/ANS representation, selected output-write/output-traversal
and reconstruction-specialization paths, lane/workgroup counts, stream batches, and actual
submission counts. Concurrent jobs opened through an engine or its
clones use the `WgpuBackend`'s shared
transient memory budget by default, so decode, encode, and generic readback apply one aggregate
admission bound. `WgpuSubmissionEngine::with_memory_budget` instead accepts an explicit cloneable
`MemoryBudget` for applications that intentionally define another sharing group. Admission is
non-blocking and memory pressure returns a typed, retryable error.

Transient stream-window/entropy-metadata/reconstruction/status bytes remain reserved until the
status map has completed. The shared host codestream `Arc` is not counted as GPU memory. Output
bytes are carried by `GpuBufferLease`, so explicitly cloning that lease extends
the same reservation and dropping the decode session cannot release it prematurely. GPU frame and
output containers are intentionally not cloneable. A raw `wgpu::Buffer` cloned through
`GpuBufferLease::as_wgpu_buffer()` remains valid wgpu ownership but is outside the byte budget; the
reservation returns after the final tracked lease is dropped. The same ownership contract applies
to every pending and presented frame of a stock Replace animation, including cancellation while
several Modular presentations are prefetched or a VarDCT LF cursor map is pending.

Repeated small and sequential decodes reuse a decoder-local, bounded cache for entropy metadata,
reconstruction, status, status-staging, and POD parameter buffers (plus the native-F64 dummy when
needed). A cache hit requires the exact allocation size, usage flags, and ABI alignment. The raw
JPEG XL codestream and caller-owned output are never admitted to this pool. Codestream upload reads
only each planned bounded range from a checked table of shared input spans, even when the range
crosses physical chunks, while metadata and packed 256-byte `ShaderParams` records (including the
12-byte shared entropy prefix) use `Queue::write_buffer`; no second full-codestream host `Vec` is
created.

Idle retention defaults to 32 MiB, 256 buffers total, and 32 buffers per exact key.
`WgpuSubmissionEngine::{buffer_pool_limits,set_buffer_pool_limits,clear_buffer_pool,buffer_pool_stats}`
expose limits, generation invalidation, hits/misses, idle/leased bytes and objects, and eviction
counters. An explicit clear invalidates outstanding generations without disrupting their GPU
work; those leases are destroyed after completion instead of re-entering the cache. A dropped
session or Future is also safe: the map callback retains every transient lease, then unmaps status
staging before returning allocations. The shared `MemoryBudget` still charges the complete active
logical job exactly once. Idle physical cache bytes are reported and bounded separately, rather
than being double-counted as active work.

`GpuPendingFrame::poll_complete` registers the latest supplied `Waker` and returns quickly while
status readback is pending. Native builds drive `Device::poll` on the backend's bounded completion
worker; poll admission is reserved before source consumption or queue submission, so saturation
is exposed in `PrefetchProgress` and can be retried without losing the source. Browser WebGPU uses
the polling/future API. There is no Tokio or async-std dependency.

The WGSL storage/uniform ABI and F64 output words are explicitly little-endian. Because the
`repr(C)` + `bytemuck::Pod` host structs use native endian, non-little-endian targets are rejected
at compile time rather than silently corrupting transport values.
