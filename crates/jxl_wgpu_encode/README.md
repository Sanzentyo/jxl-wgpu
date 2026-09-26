# jxl_wgpu_encode

GPU-required JPEG XL encoding orchestration for `wgpu`. This crate does not contain a CPU pixel
encoder or a CPU fallback. `LosslessModularEncoder` reads Gray, GrayAlpha, RGB, or RGBA integer or
binary floating-point pitch-linear storage directly on the GPU and emits a standards-compatible lossless
Modular codestream or `jxlc` container.

The complete encoder backlog, dependencies, and acceptance gates are tracked in
[`FULL_JPEG_XL_ROADMAP.md`](../../docs/FULL_JPEG_XL_ROADMAP.md). This README describes only the
currently executable profiles.

GPU-produced codestreams can be wrapped with Exif/XMP/JUMBF or other opaque payloads using
`jxl_gpu_bitstream::metadata::Metadata::write_container`. `MetadataBox::new` supports plain or
Brotli-compressed metadata with explicit size, ratio and window limits; `as_container_box` also
feeds `FragmentedContainerWriter`. Existing payloads can be preserved or replaced independently
of image encoding. [Metadata API and native interoperability](../../docs/CONTAINER_METADATA.md).

## Lossless Modular profile

- Extents are `1..2^30` on each axis, further bounded by the selected WebGPU device's storage
  binding, buffer, and dispatch limits.
- Valid sample depths are every integer in `1..=31`. The canonical constructor stores `1..=8`
  in one native `u8` word per
  component; `9..=16` use `u16`, and `17..=31` use `u32`. The valid sample occupies the low
  bits and high padding bits are ignored. `LosslessModularFormat::pixel_format` constructs this
  explicit storage/valid-bits contract, including native-U16 10/12-bit and native-U32 24/31-bit layouts.
- All 154 legal floating precisions are supported: one sign bit, 2–8 exponent bits and 2–23
  trailing significand bits. `jxl_gpu_formats::FloatPrecision::new(bits, exponent_bits)` checks
  these bounds; `LosslessModularFormat::custom_float_pixel_format(precision)` stores the raw words
  using `SampleKind::CustomFloat`. The existing `float_pixel_format(16/32)` retains its IEEE
  layouts and codestream bytes. Encoding preserves every bit, including signed zero, subnormals,
  infinities and NaN payloads, without floating-point arithmetic. Binary64 is outside JPEG XL's
  sample domain and remains rejected. Alpha shares the color sample precision.
- Custom `PixelFormat` layouts may partition Gray/GrayAlpha/RGB/RGBA components among one through four
  full-resolution planes in the same buffer. Packed, planar and split color/alpha layouts are
  supported, including BGR/BGRA and arbitrary bijective component swizzles. Gray also accepts
  `ColorModel::Gray` with `X001` and the color declarations below. GrayAlpha uses `ColorModel::Gray`
  with `X00W`, or another bijective gray/alpha component pair with zero Y/Z swizzle outputs.
- Each independently endian-addressed 8/16/24/32-bit word may contain multiple equally precise
  components and padding. Integer and floating fields may occupy any bit position, including
  MSB alignment; floating field widths must match the declared precision. Native, Little and Big byte order are supported.
  All components retain the same declared precision. Missing, duplicated or discarded components,
  signed samples, subsampling and unsupported color metadata are rejected.
- Plane offsets and row pitches may be unaligned, independently padded and physically reordered.
  Every public layout field is revalidated before admission. Four read-only source bindings
  address the planes directly; unused bindings alias the first. The kernel requires six storage
  bindings including parameters/artifacts, with typed rejection on devices configured below that
  count. Batches split at either source-plane or artifact binding limits, without a normalized
  image allocation or host pixel conversion.
- `Default` and `Undefined` RGB color specifications are interpreted as sRGB, matching the compact
  all-default JPEG XL color header. GrayAlpha and RGBA carry one alpha extra channel at the same
  declared sample precision as color.
- `with_alpha_association(AlphaAssociation::Associated)` declares that supplied color values are
  already multiplied by alpha. The default is `Unassociated`. This image-wide declaration applies
  to stills and every animation frame; encoding never multiplies, divides or discards source
  values, including invisible color at zero alpha. Associated input without an alpha channel is
  rejected before GPU admission. Independent alpha precision and additional extra channels remain
  unsupported.
- `Defined` full-range RGB/Gray accepts BT.709, BT.2020, Display-P3 and nonsingular custom RGB
  geometry, with D65, E, DCI or custom white. Linear, sRGB (including the Sycc alias), BT.709,
  PQ, HLG, DCI and Gamma transfer declarations are serialized without changing samples.
  Custom xy coordinates round to the nearest `1e-6`; Gamma's source OETF exponent must be in
  `[1/8192, 1]` and rounds to `1e-7`. Out-of-range coordinates and geometry that becomes singular
  after rounding are rejected. Gray retains white/transfer; its RGB primaries are not encoded.
  The YCbCr encoding field must be `Undefined`. BT.2020's distinct transfer, limited range,
  undefined transfers and non-RGB/Gray declarations remain unsupported.
- `ColorSpecification::Icc` embeds a structurally validated `IccProfile` with RGB or Gray device
  space, preserving every original byte, including private tags. `Rgb`/`Gray` models use their
  existing component swizzles. `IccDevice` instead uses `Swizzle::Device`, `Channel::Device(0..N)`
  for the profile's color components and optional `Channel::Alpha`. The source precision and
  storage rules above apply to both forms. No profile evaluation or image conversion is needed.
  `with_max_icc_profile_bytes` defaults to 16 MiB; zero disables ICC input. Original and transformed
  profile streams must also fit JPEG XL's 256 MiB limits. Limit failures use `EncodeError::IccLimit`
  before variable-sized header allocation or GPU admission. CMYK and other device spaces remain
  unsupported.
- `with_color_options(ImageColorOptions)` selects all four ICC rendering intents and
  a positive exact `FiniteF16` image white in cd/m². Defaults are Relative and 255 cd/m², also for
  HDR; the caller explicitly selects another known source white. This declares metadata and
  performs no tone mapping, primary conversion or alpha-association change. For ICC input, select
  the intent from `profile.header().rendering_intent`; a conflict is rejected instead of modifying
  the profile. Image metadata is validated before GPU admission. GPU storage and submission counts
  are unchanged; variable-sized ICC headers use the shared byte budget described below.
- `LosslessModularConfig::color_transform` selects `Auto`, `None`, `GlobalRct` or `LocalRct`.
  The default `Auto` uses YCoCg (wire type 6) for integer RGB(A) and no transform for Gray/GrayAlpha
  or floating input. `LosslessModularRctType::new(0..=41)` selects every normative operation and
  permutation; `IDENTITY` and `YCOCG` are named constants. Explicit RCT requires RGB(A), including
  embedded RGB ICC and floating samples. WGSL transforms raw words with wrapping integer arithmetic,
  preserving NaN payloads, signed zero and independent alpha. The host performs no pixel transform.
  Global RCT is declared once in DC-global; local RCT is declared in each pass group independently
  of MA-tree placement. A fused single-group frame declares either choice in DC-global.
  The same selected type applies to every group/frame; arbitrary Palette placement and adaptive
  per-group selection remain open. Invalid type/channel combinations fail before GPU admission.
- `LosslessModularConfig::local_transforms` accepts an immutable `LosslessModularSqueeze` policy
  via `.into()`, or an explicit `LosslessModularLocalTransforms::sequence` described below.
  Squeeze policies have named constants `None`, `Horizontal`, `Vertical`,
  `HorizontalThenVertical` and `VerticalThenHorizontal`.
  The default `None` preserves existing bytes. Other named policies transform all image channels
  after RCT and optional Palette, appending residuals in source-channel order.
  `with_channels(begin, count)` selects a nonempty contiguous range in that post-Palette image
  topology, excluding the meta table. For example, Palette on RGBA components 1–2 leaves image
  channels `[component 0, index, component 3]`; Squeeze range `(1, 1)` selects only the index.
  Unselected channels retain their dimensions and samples. Invalid ranges return
  `EncodeError::InvalidModularSqueezeChannels` before GPU admission, including on one-pixel groups.
  `with_in_place(true)` places residuals immediately after each step's selected range; the default
  appends them at the end. This selects wire channel order, not GPU buffer aliasing.
  Both-axis policies apply the second axis to all selected averages and residuals, using separate
  wire steps when tail placement leaves unselected channels between them. The descriptor retains
  the previous named values, but is no longer an integer-representable enum.
  Named policies skip group axes of length one; odd tails remain in the average channel.
  `LosslessModularSqueeze::sequence` accepts 1–296 ordered `LosslessModularSqueezeStep` values.
  Each step selects an axis, current image-channel range and residual placement, allowing repeated
  axes and selection of earlier residuals. `LosslessModularSqueezeStep::new` checks the wire bounds
  (begin at most 9,287 and count 1–19); the shared plan checks actual ranges, nonempty input channels
  and both cumulative shifts at most 30 before each step. Explicit steps retain zero-sized residual
  slots on one-pixel axes. Their ranges exclude the Palette meta prefix throughout.
  `steps()` exposes sequence descriptors; `channel_range()` and `in_place()` describe named
  policies and return `None` for sequences. `with_channels` rejects sequences, while
  `with_in_place` updates all their steps. Policies and `LosslessModularConfig` are `Clone`, not
  `Copy`; cloning a sequence shares immutable step storage.
  Each pass group declares its own transform, or DC-global declares it for a fused single-group
  frame. Named policies compute samples directly; explicit sequences materialize GPU intermediates
  in a planned reusable arena. Signed wide GPU arithmetic preserves normative average/tendency
  rounding without pixel readback. If any intermediate residual cannot fit a signed 32-bit
  working word, completion returns `BackendError::ModularSqueezeOverflow` and no codestream.
  Thus explicit Squeeze accepts only representable transforms of the integer/IEEE input domain;
  it does not promise every full-width source is representable. This applies before prediction
  under both entropy policies and both resident/streamed completion paths. Cross-group/global-LF
  Squeeze, arbitrary Palette placement, adaptive transform choices and
  progressive Modular remain open.
- `LosslessModularLocalTransforms::sequence` accepts 1–273 ordered
  `LosslessModularTransform::{Rct, Squeeze}` operations after the source color transform and
  optional Palette. Each entry emits one wire transform; each Squeeze entry holds one
  `LosslessModularSqueezeStep`. Its ranges address the current image channels, excluding the
  Palette meta table. RCT selects three consecutive channels at `begin_channel`, including
  alpha, Palette indices or earlier Squeeze residuals. All three dimensions and shifts must
  match; equal empty residuals are valid and require no GPU job. RCT uses wrapping words,
  including raw IEEE representations. The checked plan validates every group shape before
  admission and allocates all outputs before retiring any input span.
  The complete local header, including preceding local RCT/Palette, must fit 273 entries.
  In a fused single-group frame the source global RCT also belongs to this header.
  Invalid counts, ranges or unequal geometry return typed errors before allocation.
  `operations()` exposes the explicit program; `squeeze_policy()` exposes a converted
  Squeeze policy. Clones share immutable operation storage. Global/LF/HF programs and
  Palette interleaving remain unsupported.
- `LosslessModularConfig::palette` optionally selects `LosslessModularPalette::new(max_colors)`;
  the default is `None`. The checked limit is `1..=70_911`. Each pass group builds an exact
  first-occurrence dictionary of selected component tuples on GPU after RCT, including alpha
  and raw IEEE words. Signed zero and distinct NaN payloads remain distinct. The same policy
  applies to every group/frame, with an independent dictionary and actual color count per group.
  All components participate by default. `with_components(begin, count)` selects a nonempty
  contiguous post-RCT range; for example, `with_components(0, 3)` palettes RGB while preserving
  independent RGBA alpha. `component_range()` reports the explicit range or `None` for all.
  Empty, overflowing or out-of-format ranges return `EncodeError::InvalidModularPaletteComponents`
  before GPU job admission. This selection applies to every Palette policy below.
  Optional Squeeze transforms the index and unselected image channels and their residuals, skipping the palette
  meta channel. The host validates the GPU color count and token coverage before writing the
  local transform header; a fused single-group frame defers its DC-global header until then.
  Exceeding the limit returns `BackendError::ModularPaletteOverflow` without a codestream.
  Invalid constructor limits return `EncodeError::InvalidModularPaletteLimit`.
  `LosslessModularPalette::deltas(max_deltas, predictor)` instead stores exact wrapping residual
  tuples, with a checked `1..=66_816` entry limit and any of the 14 predictors. Prediction uses
  each post-RCT component's original neighbors and resets per group/component. The delta
  predictor is independent of the token predictor; both use the configured Weighted coefficients
  with separate state. Every dictionary entry is a delta and the wire color count is zero.
  `LosslessModularPalette::mixed(max_colors, max_deltas, predictor)` combines both dictionaries.
  It retains the first distinct absolute tuples up to the color limit, then uses residual tuples
  for other colors. An absolute match always takes priority. The two nonzero limits are checked
  independently; identical words in the two dictionaries remain distinct entries. Weighted delta
  state observes every original sample, including absolute entries. The wire table places used
  deltas before used colors, with no unused reserved entries. A group can use zero deltas.
  `max_colors()` reports total dictionary capacity, including deltas; `max_deltas()` reports the
  delta limit, and `delta_predictor()` is absent only for the color-only policy. Invalid delta
  limits return `EncodeError::InvalidModularPaletteDeltaLimit`; capacity overflow uses the same
  completion error and returns no stream.
  `LosslessModularPalette::implicit(max_deltas, predictor)` also selects exact implicit entries.
  It first matches the selected post-RCT tuple against the 64/125-entry cubes, then looks for an
  exact predictor residual among the 143 canonical signed entries, and finally stores an explicit
  residual. No component is rounded, including alpha or raw IEEE words. The explicit delta limit
  includes one declared zero entry, which avoids native libjxl's single-channel zero-delta index
  clamp. Cube selection is limited to working depths 1–24, where native and normative scaling
  agree; wider integers and binary32 retain implicit signed deltas and exact explicit residuals.
  Implicit entry components are relative to the selected range, so an alpha-only selection uses
  entry component zero. `uses_implicit_entries()` identifies this policy. Arbitrary stacks, table-free
  implicit policies and automatic policy selection remain open.
- `LosslessModularConfig` selects all four standard PassGroup sizes with
  `LosslessModularGroupSize::{Pixels128, Pixels256, Pixels512, Pixels1024}`, the MA-tree mode,
  reversible color transform, Palette, Squeeze, prediction, LZ77 and entropy coding.
  `LosslessModularEncoder::with_config` and `LosslessModularBackend::with_config` use the same
  immutable policy; `config()` reports it. The default remains 256×256. LF groups cover eight
  PassGroups per axis. Edge groups may be one pixel wide or high; cropped animation frames use
  their own extent to calculate both grids.
- `LosslessModularConfig::predictor` selects all 14 standard `LosslessModularPredictor` values:
  Zero, West, North, AverageWestNorth, Select, Gradient, Weighted, NorthEast, NorthWest, WestWest,
  AverageWestNorthWest, AverageNorthNorthWest, AverageNorthNorthEast and AverageAll.
  `weighted_predictor` accepts `LosslessModularWeightedPredictor::new([u8; 7], [u8; 4])` with
  coefficients 0–31 and maximum weights 0–15; out-of-range fields return a typed error. Every
  Modular header retains the parameters, including when the tree uses another predictor.
  One selected predictor applies to all groups/components; each group/channel resets its own
  state. The default remains Gradient and the default coefficients retain existing bytes.
  Predictor and parameter search, learned MA trees and previous-channel decisions remain open.
- `LosslessModularConfig::lz77` selects `LosslessModularLz77::ZeroRuns` (the byte-preserving
  default) or `Greedy`. Greedy computes every residual on GPU, then searches a three-symbol
  hash chain for arbitrary repeated sequences. It checks at most 32 prior candidates, retains
  the nearest equal-length match and supports overlap and regular distances throughout the
  channel's group history. A match covers at least seven residuals. The bucket count is the
  next power of two of the group pixel count, capped at 65,536. Prediction state still advances
  for every pixel; search history resets at each group/channel. Host code checks canonical
  length/distance events, history bounds, all histograms and exact sample coverage, then writes
  the selected entropy metadata. This is an explicit policy, without automatic effort selection.
- `LosslessModularConfig::entropy` selects `LosslessModularEntropyCoding::Prefix` (default,
  preserving previous bytes) or `Ans`. ANS consumes the GPU events in reverse channel/event
  order with one shared state per complete group, including Palette and Squeeze channels.
  Its four channel contexts plus distance share one to five distributions selected from all
  52 partitions by integer rate estimates plus serialized metadata, including repeated local
  headers. Joint selection includes 40 GPU-profiled split/MSB/LSB settings covering all u32
  residuals/distances and five length splits 0–4 covering the full 20-bit match domain.
  A checked coding plan resolves the LZ77 start symbol and 64/128/256-symbol alias alphabet
  together with these settings: 94 full-domain combinations, with exact extra-bit and repeated
  header costs. Normalization remains 4096 slots. The minimum 33 raw and 21 length symbols
  exclude a 32-symbol alphabet under this full-domain policy. Canonical tokenization and default
  Prefix bytes are unchanged. Host work selects bounded histogram metadata and only then builds
  the chosen reverse aliases; every ANS symbol, renormalization and extra bit is emitted on GPU.
  There is no fallback to Prefix. The immutable codebook owns the complete wire/GPU symbol domain
  and context map. This is a size estimate, without an always-smaller-stream guarantee;
  learned contexts, input-domain-specific configurations and effort policy remain open.
- One GPU invocation handles each PassGroup/channel pair without Palette. With Palette, the
  group's first invocation builds its dictionary and encodes all its channels sequentially;
  the remaining invocations return. This avoids cross-workgroup synchronization or extra submissions.
  Dispatch parameters and artifacts use
  group-major, channel-major order. Small jobs use one mapped artifact allocation. Larger jobs use
  complete-channel-group batches bounded by storage-binding and dispatch limits.
- Multi-batch jobs and every ANS job use a histogram pass to derive one frame-wide codebook.
  A second pass regenerates tokens, runs GPU ANS when selected, validates and assembles each batch,
  then releases its mapped artifact storage. ANS therefore uses two submissions even for one batch;
  the Prefix single-batch path retains one submission.
  Native builds drive the sequence with one runtime-neutral worker. Browser WebGPU drives the same
  two-pass sequence from map callbacks and the returned `Future`: each callback wakes the caller,
  and the next poll records exactly one next batch without requiring a Web Worker or a particular
  async runtime. Peak GPU memory is therefore bounded independently of total image area even though
  the final standard codestream remains contiguous.
- Every group/channel produces independent selected-predictor residuals, LZ77/raw token events, and
  histograms. The host validates every artifact, combines histograms for channels 0/1/2 separately
  and channels 3 onward together, creates the four
  JPEG XL channel distributions plus distance, and assembles channels inside standard row-major
  TOC groups. Prefix writes validated events on the host; ANS requires a completed GPU fragment
  with a bounded bit length, matching expanded symbol count and zero tail padding.
- LF global always carries a valid shared Modular tree and entropy code; LF groups and HF global
  are empty. `LosslessModularTreeMode::SharedGlobal` makes each PassGroup select that descriptor.
  `LocalPerGroup` instead writes a complete standards-compliant MA/entropy configuration after
  every pass-group header. The current local mode repeats the frame-trained codes, providing a
  deterministic interoperable policy and decoder/conformance input without pretending to perform
  independent per-group tree learning. A streamed 16K×1 RGB8 test exercises this mode across
  multiple bounded artifact batches through blocking and runtime-neutral completion.

`LosslessModularEncoder::memory_plan` reports the detected valid bits, exponent width (zero for
integers), largest component storage-word width, full and peak unions of source plane binding
ranges, the maximum transformed `channel_count` in any group, peak parameter/artifact/readback bytes, `weighted_predictor_scratch_bytes`,
`lz77_scratch_bytes`, `palette_scratch_bytes`, `transform_scratch_bytes`, `ans_output_bytes`, `hybrid_histogram_bytes`, diagnostic total artifact bytes, batch count, exact GPU submission count,
two-pass `streaming` mode, total encoder-owned live bytes, and the group grid before submission. Streamed jobs
report exactly twice the batch count:
one histogram and one serialization submission per batch. Every live batch uses the same shared
`MemoryBudget`. Its exclusive buffer-pool lease and reservation survive until the map callback and
mapped-range consumer are both finished, including when the returned future is abandoned.
ANS output is part of the same artifact allocation, lease and budget. For `E` maximum events
summed across a group's channels, it reserves a 16-byte completion record plus
`4 * ceil((80 * E + 32) / 32)` compressed bytes. Each batch also reserves 188,008 artifact bytes
for 40 × 5 × 235 hybrid histogram bins and two completion words. The five-table upper bound
reserves 92,180 parameter bytes, including each table's selected hybrid setting, plus a 44-byte
batch header, 16-byte group/channel descriptors and 40 candidate settings in an aligned suffix.
Only the selected one to five tables are uploaded and bound. Admission retains the upper bound
before the histogram pass, so selection never needs a late reservation or extra submission.
The corresponding artifact/readback and parameter allocations are included in admission;
`ans_output_bytes` and `hybrid_histogram_bytes` are subtotals, not additional allocations. One GPU
invocation owns each whole group's bit writer. This is a correctness baseline with no speed or ratio guarantee.
Source range accounting excludes gaps between planes and counts shared alignment prefixes once.

One checked transform plan resolves group-channel geometry, Palette capacity and ordered wire
operations before allocation. Dispatch and all three resident/native-streamed/browser-streamed
assembly paths share it; GPU parameters carry the resolved sample source, Squeeze band or arena
offset. Ordered sequence jobs and their metadata/sample capacity belong to that same plan.
Actual Palette counts must validate against that plan before they can determine a header.
Streamed submission checks the reported peak against available budget before allocating its first
batch, even when a later batch is larger. Each batch still acquires and retains its own reservation;
concurrent allocations between batches can cause a later typed backpressure failure with no output.

The selected `group_size` is included in `group_grid`. Larger groups increase each channel's
worst-case event artifact to `400 + 16 * (pixels + ceil(pixels / 8) + 1)` bytes. Weighted adds
`20 * group_width` bytes of row state per channel inside that artifact allocation; its reported
scratch subtotal is already included in owned bytes and any separate readback copy. Greedy adds
`8 * pixels + 4 * hash_buckets` bytes per channel for residual words, chain links and bucket heads,
also inside the artifact allocation and its reported scratch subtotal. A complete group must fit
the checked source/artifact binding limits. With Squeeze, these formulas use each transformed
channel's width, height and area. Named policies produce at most 16 image channels per group;
single-pixel edge axes may produce fewer. Palette adds one meta channel and replaces its selected
components with one index channel. Explicit sequences add each step's count to the current channel
list, including empty residual slots; their exact topology determines event and predictor storage.
Each program's `transform_scratch_bytes` includes a job-count word, 64 bytes per GPU job
(one selected Squeeze channel or one nonempty RCT triple), and the peak live sample arena.
This private region is inside the artifact/readback allocation. The job table is also charged
in parameter storage and copied to the private region
before execution. Arena reuse never overwrites a job's still-live inputs.
Its meta channel reserves `k × selected_components` samples, where `k` is the sum of the separately
pixel-clamped color and delta limits. The implicit policy instead clamps its delta limit to
`group_pixels + 1`, including its declared zero entry. Only declared entries are encoded.
For capacity `k` and `c` selected components, Palette adds
`4 * (2 + k * c + next_power_of_two(2 * (k + i)))` scratch bytes per group, where `i` is 143 for
implicit lookup and zero otherwise. This covers total/delta counts, the dictionary and hash table. The peak subtotal is
included in the artifact allocation and any readback copy.
Delta, mixed and implicit modes additionally retain `4 * group_pixels * selected_components` residual bytes and,
for Weighted delta prediction, `20 * group_width` row-state bytes reused between components.
These are included in `palette_scratch_bytes`, separately from token-predictor scratch.
These private scratch bytes are mapped with that allocation; host assembly reads the counts and
encoded events, without inspecting dictionary entries or source pixels. Selecting 1024 does not guarantee that every device
or memory budget can admit it. Batch splitting, peak reservations and exact submission counts
are recalculated from that geometry. Long zero runs use the full valid prefix alphabet through
the 1024²-sample case; histogram, canonical extra-bit and exact sample-count checks remain required.

`icc_profile_bytes` reports the original caller-owned profile retained by a source or animation
descriptor and included in addressed bytes. `icc_storage_bytes` adds twice the complete image-header
size to owned/addressed bytes: one header plus the possible temporary copy during exact resizing
or container wrapping. The encoder reserves this amount before writing the ICC payload; it writes
header predictions and literal metadata directly, without an intermediate transformed-profile
allocation. The permit survives assembly and retires on completion even if the completed future
remains alive. Cancellation drops the host header while active GPU batches keep their own permits
through completion. Returned codestream/container vectors are caller-owned after completion.
The frame-only backend reports zero ICC storage; an animation session admits one image header at
creation and holds it until finishing or dropping, independently of its frame jobs.

The returned `LosslessModularSubmission` implements `Future` without depending on an async runtime;
native callers may instead use `wait`. Browser builds intentionally reject blocking `wait`, because
WebGPU completion is delivered by the browser event loop. Dropping an in-progress browser future
keeps the active batch's lease and shared byte-budget reservation alive through its map callback,
then releases them without submitting another batch. `group_grid` and `ordered_groups` expose the
exact dispatch rectangles and normative PassGroup order before completion.

```rust,no_run
# use jxl_wgpu_encode::{
#     BufferImageSource, LosslessModularConfig, LosslessModularEncoder, LosslessModularFormat,
#     LosslessModularColorTransform, LosslessModularGroupSize, LosslessModularRctType,
#     LosslessModularSqueeze, LosslessModularTreeMode, WgpuContext,
# };
# fn submit(
#     context: WgpuContext,
#     source: BufferImageSource,
# ) -> Result<(), jxl_wgpu_encode::EncodeError> {
let encoder = LosslessModularEncoder::with_config(
    context,
    LosslessModularConfig {
        group_size: LosslessModularGroupSize::Pixels512,
        tree_mode: LosslessModularTreeMode::LocalPerGroup,
        color_transform: LosslessModularColorTransform::LocalRct(LosslessModularRctType::new(41)?),
        squeeze: LosslessModularSqueeze::HorizontalThenVertical,
        ..Default::default()
    },
);
let plan = encoder.memory_plan(&source)?;
assert_eq!(plan.group_grid.groups, plan.group_grid.columns * plan.group_grid.rows);
assert!((1..=32).contains(&plan.bits_per_sample));
let precision = plan.sample_bit_depth();

// Use this descriptor when constructing a packed native-U16 RGB10 source layout.
let _rgb10 = LosslessModularFormat::Rgb.pixel_format(10)?;
let _rgba_half = LosslessModularFormat::Rgba.float_pixel_format(16)?;

let submission = encoder.submit_container(source)?;
let source_format = submission.format();
for group in submission.ordered_groups() {
    // `group.index` is also its standard row-major PassGroup/TOC index.
    let _rectangle = (group.x, group.y, group.width, group.height);
}
let jxl_container = submission.wait()?;
# let _ = (jxl_container, source_format, precision);
# Ok(())
# }
```

Single-group Gray8 Prefix containers with default color/intent/intensity additionally carry the optional
private `jwgp` acceleration index. Explicit sRGB matching those defaults retains the same bytes.
Its current schema represents one contiguous 8-bit single-channel token span, so other depths,
GrayAlpha, RGB(A), explicit Palette/Squeeze, and multi-group containers intentionally omit that private box; all remain ordinary
interoperable JPEG XL containers. Conformance tests cover every depth `1..=31`, the
1/255/256/257 group boundaries, and extreme aspect ratios. A streamed 16,384×1 RGB8 case is exact
through both the published Rust `jxl` decoder and reference `djxl`, with identical blocking and
runtime-neutral Future codestreams. Browser/WASM compilation covers that same multi-batch state
machine; browser execution still requires a WebGPU-capable page and executor/event-loop integration
provided by the application.

The [wide-integer matrix](../../docs/CONFORMANCE_CORPUS.md#wide-integer-modular-encoding)
checks all 17–31-bit depths with shared and local trees. Original integer planes from jxl-oxide
must match every source word exactly; native libjxl independently checks normalized output.
Whole and bounded fragmented GPU decoding also preserve every source word, including values
beyond F32's exact integer range. Full-canvas Replace animations cover 17/24/31-bit timing and
retained output; streamed 16K×1 and resident RGBA31 cases cover exact budget admission,
cancellation and reuse.
This exact-word claim does not extend to floating-point animation composition.

The [IEEE floating-point matrix](../../docs/CONFORMANCE_CORPUS.md#ieee-floating-point-modular-encoding)
checks raw working words with jxl-oxide and original F32 output with libjxl, including every
binary16 word and both signs at every binary32 exponent. Whole and bounded GPU output matches
exact F32 bits for all selected components, independent alpha and RGBA presentation. Replace
animations preserve those words and retained outputs; finite crop/Add/Multiply animations match
both CPU decoders and the GPU. Arithmetic composition follows floating-point blend semantics;
it is not a promise to preserve original source words after arithmetic. Resident and streamed
RGBA32 jobs retain the existing exact admission, cancellation and pool-reuse contract.

The [custom floating matrix](../../docs/CONFORMANCE_CORPUS.md#custom-floating-point-modular-encoding)
extends those checks to every legal precision under Prefix and ANS, with independent original
words, exact native F32 output, both tree placements, arbitrary field/word packing and bounded GPU
decoding. Additional cases cover RCT/Palette/Squeeze/Weighted composition, Replace animations,
precision mismatch before admission, IEEE byte compatibility and resident/streamed ownership.

`EncodeProfile::ModularLossless` carries `sample_bit_depth: SampleBitDepth`, distinguishing
integer depth from floating depth and exponent width. Matching storage widths do not permit
changing numeric type between frames. `LosslessModularAnimationDescriptor::from_pixel_format`
also infers the stream's enumerated color declaration or retains its original ICC profile. Frames
may change physical layout but must keep the same serialized color, precision and logical channels;
ICC identity compares every original byte. Mismatch leaves the next frame index unchanged. The
descriptor is `Clone`, and `session.descriptor()` borrows it. The existing `new`/`new_float`
descriptors select default sRGB/gray.
The [source-color matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-source-color)
checks independent native profile bytes, exact original samples, requested color conversion and
Replace animations. The [alpha-input matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-alpha-input)
checks exact GrayAlpha/associated source words, all three output alpha policies and six-frame
compositions against independent native/Rust decoders, with retained output and cancellation.
The [embedded-ICC matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-embedded-icc)
checks independent original-profile bytes and source words, requested color output, private tags,
all intents, animation identity, size limits and shared-budget lifetime.
Independent extra-channel declarations, CMYK, YUV and textures remain outside this profile.
The [group-size matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-group-sizes) covers every
size with shared/local trees, integer/IEEE source words, LF boundaries, full tiles, cropped and
Replace animations, bounded GPU output and admission/cancellation. The default 256 configuration
retains the existing checked-in Gray8 codestream bytes.

The [predictor matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-predictors) checks all
14 predictors, custom Weighted parameters, exact integer/IEEE words, streamed ownership and
comparison against Gradient on a shifted-row source. Gray8 containers attach the private `jwgp`
shortcut only for Prefix with Gradient and ZeroRuns; other policies use the standard Modular path.

The [general LZ77 matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-general-lz77) adds
independent distance/overlap checks through the 2²⁰ history limit, all predictors and group
sizes, integer/IEEE words, high-depth RCT, animation and streamed resource ownership. A periodic
source produces fewer bytes with Greedy than ZeroRuns; no universal compression gain is claimed.

The [Palette matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-palette-encoding) checks
every color-count wire bucket through 70,911, all integer precisions, raw IEEE special words,
RCT/predictor/Squeeze composition, retained animation output and bounded GPU decoding. Exact
budgets, cancellation and late streamed capacity failures retain the existing ownership contract.
The [Delta Palette matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-delta-palette-encoding)
uses the pinned scalar libjxl's original integer planes, including low bits beyond F32 precision,
and checks the full delta-count wire domain and all predictors. A smooth Gray31 source verifies
successful encoding with four deltas and fewer bytes than the ordinary Gradient stream, while
an exact four-color palette rejects it; this does not establish a universal compression gain.
The [mixed Palette matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-mixed-palette-encoding)
checks the simultaneous 70,911-color/66,816-delta maximum, equal words in different partitions,
all predictors, integer/IEEE/RCT/Squeeze composition, animations and resident/streamed ownership.
It uses the same independent native exact-word and F32 oracles without changing their bounds.
The [implicit Palette matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-implicit-palette-encoding)
uses native implicit entries as inputs and audits native-decoded indices before inversion.
Both cubes, signed entries, wider exact residuals, RCT/Squeeze, animation and lifetime checks keep
the original-word and F32 bounds. The declared zero entry and depth policy preserve interoperability.
The [component-selection matrix](../../docs/CONFORMANCE_CORPUS.md#lossless-modular-palette-component-selection)
checks every contiguous range in the four input formats with all four policies, unchanged
unselected components, every integer/IEEE precision, RCT/Squeeze composition, relative implicit
entries, animation, invalid ranges, overflow and resident/streamed lifetime.

## Experimental VarDCT profile



`VarDctEncoder::new` takes an explicit `VarDctStrategy` and accepts one packed/planar/split sRGB8
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
`color_transform` selects `VarDctColorTransform::Xyb` (default) or `Original`. Both accept
the configured integer or floating Gray/RGB source layout. `source_color` declares full-range
enumerated BT.709, BT.2020, Display-P3 or custom primaries with D65, E, DCI or custom white;
Linear, sRGB/Sycc, BT.709, PQ, HLG, DCI and checked Gamma transfers are supported.
The default remains sRGB/D65. `color_options: ImageColorOptions` binds all four rendering
intents and positive exact binary16 image white (default 255 cd/m²), shared with Modular.
XYB applies the shared GPU EOTF, PQ/HLG display luminance, Bradford-adapted primary matrix,
image-white scaling and opsin transform. Original directly transforms normalized components.
`color_transform()` and `source_color()` report the configured policy;
`VarDctConfig::pixel_format()` constructs canonical storage including its color declaration.
One checked color plan owns image/frame syntax, logical-to-working component mapping, GPU
normalization and HF multipliers. It expands serialized custom xy/gamma values before GPU
lowering, so metadata and arithmetic use the same rounding. Every admitted frame must have
wire-equivalent color, channels and precision; explicit sRGB/Sycc aliases may match defaults.
Undefined color, unsupported transfers, limited-range RGB/YUV and unrepresentable geometry
are rejected before job admission. Original components omit XYB-only matrix-scale fields and
use their implicit neutral multipliers.
LF metadata, matrices, orders and quantizers retain their explicit values in either domain;
switching the domain does not imply equivalent bitrate or perceptual quality.
The [original-RGB corpus](../../docs/CONFORMANCE_CORPUS.md#original-rgb-vardct-encoding)
covers all three backends, sequences, progressive passes and independent pixel/coefficient checks.

`source_color: ColorSpecification::Icc(profile)` also accepts RGB or Gray profiles with matching
logical channels, ordinary swizzles or explicit `IccDevice` components. The original profile is
preserved byte-for-byte; `color_options.rendering_intent` must equal its unchanged header intent.
`max_icc_profile_bytes` defaults to 16 MiB; zero disables ICC input, and the standard's original
and transformed profile limits also apply. Original coding preserves normalized device components
without selecting an ICC method. XYB selects the profile's relative device-to-PCS method and
uses the existing resident GPU ICC pipeline to produce linear BT.709 or one linear Gray component
(PCS Y). Gray expansion occurs after that connection. The working intent remains Relative
independently of the preserved presentation intent. No host pixel conversion is performed.
Unsupported selected methods and resource limits fail before submission; finite source values
whose color conversion becomes nonfinite return `BackendError::VarDctColorConversionNonFinite`
before any packet is exposed. [ICC evidence](../../docs/CONFORMANCE_CORPUS.md#icc-vardct-and-mixed-input).

The common frame plan rejects post-color-transform reference storage for XYB plus ICC, as required
by F.2. This includes non-final hidden frames and reference-only frames. Final stills and timed,
unretained XYB presentations are supported; Original ICC sequences retain the existing reference
and crop/blend contract. VarDCT pre-transform reference storage remains unimplemented.

`VarDctConfig::sample_format` uses checked `ColorSampleFormat::integer(channels, bits)` for all
1–31-bit unsigned sources or `ColorSampleFormat::float(channels, bits, exponent_bits)` for all 154
legal floating precisions (2–8 exponent and 2–23 fraction bits). `ColorChannels::Gray` selects
one logical component; `ColorChannels::Rgb` selects three. The default remains RGB8.
RGB may be packed, planar or split across full-resolution planes in one buffer, with
bijective RGB component swizzles, shared or separate 8/16/24/32-bit words, arbitrary valid
sample bit positions and Native/Little/Big byte order. Component word widths may differ;
logical sample and exponent widths must agree. Byte offsets and independent row pitches may
be unaligned, and physical planes may appear in any nonoverlapping order. Padding bits are
ignored. `sample_format.pixel_format()` remains a canonical native-endian interleaved format
with separate 1/2/4-byte component words and right-aligned samples.

Gray uses `ColorModel::Gray`, the configured white/transfer and a single stored channel selected by
the gray swizzle (X, Y, Z or W, independently of optional alpha). Its checked source record is bound to each of
the three standard VarDCT working components. The GPU reads the same source sample for
each, while the header declares Gray and omits RGB primaries. Source windows count those
bytes once; enumerated/original Gray needs no expanded input buffer or host pixel conversion. Gray keeps the
same word sizes, bit positions, byte order and unaligned row/offset support as RGB.

`VarDctConfig::alpha: Option<AlphaAssociation>` adds full-resolution GrayAlpha or RGBA at the
same sample precision. `None` keeps color-only input; `Some(Unassociated)` and `Some(Associated)`
declare the supplied association without multiplying, dividing or replacing invisible color.
`config.pixel_format()` includes alpha; `config.sample_format.pixel_format()` describes color alone.
`alpha_association()` reports the checked declaration on VarDCT and mixed encoder handles.
One immutable image-sample plan owns the logical component count, alpha source index and extra-channel
metadata for both codecs. The three color working planes retain their existing transform contract.

Alpha is compressed losslessly from the original GPU words, including floating signed zero,
subnormals, infinities and NaN payloads. A separate checked Modular side-plane plan routes images
up to 256×256 to LF-global and larger images to 256-pixel pass groups, at the first declared
full-resolution endpoint or the final pass. Row tasks read neighboring original samples within
their group and produce stateless Gradient/prefix fragments; host assembly concatenates only
validated compressed bits. The same plan supplies GPU ranges, resource bounds and packet placement.
`VarDctMemoryPlan::alpha` exposes parameters, artifact and readback bytes, all retained by the
existing job reservation/completion owner. No additional submission or map is introduced.
[Alpha conformance](../../docs/CONFORMANCE_CORPUS.md#alpha-vardct-and-mixed-input) covers exact words,
independent native output, color independence, layouts, progression, composition and ownership.

`VarDctConfig::extra_channels` declares additional scalar sources through checked
`ExtraChannel` values. `SamplePrecision` accepts the same 31 integer and 154 floating precisions
independently of color. Alpha (either association), Depth, SpotColor, SelectionMask, Black, CFA,
Thermal and Optional are supported. Spot metadata uses finite binary16 values; CFA indices are
0–274. UTF-8 names retain their exact bytes through 1071 bytes. At most 256 resolved channels,
including packed alpha, are admitted. `max_extra_channel_metadata_bytes` defaults to 1 MiB and
bounds the sum of name bytes plus 32 bytes per declaration before serialization.

Attach one `BufferImageSource` per configured extra with `source.with_extra_channels(inputs)`.
These sources follow `config.extra_channels` order; optional packed alpha precedes them in the
image header and needs no attachment. `encoder.extra_channels()` returns this resolved order;
`alpha_association()` continues to describe packed alpha only. Each attachment is a single scalar
STORAGE-buffer view, using the shared byte-order, bit-position and unaligned-pitch addressing.
`SamplePrecision::pixel_format()` provides its canonical non-color layout. For intrinsic shift
`s` in 0–3 and color factor one, the default is `ceil(frame_width / 2^s)` by `ceil(frame_height / 2^s)` samples. Geometry is
relative to each physical frame/crop. Nested attachments, count/precision/extent mismatches and
unreadable buffers reject before admission; input planes may alias the same allocation.

All extra samples are encoded losslessly on GPU. A common image plan resolves the leading small
channels that belong to global Modular; after the first channel larger than 256 on either axis,
shifts relative to the coded color grid of three through six use LF groups and shifts zero/one/two use the applicable progressive
pass. Small channels after that boundary stay in their LF/pass streams. The serializer consumes
the same plan. Presentation interpolation is separate from exact encoded sample preservation.

`FrameOptions::upsampling` selects `UpsamplingFactor::{One, Two, Four, Eight}` for color
reconstruction in Modular, VarDCT and mixed sequences. Supply already reduced pixels of extent
`factor.source_extent(frame_extent)` (ceil division); the encoder does not downsample source
pixels. Canvas, crop offsets/sizes, blending and references remain in displayed pixels. Source
layout, transform maps, groups, progression and GPU memory use the coded extent. Modular retains
every supplied word exactly; interpolated display pixels are a separate reconstruction contract.

`FrameOptions::extra_channel_upsampling` selects a factor per resolved extra channel, including
reference-only frames. An empty list uses the color factor for every channel; a nonempty list
must include every channel, starting with packed alpha when present. Packed alpha must equal the
color factor. Independent VarDCT planes use `factor * 2^s`, through 64, and that effective factor
must be at least the color factor. `ExtraChannel::source_extent_with_upsampling` computes the
required reduced extent; `source_extent` retains factor-one behavior. The shared
`UpsamplingFactor` replaces `ExtraChannelUpsampling`. One checked frame plan owns coded geometry,
wire factors and relative shifts for source validation, global/LF/pass routing and admission.

Use `begin_sequence` with `AnimationHeader::Still` and one `submit_last_frame` to select factors
for a still. Timed and layered sequences can change color and extra factors per physical frame;
image metadata and canvas stay fixed. `memory_plan_for_request` reports their exact resources;
the ordinary `memory_plan` and `encode`/`submit` methods retain factor-one defaults. Invalid
counts, incompatible color/extra factors and source extent mismatches reject before admission
without consuming frame finality. [Sampling evidence and oracle limits](../../docs/CONFORMANCE_CORPUS.md#per-frame-vardct-extra-sampling)
cover factors through 64, exact coded words, interpolation, references and ownership.
[Color sampling coverage](../../docs/CONFORMANCE_CORPUS.md#per-frame-color-and-relative-extra-sampling)
adds both codecs, all VarDCT topologies, mixed references, odd/thin group boundaries and
immutable progressive output. Independent Modular/mixed extras remain unsupported.

`VarDctMemoryPlan::extra_channels` includes every scalar plane, while `alpha` is the overlapping
packed-alpha breakdown (`VarDctAlphaMemoryPlan` remains a compatibility alias). Aggregate readback
follows optional raw matrices in one map and is charged once. `source_binding_bytes` counts the
union of aliased caller-owned windows. Jobs retain source buffers and all scratch until GPU
completion. Variable image metadata has a still/session reservation through assembly or drop;
ICC headers already include it, so `extra_channel_metadata_bytes` is zero for ICC input.

Fixed-VarDCT sequences retain these declarations across frames. Supply either an empty
`extra_channel_blends` list for all-Replace or one entry per resolved channel. `FrameBlend::alpha_channel`
selects an existing extra index 0–10 for Blend/MultiplyAdd; other modes require zero. Each plane
retains its own reference slot and clamp contract. [Conformance evidence](../../docs/CONFORMANCE_CORPUS.md#independent-vardct-extra-input)
covers exact words, shifted progressive streams, native/GPU composition and ownership.
Independent extras for complete Modular or mixed-codec encoding remain unsupported and are
explicitly rejected. CMYK input, NonOptional/unknown semantics and per-extra lossy encoding are
not introduced by the Black/Optional scalar declarations.

The logical input API is `ColorSampleFormat` and `ImageSequenceDescriptor`; these replace
the former RGB-only names. A sequence fixes channels and precision together, even when
its physical frames change layout or codec. The color plan independently binds the declared color and image-white options.

The shared checked source plan owns physical addressing and per-plane binding windows for
both Modular and VarDCT; the immutable color/sample plan owns logical normalization and image
metadata. Neither dispatch path nor the serializer independently reconstructs the layout.
The same GPU byte/word loader feeds Modular, general/tiled VarDCT and local-contrast ordering.
IEEE binary16/binary32 accept `SampleKind::Float` or equivalent `CustomFloat` spelling.
Integers divide by `2^bits - 1`; floats rebase their fields into F32 on GPU.

VarDCT checks raw floating exponent fields on GPU, returning typed
`BackendError::VarDctNonFiniteSource` for NaN or either infinity before exposing any frame.
Finite values that overflow quantized i32 coefficients retain the distinct quantization error.
Original components retain finite negative/greater-than-one values within that quantizer range.
XYB uses each declared transfer's extension and its existing nonnegative opsin-absorbance clamp;
no arbitrary extended-range reconstruction or general quality guarantee is implied.
Subnormals enter the lossy F32 arithmetic contract; VarDCT is not a bit-preserving float codec.
General transforms budget per-workgroup completion/error records inside the artifact/readback;
tiled DCT8 carries errors in its existing block records. Missing or malformed validation
records cannot publish an artifact. XYB ICC additionally budgets those records for integer and tiled sources.
Chroma subsampling, per-extra lossy distance, automatic source downsampling and texture inputs remain outside the VarDCT contract.

VarDCT and mixed sequences declare 32-bit Modular working buffers, independently of input
depth: their quantized LF coefficients are checked i32 values. This corrects the earlier
16-bit declaration, which could truncate large LF values in decoders using i16 storage.
The image header therefore uses explicit metadata even for default RGB8/XYB. Source precision
and codec working precision are separate; existing quantization-overflow rejection remains.
The [integer-input corpus](../../docs/CONFORMANCE_CORPUS.md#integer-rgb-vardct-input) records
independent coefficient/pixel, source-layout, sequence and ownership evidence; the
[floating-input corpus](../../docs/CONFORMANCE_CORPUS.md#floating-rgb-vardct-input) adds
all floating precisions, nonfinite rejection and completion ownership. The
[source-layout corpus](../../docs/CONFORMANCE_CORPUS.md#shared-encoder-source-layouts) covers
physical layout variants and per-frame layout changes. The
[Gray-input corpus](../../docs/CONFORMANCE_CORPUS.md#gray-vardct-and-mixed-input) adds all precisions,
all 27 strategies, cross-codec references and per-frame Gray layouts, retaining independent
coefficient/pixel bounds and documenting external whole-stream oracle limitations.

See the compiled configuration example on [`ColorSampleFormat`](src/sample_format.rs).

The config's
`lf_metadata` field holds validated `VarDctLfMetadata`. Its LF dequantization and base-correlation fields retain exact finite
binary16 values, while the colour factor and signed LF factors use their normative integer
domains. Construction rejects dequantized coefficients below libjxl's `1e-8` threshold, colour
factors outside `2..=65793`, and base correlations outside `[-4, 4]` with typed `EncodeError`
variants. Default and explicit bundles share one serializer, and both single-transform and tiled GPU
kernels subtract the selected LF chroma-from-luma slopes and quantize with the selected channel
dequantization multipliers. Generated explicit-metadata streams are parsed back by the stock
frontend and agree across Rust `jxl`, the stock GPU decoder, and optional `djxl` within one RGB8
code; blocking and runtime-neutral Future assembly are identical.

The GPU executes the selected color normalization, forward transforms, LF/AC quantization, the per-8×8
clamped-Gradient DC predictor, signed tokenization, prefix packing, histogramming, and the
standard strategy map. All 27 strategies and `TiledVarDctEncoder` use default or caller-selected
parametric/raw dequantization matrices and natural or caller-selected coefficient orders, with one prefix distribution
for all 495 coefficient contexts and no LZ77. `VarDctQuantization` validates exact global scale
`1..=73728`, LF quantizer `1..=65536`, and a default `VarDctHfMultiplier` in `1..=256`.
`VarDctTransform::with_hf_multiplier` overrides the default for that transform; sorting a map
preserves its associated multiplier. GPU quantization and serialized LF/HF metadata use these
same values. Defaults are `(8813, 10, 6)` and carry no perceptual-distance claim. The former
`PerceptualDistance` API was removed; general distance/quality guarantees, adaptive selection,
and rate control remain unimplemented.

`VarDctConfig::coefficient_orders` accepts `VarDctCoefficientOrders`, shared by single transforms,
mixed maps and tiled DCT8. `with_order(strategy, [x, y, b])` validates three permutations of
natural ranks `0..width*height`; the first `width*height/64` LF ranks must stay in place. All
13 JPEG XL size classes are supported, including the separate special-8×8 class. Transposed
rectangles share a class. A later call replaces that whole class; unspecified classes and explicit
identity permutations use natural order. Length, range, duplicate and LF-prefix errors are typed.
The GPU quantizes every AC location and serializes each channel in its selected order. Host work
only validates and serializes caller-supplied permutation metadata using bounded Lehmer coding;
it neither selects orders from pixels nor reads coefficients. Config clones share immutable tables.
Content-adaptive order selection remains a separate roadmap item.

`VarDctConfig::dequant_matrices` accepts `VarDctDequantMatrices`. Its
`with_matrix(strategy, VarDctMatrixEncoding)` selects modes 0–6 for all 17 matrix families;
transposed strategies and AFV orientations share their family's immutable parameters.
Parameters retain exact `FiniteF16` wire values. DCT band vectors must have equal X/Y/B
lengths in `1..=16`; modes 1–5 require an 8×8 family. Construction rejects incompatible
encodings and expanded scales outside the finite interval `(0, 1e8)` before GPU work.
`Default` restores a family's standard matrix. Bounded scalar expansion is shared with the
decoder, including the normative ×64 wire scaling of Hornuss and DCT2 parameters. The GPU
uses the resulting scales for quantization; no image samples or coefficients are processed
on the host.

`with_raw_matrix(strategy, denominator, [x, y, b])` selects mode 7 for the same shared families.
Each channel contains exactly `transform_width * transform_height` positive `i32` samples in
the wire raster, whose width is `min(transform_width, transform_height)`. Transposed strategies
share these samples without transposing the flattened matrix. The positive `FiniteF16`
denominator times each sample must be finite and in `(0, 1e8)`. `raw_matrix(strategy)` exposes
the validated `VarDctRawMatrix`; `encoding(strategy)` returns only parametric metadata.
GPU work performs Gradient prediction, signed tokenization and prefix packing for each raw
Modular side image, using the global tree without transforms or LZ77. Host validation checks
the fragments against the caller's bounded matrix metadata before appending their bits to
HF-global. Default, parametric and raw families can be interleaved. Content-adaptive matrix
selection remains unimplemented.

`VarDctConfig::progressive` selects a validated `ProgressivePlan` of 1–11 spectral/quantized
AC passes; the default is one complete pass and preserves the existing single-pass bytes.
Each `ProgressivePass::coefficient_square` is a size in `1..=8`, measured in eighths of both
canonical frequency axes, with the longer axis horizontal. `shift` is in `0..=3`. A pass adds
frequencies or reduces the shift at its current size; the last entry must be `(8, 0)`.
The GPU divides the remaining signed coefficients by `2^shift` toward zero and emits each
contribution independently of the caller's coefficient order. Prior contributions are subtracted
only where their spectral rectangle included that coefficient. Thus increasing spectral size
can also increase shift without losing newly introduced frequencies, and all passes reconstruct
the exact single-pass quantized coefficients.

`ProgressivePlan::with_downsampling` adds up to four `ProgressiveDownsampling` stopping points
to a multi-pass plan. Factors decrease through `8`, `4`, `2`, `1`; zero-based `last_pass` indices
increase, are less than the pass count, and fit the wire syntax's `0..=7` range. These declarations
do not change coefficient splitting. Initial DC detail of eight and final detail of one remain
implicit. An empty list preserves the previous header bytes.

`VarDctConfig::group_order` defaults to raster order. `VarDctGroupOrder::center_first()` starts
at the group containing the middle image pixel; `centered_at(x, y)` uses a caller-selected
source pixel. Both visit concentric Chebyshev group rings, then sort by squared distance from
that pixel to the clipped group's center, breaking ties by raster ID. Integer geometry makes
the order deterministic. `explicit(Vec<u32>)` accepts each raster AC-group ID exactly once,
with at most 4096 groups and an exact match to the submitted image's grid. Invalid centers or
grid lengths fail before GPU admission. LF-global, LF groups and HF-global remain first;
every AC pass repeats the selected group order. A standard entropy-coded TOC permutation
records the physical order. Identity order preserves the existing bytes. Caller metadata and
geometry drive this bounded host assembly; there is no CPU image analysis.

`VarDctGroupOrder::saliency_first()` selects order from GPU-computed local contrast. For each
visible RGB pixel, it sums absolute differences in an integer 0–255 proxy to the existing left and upper image
neighbors. Precisions above eight use the top eight valid bits; smaller precisions scale to
0–255 with nearest-integer rounding. This only orders groups; the forward transform retains
full source precision. An edge crossing a group boundary belongs to the group containing the right/lower
pixel. Padding is excluded. Groups sort by mean summed RGB difference per edge, using exact
integer cross-products and raster-ID ties; the one-pixel image has zero score. This is a local
contrast heuristic, not a general perceptual-quality or bitrate guarantee.

One GPU reduction pass computes a checked 16-byte record per group, followed by host sorting
of only those bounded statistics. Pixels and coefficients remain on the device. The records
share the existing artifact and aggregate readback, with no additional submission/map.
`VarDctMemoryPlan::saliency_metadata_bytes` includes their 256-byte alignment and is already
part of both `artifact_storage_bytes` and `readback_bytes`; total owned memory charges both
copies. Ordinary raster/center/explicit modes add no statistics or GPU work. The selected
forward/tiled workgroup variant also controls the integer reduction.

`FramePacketSet::with_order` also accepts a complete physical sequence of packet identities
for generic frame assembly. `packets()` remains canonical, while `packets_in_file_order()`
follows the requested sequence. Every packet must appear exactly once; malformed orders are
typed errors. `assemble_frame` writes TOC sizes and payloads in physical order with the inverse
canonical-to-physical permutation.

Separate DC progressive frames remain unimplemented. Broader saliency/perceptual evaluation
and adaptive transform/quantization selection remain required. Encoding
still completes one whole frame per submission; progressive syntax does not imply an early
encoder byte-stream API. Native decoders can display byte prefixes ending at a complete AC
pass; the GPU frontend's whole-input and bounded-window progression are tested separately.

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

`TiledVarDctEncoder` accepts nonzero Gray/RGB dimensions through the checked 16,384-pixel per-axis
bound. Partial edge blocks replicate the final source row/column on GPU. A single AC group with one pass uses
the standard fused packet, including tiny and odd images; larger images carry every
`ceil(width / 256) * ceil(height / 256)` AC group and
`ceil(width / 2048) * ceil(height / 2048)` LF group. Each block is an independent DCT8 transform.
The first pass dispatches a two-dimensional block grid, with 64 lanes by default. Each workgroup
uses 2,048 bytes for 64 normalized color vectors and 64 quantized AC vectors, plus a four-byte quantization error flag. Coefficients stay in shared
memory and are immediately packed into one word-aligned block fragment per AC pass; adjacent workgroups never
write the same storage word. The second pass predicts and packs DC, resetting Gradient at LF-group
boundaries and writing a checked descriptor per LF group. Ending the first compute pass is the
global visibility boundary before the control pass publishes the completed artifact.

The host validates status, every layout field, live counts, DC residuals/histogram, AC counts and
coefficient ranges, exact fragment consumption, and zero padding. It appends the GPU-owned block
bits in pass-major, AC-group raster order (Y, X, B inside each block), with byte alignment only at packet ends.
There is no host transform, quantization, source padding, coefficient re-encoding, or pixel-codec
fallback. The independently concatenable block format relies on the single-distribution prefix
policy; future contextual or ANS encoders must maintain their state on GPU.

`VarDctMemoryPlan::kernel_layout` distinguishes `SingleTransform`, `StrategyMap` and `TiledDct8`. All use
892-byte parameters and a runtime-sized artifact with a 272-byte header. The former carries
the pass count, per-pass word stride and eleven spectral/shift descriptors; the latter records
the pass count. The source layout replaces three old geometry/width words with three 24-byte
component records, increasing parameters by 60 bytes; the artifact header is unchanged. LF descriptors
follow the header; the subsequent strategy, sample and entropy sections align to 256 bytes. Single-transform plans additionally report exact normalized-color, raw coefficient,
LF, quantized coefficient, matrix/order, transform-task and forward scratch allocations in `transform`.
Mapped plans report their aggregate allocation sizes: basis/uniform storage is shared per strategy,
while all transforms share image-wide color/coefficient/LF/quantized arenas. `xyb_bytes` retains
its public name and accounts for either normalized color domain. Each strategy batch
owns one horizontal scratch allocation and a 20-byte forward task per transform; encoder tasks
occupy 44 bytes per transform. No GPU allocation is created per individual transform.
Forward transforms use channel origins within one complete normalized-color binding, so small maps
also work on devices requiring 1024-byte storage offsets without padding each channel allocation.
Raw input instead uses four aligned plane bindings (unused entries alias the first); the reported
`source_binding_bytes` is their union, with overlap counted once and gaps between windows excluded.
Each general-transform matrix/order entry contains three F32 scales and three U32 indices in
24 bytes. An 8×8 DCT submission owns 12,304 bytes: 892 parameters, 3,072 artifact, 3,072 readback
and 5,268 resident transform bytes. Tiled DCT8 retains the same 24-byte entry layout in a
1,536-byte matrix/order table at read-only storage binding 3, reported by
`quantization_metadata_bytes` and included in the
job's owned bytes. It requires seven storage bindings and retains the table through completion
or cancellation. No coefficient readback is added.
XYB ICC adds `VarDctMemoryPlan::icc`: padded input and linear output planes (one plane each for
Gray, three for RGB), the lowered program, an original 892-byte source-layout record and the
resident 320-byte dispatch record. All belong to the existing job reservation, command submission
and map completion. Saliency still reads original device samples. Original ICC adds no color
intermediate. `icc_profile_bytes` includes the caller-owned retained profile in addressed bytes.
The still frontends' `memory_plan` includes `icc_storage_bytes`, twice the complete image-header
size for serialization/assembly overlap. The frame backend reports zero header storage; a sequence
reserves one header at creation and retains it through assembly or session drop, independently of
its submitted jobs. Success, rejected output and cancellation release their respective permits.

Mode-7 selections add `raw_matrix_input_bytes` for immutable sample/descriptor/prefix storage
and `raw_matrix_artifact_bytes` for compressed fragments and status. `readback_bytes` includes
those fragments at the end of the existing mapped buffer. All three allocations belong to the
same job reservation, submission and completion callback, including abandoned jobs. Families
without raw matrices incur none of this storage; image parameters and artifact layouts stay
unchanged. The [memory contract](../../docs/WGSL_MEMORY.md#raw-vardct-matrix-encoding)
defines the bounded sizes and word ownership.
Single transforms reserve one AC slot for three counts and at most `area - area / 64` coefficients
per channel; the largest 256×256 slot has 217,730 words. Mixed maps reserve the exact
strategy-specific bound per transform, with one length word each and no maximum-size slot
for smaller transforms. Tiled artifacts add one length word and a
214-word AC slot per block, with each section aligned to 256 bytes. The slot bound comes from
the actual prefix lengths for three counts and at most 63 signed coefficients per channel.
Every AC pass reserves its own length array and complete slot arena; shared quantized coefficients,
LF, raw-matrix fragments and matrix/order metadata are retained once. `grid().passes` and
`toc_entries()` include the configured pass count, including non-fused tiny progressive frames.
All pass fragments validate before assembly, using the same one submission and aggregate map.
The complete parameter + artifact + readback + resident transform reservation remains live through validation or
abandoned-job cleanup; caller-owned source bytes are reported separately. Source binding,
artifact and transform bindings, buffer size, workgroup storage, invocation count, and per-axis dispatch limits
are checked before submission. A full 16K square therefore also depends on adapter and budget
capacity.

Actual-GPU tests compare emitted streams with Rust `jxl`, installed `djxl`, and the stock GPU
decoder. All 27 strategies run textured RGB8 inputs with default/custom correlation, natural/custom
orders and parametric/raw matrices: each AC coefficient is checked against independent f64 transforms,
pinned native bases/orders and independent matrix expansion. All 162 streams agree across the
three decoders within one RGB8 code. Modes 1/2 use pinned libjxl matrix records; modes 3–6 use
the independent `jxl-vardct` parser. The shared
forward primitive separately checks 667 native coefficient/LF cases, including complete impulse
bases for all ten strategies with an 8×8 footprint. See the
[native fixture generator](../jxl_wgpu/test-data/forward_vardct_generator/README.md).
Procedural checkerboards, stripes, impulses, gradients, and colour patterns also exercise
single-packet images, AC/LF boundaries, custom correlation, and a 2057×2057 four-LF-group image.
Mixed-map cases cover all 27 strategies in one 512×512 image, a 2057×17 LF-boundary image,
and 13×21 non-DCT8 edge replication, including independent coefficient checks and all three decoders.
The fifteen maps include parametric and raw matrices combined with custom orders and LF metadata.
Tiled custom matrices and interleaved rectangular raw/parametric families retain whole/fragmented-input
agreement through 40-byte GPU windows. Independent entropy reconstruction checks every raw sample
in all 17 families, including `i32::MAX`, and malformed GPU fragments are rejected before assembly.
Two additional wide-sample images use independent `jxl-oxide` and native `djxl` with the same
one-code bound because Rust `jxl` 0.6.0 disagrees on those cases; the
[corpus](../../docs/CONFORMANCE_CORPUS.md#procedural-vardct-encoder-matrix) records the discrepancy.
The batched forward primitive separately checks disjoint/reordered source and output ranges for
all 27 strategies, with poisoned gaps and two transforms per batch under all five variants.
An independent f64 cosine-sum reference checks AC values within one integer quantizer step;
this is a numerical regression bound, not ISO precision or perceptual-quality certification.
Blocking/Future assembly and all supported linear workgroup variants produce identical bytes.
The [progressive encoder matrix](../../docs/CONFORMANCE_CORPUS.md#progressive-vardct-encoding)
adds 81 single-transform streams with exact coefficient accumulation, mixed maps, signed integer
endpoints, resolution stopping points, center/explicit/GPU saliency group order, native partial-input images
and whole/fragmented convergence. F32 references use the
pinned scalar libjxl oracle; normal native SIMD retains its independent RGB8 comparisons.
The suite also rejects malformed or missing GPU AC output in every pass, checks an insufficient device binding,
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
# use std::num::NonZeroU8;
# use jxl_wgpu_encode::{BufferImageSource, ProgressiveDownsampling, ProgressivePass, ProgressivePlan, TiledVarDctEncoder, VarDctConfig, VarDctGroupOrder, WgpuContext};
# fn encode_tiled(
#     context: WgpuContext,
#     source_768_by_513: BufferImageSource,
# ) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let progressive = ProgressivePlan::new(
    [2, 4, 8].into_iter().map(|size| ProgressivePass {
        coefficient_square: NonZeroU8::new(size).unwrap(),
        shift: 0,
    }).collect(),
)?.with_downsampling(vec![
    ProgressiveDownsampling { factor: 4, last_pass: 0 },
    ProgressiveDownsampling { factor: 2, last_pass: 1 },
])?;
let encoder = TiledVarDctEncoder::new_with_config(
    context, VarDctConfig {
        progressive,
        group_order: VarDctGroupOrder::saliency_first(),
        ..Default::default()
    },
)?;
let plan = encoder.memory_plan(&source_768_by_513)?;
let grid = encoder.grid(&source_768_by_513)?;
assert_eq!(plan.kernel_layout, jxl_wgpu_encode::VarDctKernelLayout::TiledDct8);
assert_eq!(grid.ac_group_count()?, 3 * 3);
assert_eq!(grid.passes, 3);
assert_eq!(grid.toc_entries()?, 1 + 2 + 3 * 9);
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

## Frame sequences

### Mixed Modular and VarDCT frames

`MixedModeEncoder::new(context, MixedModeConfig)` accepts explicit per-frame choices through
`MixedModeFrameEncoding::{Modular, VarDct}`. `begin_sequence(ImageSequenceDescriptor)` creates
one layered still or animation with a shared integer/floating Gray/GrayAlpha/RGB/RGBA image contract. Both codecs
use the precision in `config.vardct.sample_format` (1–31 integer bits or a legal floating precision, default RGB8) and the
`source_color` and `color_options` in that same VarDCT configuration. Each physical frame may independently select any supported
packed, planar or split layout, swizzle, bit position and word byte order; channels, precision and color stay fixed. The default config uses original-component
VarDCT, tiled DCT8 and default lossless Modular. An explicit XYB configuration is rejected.

`MixedModeConfig` fixes each codec's policy. `vardct_transform` selects `TiledDct8`,
`Single(strategy)` or `Map(checked_map)`. Fixed transform/map extents constrain VarDCT sources
only; Modular and tiled sources can vary per frame. Modular retains its configured Prefix/ANS,
predictor and transform policies. Regular VarDCT frames retain configured AC progression;
reference-only VarDCT frames use one implicit complete pass. The common contract supports
all five blend modes, signed crops, hidden layers, timecodes and four post-transform reference
slots across codec boundaries. Optional alpha uses `config.vardct.alpha` at the color precision,
with an independent extra-channel blend/reference field. Embedded RGB/Gray ICC uses exact profile
identity and one shared budgeted header. Pre-transform references and automatic mode selection
remain unsupported.
Modular preserves its physical source words, including nonfinite float payloads; selected VarDCT frames require finite color samples; alpha remains lossless.
Modular/VarDCT mode selection preserves these distinct contracts; including VarDCT does not make the presentation lossless.

The session's `submit_frame` and `submit_last_frame` take the source, encoding choice and
`FrameOptions`. `memory_plan(source, encoding, options, is_last)` uses the same checks without
advancing state or reserving memory and returns a `MixedModeMemoryPlan`. Both modes share the
context's byte budget; streamed Modular jobs reserve their existing live-batch peak. Failed
admission leaves the index and finality available for retry. Wait/Future completion, cancellation,
out-of-order insertion, and raw/plain/indexed-container assembly reuse the existing ownership
and validation boundaries. See the compiled usage example on
[`MixedModeEncoder`](src/mixed_encoder.rs) and the
[mixed-codec corpus](../../docs/CONFORMANCE_CORPUS.md#mixed-codec-sequence-encoding).

### Fixed-codec sequences

`begin_sequence` on either codec accepts a `LosslessModularSequenceDescriptor` or
`VarDctSequenceDescriptor` (an alias of `ImageSequenceDescriptor`) and returns its corresponding
`SequenceSession`. Descriptors select
`AnimationHeader::Still` for a layered still or `AnimationHeader::Animation` for a timed sequence.
The former omits animation metadata and requires zero duration and no timecode on every frame.
Non-final regular layers and reference-only producers remain hidden; the final regular layer
produces exactly one still image. Crops, blend modes, references, finality and admission use the
same checked rules in both sequence kinds. A single cropped final layer is also supported.
Call `submit_frame` for producers and `submit_last_frame` for the final layer, then insert the
completed artifacts and finish normally. Indexed stills describe one presentation with zero
duration and a 1/1 clock. [Layered-still evidence](../../docs/CONFORMANCE_CORPUS.md#layered-still-encoding).

```rust,no_run
# use jxl_wgpu_encode::{AnimationHeader, BufferImageSource, FrameCrop, FrameOptions,
#     LosslessModularEncoder, LosslessModularFormat, LosslessModularSequenceDescriptor};
# fn layered_still(encoder: &LosslessModularEncoder, background: BufferImageSource,
#     patch: BufferImageSource) -> Result<Vec<u8>, jxl_wgpu_encode::EncodeError> {
let mut sequence = encoder.begin_sequence(LosslessModularSequenceDescriptor::new(
    640, 480, LosslessModularFormat::Rgba, 8, AnimationHeader::Still,
)?)?;
let base = sequence.submit_frame(background, FrameOptions::default())?; // hidden slot 0
let final_layer = sequence.submit_last_frame(patch, FrameOptions {
    crop: Some(FrameCrop::new(100, 100, 64, 64)?),
    ..FrameOptions::default()
})?;
sequence.insert(base.wait()?)?;
sequence.insert(final_layer.wait()?)?;
sequence.finish_raw()
# }
```

The former `LosslessModularAnimationDescriptor/Session` and `VarDctAnimationDescriptor/Session`
names remain type aliases. Their constructors now also accept `Still`; `begin_animation` retains
its requirement for an animation timebase and delegates to the same sequence implementation.
Existing `encode`/`submit` still APIs retain their single full-canvas output. The corrected VarDCT
working-buffer declaration changes its image header; mixed all-Modular frames retain identical
frame payloads and metadata apart from that image-wide working-buffer flag.

`VarDctEncoder::begin_sequence` and `TiledVarDctEncoder::begin_sequence` share the same descriptor
and session. The descriptor fixes canvas and optional timebase; the encoder binds the selected color/alpha channels, integer or floating precision and enumerated/ICC input color,
XYB or original-component coding, transform policy,
quantization, matrices/orders and AC passes. Single transforms and maps retain their source
extent on each frame; tiled DCT8 accepts separately checked crop extents through its 16K axis
bound. Both support all five blend modes with alpha, signed crops, hidden zero-duration regular frames
and four post-color-transform references. Alpha has an independent blend/reference field.
Pre-color-transform reference storage remains unsupported. Fixed-VarDCT sequences accept the independent extra-channel declarations described above; mixed sequences currently accept packed alpha only. Mixed Modular/VarDCT sessions use
`MixedModeEncoder`; frame names and previews remain unimplemented.

Both codecs accept `FrameOptions { kind: FrameKind::ReferenceOnly, .. }` in either sequence kind.
Such a frame stores its decoded source in any of the four `save_as_reference` slots without a
presentation or blend. It must be non-final, retain default timing/color blend and an empty
extra-channel blend list, even when the animation declares timecodes. Its optional crop describes
only dimensions: the origin must be `(0, 0)`. Post-color-transform storage must cover the canvas;
Modular also accepts bounded smaller references with `save_before_color_transform = true`.
Reference-only syntax implies a single complete pass. VarDCT lowers the configured progression
to that pass before planning GPU parameters, packets and resource bounds; regular frames retain
their configured progression. `memory_plan_for_request` reports this exact per-frame admission,
while `memory_plan` continues to report the configured regular-frame resources. See the
[reference-only evidence](../../docs/CONFORMANCE_CORPUS.md#reference-only-frame-encoding).

Both codecs lower frame options once into a checked, immutable `FrameHeaderPlan` before GPU
admission. Resident, native-streamed and browser-streamed Modular jobs and VarDCT jobs append
that plan's bounded control bits after their codec-specific prefix. The same plan supplies the
wire frame kind, pass-bundle presence, physical frame index and final flag. Invalid controls or budget admission failures
do not advance the session or close its final-frame slot. No new GPU allocation, binding,
submission or map is introduced by animation control. See the
[VarDCT animation evidence](../../docs/CONFORMANCE_CORPUS.md#vardct-animation-encoding).

`LosslessModularEncoder::begin_sequence` writes one stream-wide image header and keeps a reusable
GPU session open for multiple frames. The descriptor fixes the canvas, format, sample precision,
optional tick rate, loop count, and timecode presence. Use
`LosslessModularSequenceDescriptor::new` for integers, `new_float` for binary16/binary32, or
`from_pixel_format` for explicit floating precision and source color.
Each regular frame supplies an exact duration,
optional timecode, optional signed crop rectangle, color blend contract, one contract per extra
channel, and the two-bit source/destination reference slots. GrayAlpha and RGBA animation carry
alpha as a standard extra channel with the encoder's declared association. Alpha-weighted
`Blend` and `MultiplyAdd` name that extra channel instead of treating alpha as a color component.
Each channel's own blend mode determines whether its source-reference field is present;
full-canvas Replace omits that field even when another channel uses Add or Multiply.

Frame submissions own their GPU work and therefore do not borrow the session. Callers may keep
multiple frames in flight, complete each with blocking `wait` or await the same runtime-neutral
`Future`, and insert completed artifacts in any order. Final assembly restores normative frame
order and rejects duplicates, gaps, or an invalid final-frame flag. All live frame jobs share the
same byte-weighted `MemoryBudget` as still encoding; Modular also uses its existing buffer pool.

Both sequence sessions and the generic `CodestreamAssembler` expose
`finish_indexed_container(inventory_limits, index_limits)` for opt-in plain `jxli` plus `jxlc`
output. After ordering artifacts, this bounded metadata pass inventories the actual assembled
headers and uses the shared `FrameSequencePlan` to generate every independent presentation anchor.
Hidden frames belong to their presentation; dependent intervals sum original ticks exactly.
Logical offsets exclude container boxes. Inventory limits bound physical frames, headers, ICC and
TOCs; index limits bound displayed frames, entries and payload. Failures remain typed, and no
partially indexed container is returned. Existing raw/unindexed methods keep their behavior.
The metadata pass does not validate image entropy or decode pixels. See the
[index contract](../../docs/FRAME_SEEKING.md#encoder-index-emission).

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
animation.finish_indexed_container(Default::default(), Default::default())
# }
```

The conformance suite exercises full-frame Replace, cropped Add, reference-slot persistence, RGBA
alpha-weighted Blend, mixed blocking/Future completion, and out-of-order completion. Every
displayed frame is compared exactly with both published Rust `jxl` and reference `djxl`.
