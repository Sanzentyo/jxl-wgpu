# GPU-only JPEG XL encoder architecture

Status: executable lossless Modular profile, experimental VarDCT profile, plus production-facing
orchestration API. Concrete backends advertise only implemented profiles and stages; unsupported
formats or features are rejected through typed capability errors.

The cross-codec capability source of truth and complete encoder backlog are in
[`FULL_JPEG_XL_ROADMAP.md`](FULL_JPEG_XL_ROADMAP.md). This document records the encoder design and
implemented ABI details; it must not independently broaden the advertised profile.

## Non-negotiable boundary

The production encoder may use the host for bitstream/container orchestration, validation, job
ordering, and deterministic serialization. It may not use the host to normalize pixels, predict
samples, form residuals, transform or quantize coefficients, tokenize image data, build image-data
histograms, or silently replace a failed GPU job.

For the executable profile, `lossless_modular.wgsl` reads the caller's `wgpu::Buffer` and performs
reversible transforms, the selected predictor, packed-signed residual mapping, ZeroRuns or Greedy
LZ77, hybrid-uint tokenization, and histogram accumulation. Rust builds bounded entropy metadata
and validates artifacts. Prefix assembles validated events on the host; ANS serializes complete
groups on GPU before host fragment/TOC/container assembly. A
GPU mapping or shader failure is an encode failure.

The repository does not vendor or retain `libjxl` as an upstream source tree. The production crates
use ordinary Cargo dependencies; official source is consulted only for specification and
implementation audits.

## Frame control and animation

`FrameHeaderPlan` owns regular-frame crop geometry, per-channel blend-field presence, timing,
reference-field presence and finality. Construction validates source/crop agreement and wire
bounds, then compiles at most 256 control bits before GPU memory admission. Its fields are
private; frame jobs use the retained bits and index/finality accessors instead of reinterpreting
caller options after GPU completion. `LosslessModularBackend` uses this representation in
resident, native-streamed and browser-streamed execution, and `VarDctBackend` uses it after its
own quantization/progression checks. Codec-specific headers supply coding mode, group geometry
and pass layout; the common plan supplies the regular-frame suffix and disabled restoration.
The stream serializers also share one checked animation-timebase writer.

`VarDctAnimationDescriptor` compiles bounded RGB8/XYB image metadata. Both VarDCT frontends create
`VarDctAnimationSession` around the existing generic `EncodeSession` and `CodestreamAssembler`.
Frames retain the encoder's immutable transform/quantizer/pass policy, with independently checked
tiled crop extents. Failed submissions leave the frame index and finality available for retry;
completed artifacts may be inserted in any order. Reference slots describe codestream metadata,
not host-computed image state: the GPU encodes each supplied source and the decoder composes it.
This profile emits post-color-transform references and rejects pre-transform storage, alpha
blends and extra-channel contracts. Fixed frame metadata does not change GPU ABI, allocations,
submission/map counts or the existing completion/cancellation ownership. Syntax and conformance
scope remain in the [animation corpus](CONFORMANCE_CORPUS.md#vardct-animation-encoding).

Indexed container assembly has a separate authority boundary after frame ordering.
`CodestreamAssembler::finish_indexed_container` inventories the actual output headers under
caller limits, then constructs the bitstream crate's immutable `FrameSequencePlan`. This common
plan owns frame order/finality, timing, reference-slot versions, LF producers and transitive
dependencies; decoder execution and index binding use it too. `FrameIndex::from_sequence`
compiles independent presentation intervals into `jxli` without consulting codec configuration
or trusting public artifact labels as header evidence. This is bounded host metadata work;
GPU pixels, entropy, submission ownership and the unindexed assembly APIs are unchanged.

## Implemented profile

`LosslessModularBackend` advertises exactly:

| Property | Implemented value |
|---|---|
| Coding mode | Modular lossless |
| Color models | Gray (`NonColor`/`X000` or `Gray`/`X001`), GrayAlpha (`Gray`/`X00W`), RGB/RGBA with bijective component swizzles; alpha is one extra channel with declared association |
| Sample depths | every integer `1..=31` or all 154 legal floating precisions (2–8 exponent and 2–23 fraction bits), with equal precision across components |
| Input | one pitch-linear `wgpu::Buffer`, one through four packed/planar/split planes, 8/16/24/32-bit words, arbitrary field positions, Native/Little/Big byte order, `ChromaSubsampling::None` |
| Source color | full-range enumerated RGB/Gray, standard/custom primaries and white, Linear/sRGB/BT.709/PQ/HLG/DCI/Gamma, or unchanged embedded RGB/Gray ICC; four intents and positive binary16 image white |
| Extent | `1..2^30` per axis, further bounded by device limits |
| Frame/group layout | selectable 128/256/512/1024-square PassGroups (default 256), LF groups eight times wider/taller, multi-group, row-major TOC |
| Animation | `true` (5 blend modes, signed crop, 4 reference slots, timecodes) |
| Determinism | `CrossDevice` integer GPU artifacts and deterministic host assembly |
| Progressive passes | `max_progressive_passes = 1` |
| Implemented stages | `ColorTransform`, `ModularTransform`, `ModularPrediction`, `ModularResidualTokenization`, `HistogramReduction`, plus `AnsSerialization` when selected |
| Predictor | All 14 explicit standard predictors and caller-selected Weighted coefficients; default Gradient |
| Modular transforms | caller-selected none or any of 42 RCT types in DC-global or each pass group; default YCoCg for integer RGB(A), none for Gray/GrayAlpha or floating samples |
| Entropy | Default Prefix or GPU ANS; ZeroRuns or bounded Greedy LZ77; fixed four-leaf channel MA tree |
| Filters | Gaborish off, EPF zero iterations |
| Output | raw codestream or standard `jxlc` container; private `jwgp` index emitted only for single-group Gray8 Prefix containers with default color/intent/intensity |

The backend rejects textures, chroma subsampling, YUV/NV12, signed samples, unsupported color
metadata, non-bijective/missing channels, mismatched component precision and progressive passes > 1.
Storage normalization remains fused into GPU token production; no intermediate image is allocated.

Source color lowering is bounded host metadata work. It validates full range and RGB/gray semantics,
quantizes custom xy to `1e-6` and Gamma OETF exponents to `1e-7`, and rechecks the quantized geometry.
Image headers are complete before GPU submission; encoder options declare rendering intent and
exact positive binary16 unit-white luminance, defaulting to Relative/255 cd/m². The token kernel
continues to preserve source words. Animation descriptors bind logical channels, precision and
serialized color across physical layout changes. The crate README owns the API and rejected cases;
[source-color conformance](CONFORMANCE_CORPUS.md#lossless-modular-source-color) records its evidence.

Embedded ICC lowering accepts RGB/Gray profiles with either ordinary component swizzles or
explicit `IccDevice`/`Device` channel labels. The host codes only profile metadata: a bounded
header predictor, one literal-copy command and a uniform eight-bit prefix alphabet shared by all
41 ICC contexts preserve the entire original profile. It neither evaluates profile methods nor
reads source pixels. The source profile's intent must match the encoder options. Byte limits and
an exact header plan precede variable-sized allocation; the shared budget reserves the header
and one temporary assembly copy. Assembly moves that allocation and reserves its final extension
once. Stills release the metadata permit on completion/cancellation; animation sessions hold one
permit across frames and bind byte-identical profiles. GPU parameters, bindings, artifact sizes
and submission counts do not change. [ICC evidence](CONFORMANCE_CORPUS.md#lossless-modular-embedded-icc)
covers native profile/sample interoperability and resource lifetime.

GrayAlpha maps the gray and alpha swizzle outputs to logical channels 0/1 in the existing source
parameter array. Its image header declares grayscale and one alpha extra channel, while RGB(A)
keeps its existing channel order and transform rules. `AlphaAssociation` only controls the
image-wide extra-channel declaration; the token kernel preserves all supplied words, including
nonzero color at zero alpha. It does not change bindings, artifact ABI, GPU ownership or sample
precision. [Alpha-input conformance](CONFORMANCE_CORPUS.md#lossless-modular-alpha-input) separates
exact raw-sample preservation from arithmetic composition and final alpha presentation.

`LosslessModularConfig` supplies one immutable group size, MA-tree mode and color transform to the backend and
complete encoder. The grid, LF-group count, source windows, artifact capacities and frame-header
`group_size_shift` derive from that same size. Each animation crop computes its own grid before
admission. A complete group's channels must fit a GPU batch; larger groups retain typed source,
artifact, buffer and dispatch rejection. Prefix LZ77 tokens extend through 31 for full 1024² zero
runs, while exact per-group sample counts, canonical extra bits and histogram agreement still
validate every artifact. No shader ABI or workgroup shape changes. See
[group-size conformance](CONFORMANCE_CORPUS.md#lossless-modular-group-sizes).

The RCT policy resolves against source channels/precision before admission. One typed value feeds
both the GPU parameter and resident/native/browser packet assembly. A local RCT changes only the
pass-group transform header; shared and local MA trees remain independent choices. Single-group
frames fold that transform into their fused DC-global header. The shader loads the selected
channel permutation and applies the forward wrapping-i32 lifting steps during sample prediction.
It preserves raw IEEE words and leaves alpha untouched, with no intermediate allocation,
submission or readback. The parameter word at byte 24 is now the normative RCT type `0..=41`,
or internal sentinel `42` for no transform. Wire type 42 is rejected by the public constructor.
[RCT conformance](CONFORMANCE_CORPUS.md#lossless-modular-rct-selection) covers both placements,
both trees, every operation/permutation, source-word preservation and ownership.

### Entropy planning and ANS serialization

`LosslessModularEntropyCoding` is the caller's policy. The checked dispatch plan owns group
output capacities, hybrid histogram storage and aligned per-batch metadata before admission.
`EntropyCode` owns
one immutable frame codebook; `EncodedGroup` is granted only after GPU completion, event/histogram
validation and fragment checks. Native and browser schedulers share those boundaries.

ANS reuses the two-pass batch scheduler even for a single batch: GPU histograms first, then
retokenization and ANS in the second submission. Four channel contexts (0/1/2/3+) and distance
share one to five distributions. Their immutable codebook owns the context map consumed by
both wire metadata and GPU descriptor lowering; neither consumer reinterprets the choice.
The same codebook owns a hybrid-uint configuration for each shared distribution and one global
LZ77 length configuration. `CodingPlan` additionally owns the LZ77 start symbol and alias alphabet,
so wire headers and GPU table construction use the same symbol domain.
Distributions are normalized to 4096 using exact integer largest-remainder allocation with symbol-order ties.
Each observed symbol receives at least one slot. The host serializes small/general histogram
metadata; only the final selected histograms compile to alias reverse maps. It never codes ANS
image symbols. Shared `ans.rs` owns these table rules independently of Modular transforms and
leaves room for other encoder consumers.

The first submission profiles 40 hybrid configurations on GPU after canonical tokenization.
These are all split/MSB/LSB combinations representing every u32 with at most 235 raw symbols,
leaving at least 21 length symbols within the maximum 256-symbol alphabet. Canonical events retain
split/MSB/LSB `0/0/0`; one shared WGSL helper recodes residuals and distances for profiling and final emission.
Canonical LZ77 events retain `4/0/0`. A batch-wide atomic histogram arena has fixed capacity,
independent of group count. Host code validates complete canonical events before accepting
profile completion and coarsens each profile back to the canonical histograms for comparison.
It aggregates only this bounded metadata, without host residual recoding.

Length coding evaluates all five full-20-bit configurations fitting the reserved 32 symbols:
split 0–4 with no MSB/LSB retention. A bounded histogram conversion maps the canonical direct
bins 0–15 and exponent bins 16–31 to each candidate, with exact extra-bit totals and checked sums.
No GPU re-profiling or host event recoding is required. `LengthCoding` owns the setting used by
wire headers and the GPU; unsupported length counts cannot enter selection.

The global search examines 94 threshold/length plans. Threshold candidates are each raw
configuration's full-u32 symbol count and the wire-special 224. Raising a threshold between
these boundaries only inserts empty bins without admitting another configuration; 224 is
retained because its wire field is shorter. Each plan chooses the smallest sufficient alias
alphabet (64, 128 or 256), which cannot cost more than a larger one under this objective.
A 32-symbol alphabet cannot cover the minimum 33 raw plus 21 length symbols of this policy.
Generic ANS table construction independently supports all four normative widths.

For each plan and each of the 31 nonempty context unions, clustering selects the best supported
residual/distance configuration and examines all 52 partitions of the five contexts. Candidate
`AnsHistogram` values contain normalized counts, not reverse alias tables; only the global
winner compiles its one to five `AnsCode` GPU tables. The global cost includes every repeated
LZ77 header. Equal global costs prefer smaller alphabets, thresholds, then length settings.
Its objective combines Q20 normalized cross-entropy, exact residual/distance/length extra-bit counts
derived from canonical histograms, and the actual histogram, hybrid-configuration and simple-context-map bit lengths. Header cost is charged once for a shared
tree or single group, and once plus every PassGroup for multi-group local trees. ZeroRuns contributes
one distance symbol per run. Checked histogram sums reject overflow; fixed-point binary logarithms
and u128 cost accumulation avoid host-libm decisions. Equal costs prefer fewer clusters, then the
lexicographically smaller map; hybrid ties use split, MSB and LSB order. This searches the existing
contexts, without learning a new MA tree.
State-dependent renormalization and final packet alignment are not simulated;
the estimated minimum is not a guarantee that every resulting codestream is smaller.

Before histograms are available, the dispatch plan still admits the five-table maximum. The second
pass uploads and binds only selected tables, within that reservation, and retains the same batch
lease. Profiling adds a compute pass within the first submission, without an extra submit/map.
Its storage participates in batch splitting; selection never requires a late allocation.

One serial invocation per group visits all channels/events backwards, expands ZeroRuns into
literal/length/distance symbols, recodes canonical lengths with the selected configuration, and
prepends renormalization words and hybrid extra bits. It then
prepends the 32-bit state and rebases the resulting fragment in place. Groups have disjoint output
ranges; per-channel streams cannot be independently concatenated. Empty multi-group DC-global
retains the zero-symbol ANS state emitted by libjxl's `WriteTokens`. The same codebook feeds global
or independently local headers. A completed fragment must match its admitted capacity, expanded
symbol count and zero tail padding before packet assembly. A failed stage never selects Prefix.

The existing exclusive parameter/artifact/readback lease includes tables, group/channel descriptors
and worst-case compressed storage through mapping, consumption and cancellation. The public memory
plan reports `ans_output_bytes` and `hybrid_histogram_bytes` as artifact subtotals and two submissions
per batch. GPU binding and bit-address limits are checked before admission. See the [ABI](WGSL_MEMORY.md#modular-ans-serialization)
and [independent evidence](CONFORMANCE_CORPUS.md#lossless-modular-gpu-ans-encoding).
The default Prefix path and private Gray8 acceleration index keep their previous byte contract;
ANS containers omit that Prefix-specific index. This stage establishes correctness, with no
measured throughput or compression-ratio claim. Learned contexts, clustering outside this bounded
Modular ANS codebook, input-domain-specific configurations, effort policy and parallelism within
a group remain future work.

### GPU artifact ABI

The token kernel is still `@compute @workgroup_size(1)` in `lossless_modular.wgsl`.
Parallelism comes only from the number of (PassGroup, channel) pairs in the dispatch; one
invocation scans its whole group serially. Multi-batch jobs and every ANS job use two submissions per batch
(one histogram pass, one serialization pass). This remains a correctness milestone, not the
eventual performance topology. Its readback buffer consists of little-endian `u32` words:

```text
word 0       event_count
word 1..33   raw hybrid-token counts (33 entries)
word 34..66  LZ77 hybrid-token counts (33 entries)
word 67..99  distance hybrid-token counts (33 entries)
word 100..   event_count records of:
              kind, token, extra_bit_count, extra_bits
```

`kind == 0` is a raw residual token, `kind == 1` is a zero-run token, and Greedy uses
`kind == 2/3` for a length/distance pair. No source sample or residual
plane is copied into a private container box. The ABI is bounded before allocation: at most
`pixels + ceil(pixels / 8) + 1` events per group channel.

Source precision belongs to `PixelFormat`, separately from transform or entropy policy.
`FloatPrecision` checks the sign/exponent/fraction geometry once; `SampleKind::CustomFloat`
requires every channel packing field to match its total width. Legacy IEEE storage still uses
`SampleKind::Float`. The checked source specification resolves either form to the same sample
and exponent widths used by the dispatch/memory plan, capability negotiation and image/alpha
headers. Animation descriptors infer that specification from their source format and require
both widths to match on every frame. Header serialization uses the checked precision domain,
including the 24-bit and general-width buckets, without deriving exponent width from word size.
The GPU only loads and transforms raw words, so custom precision adds no shader ABI, intermediate
allocation or submission. A custom descriptor does not grant generic F32 display/output support.

The parameter ABI is one `#[repr(C)]`, `bytemuck::Pod` Rust value and the matching WGSL structure.
`ModularSourceParams` / `Source` is 24 bytes and `ModularParams` / `Params` is 256 bytes, both
with four-byte alignment. The [WGSL memory table](WGSL_MEMORY.md#uniform-and-structured-storage-table)
owns the complete field layout, including the resolved sample source and Squeeze band.

Compile-time and shader validation tests check the ABI. The 256-byte array stride keeps each
batch parameter range aligned. Source bindings 0/3/4/5 address individual planes; bindings 1/2
retain artifacts and parameters. Unused sources alias the first plane, and configurations below
six storage bindings are rejected before pipeline creation. Each component carries its own
word offset, byte stride, field shift and source-plane selector. Word loads preserve raw floating bits.

Admission revalidates every public image-layout field and its final addressable word. Each group
uses checked `offset + (height - 1) * row_stride + (width - 1) * pixel_stride + word_bytes`, rounds
its enclosing plane window to the device/storage-word alignment, and supplies relative u32
addresses to WGSL. Batches split before either any source-plane binding or the artifact binding
exceeds the device limit. A single oversized group remains a typed resource rejection. Source
accounting takes the union of bound ranges, omitting plane gaps and duplicate alignment prefixes;
there is no additional source copy or normalized image.

Artifact and MAP_READ allocations use checked word/byte arithmetic, are four-byte copy aligned, and
must fit both `max_storage_buffer_binding_size` and `max_buffer_size`. The public
`LosslessModularEncoder::memory_plan` reports valid bits/exponent width, largest component storage-word width, channel count,
format, group grid, full and peak unions of source binding ranges, parameter storage, peak artifact storage,
diagnostic total artifact bytes, readback bytes, batch count, exact GPU submission count, streaming
mode, owned bytes per job, and addressed bytes per job. `for_in_flight(n)` reports checked aggregate
bytes for a caller-selected concurrency ceiling, while `memory_limits` exposes the relevant device
limits.

The parameter, artifact, and mapped-readback allocations form one exclusive reusable buffer set.
Sets match the exact artifact size, remain leased through map completion and consumption, and are
returned safely even when a Future is abandoned. Idle retention defaults to 32 MiB with a
256-set object cap; `buffer_pool_stats`, `set_buffer_pool_limit`, and `clear_buffer_pool` expose
reuse and control. Caller-owned source bindings are neither copied into nor retained by this pool.

The predictor compares signed sample values without overflowing the comparison differences:

```text
low = min(left, top)
high = max(left, top)
gradient = bitcast_i32(bitcast_u32(left) + bitcast_u32(top) - bitcast_u32(top_left))
prediction = top_left < low ? high : (top_left > high ? low : gradient)
```

The first row predicts from the left; the first sample of later rows predicts from the first sample
of the previous row. Residual subtraction and packed-signed mapping use modulo-32-bit arithmetic,
including the full signed range of high-depth chroma differences. Eight-sample chunks turn a run
longer than seven zeros into the configured LZ77 form.

Raw tokens use hybrid configuration `000`: token zero represents zero; for token `t > 0`, read
`t - 1` extra bits and add `2^(t - 1)`. LZ77 uses configuration `400`: values below 16 are direct;
otherwise token `t` reads `t - 12` bits and adds `2^(t - 12)`. The decoded run length is that value
plus eight and the configured distance is one.

The raw alphabet includes tokens 0–32; token 32 carries 31 extra bits. High-depth trees use at
most eight bits at the first prefix level, leaving at least seven for the nested LZ77 tree and
keeping combined lengths within 15 bits. The Prefix LZ77 alphabet still starts at symbol 224. Existing
1–16-bit prefix policies and the private Gray8 index retain their original 19-entry alphabet;
the checked 609-byte Gray8 fixture is unchanged.

## Modular transform planning direction

`ModularTransformPlan` resolves the supported RCT/Palette/Squeeze policy before dispatch
lowering. A frame shares at most four concrete group shapes through one immutable plan.
Each shape records ordered wire operations, explicit channel sources/extents/axes/bands and
Palette capacities. Global RCT placement and single-group fusion are resolved there as well.
`ModularDispatchPlan` consumes it for GPU parameters, groups, batches and memory bounds;
resident, native-streamed and browser-streamed assembly retain that same plan. Buffer
reservations still belong to execution and survive through completion/consumption.

The implementation and further transform extensions use these ownership boundaries:

| Layer | Owns |
|---|---|
| Selection policy | Requested operations, target ranges, predictor/search choices and caller limits. |
| Resolved transform plan | Validated operation order, channel roles and mappings before/after each operation, dimensions, meta/image partition, and checked resource capacities. |
| Validated GPU result | Actual entry/delta counts and entropy artifacts, checked against the plan before they can determine header fields or published output. |

Represent source-component, intermediate-channel, Palette-local component and final
encoded-channel indices explicitly; a raw channel count cannot describe their mapping.
For example, applying Palette to components 1 and 2 of a four-component image leaves
the meta table followed by source component 0, the index image and source component 3.
Squeeze selection addresses that image-channel list while excluding the meta prefix. Its immutable
policy keeps axes, target range and residual placement together. The plan validates the range before
admission and follows the selected channels and their descendants through each axis. In-place
residuals are inserted after the selected range; tail residuals may be separated from averages by
unselected channels, so a second axis can require two distinct wire steps. Unselected channels keep
their extents and carry no Squeeze axes into the kernel. Both the
resource bounds and the wire operations must follow that same resolved topology.

`local_transforms` selects either a Squeeze policy or an ordered RCT/Squeeze program. The former
retains optimized separable lowering or up to 296 parameters in one wire transform; the latter
emits one transform per operation, within the 273-entry complete-header bound. Both use current
image-channel ranges after the source-RCT/Palette prelude, excluding Palette metadata.
The same shape plan resolves ordered wire parameters, channel shifts and 64-byte GPU jobs, rejecting
empty Squeeze inputs, invalid intermediate ranges and cumulative-shift overflow before admission.
Explicit steps preserve zero-sized residual slots; named separable policies retain their one-pixel-axis
elision and byte identity. Sequence policy storage is immutable and shared on clone.

RCT requires three channels with equal extents and shifts; it may include alpha or Squeeze
residuals and retains valid empty triples in the wire topology without a GPU job. The complete
header bound includes local prelude operations and single-group fusion.

Each job reads post-RCT components, Palette indices or earlier arena views. The planner allocates
disjoint average/residual outputs or all three RCT outputs before retiring any input span, then
coalesces free spans for later jobs. Final channel descriptors carry arena offsets. One invocation owns each group's
ordered program and tokenization, so no cross-invocation synchronization is needed. The existing
parameter allocation also carries the planned metadata table; a shared resident/streamed uploader
copies it into private artifact storage after clearing and before dispatch. Metadata and the peak
live arena are charged before execution, with no additional binding or buffer-pool ownership path.

Memory admission, dispatch parameters and transform-header structure derive from the
resolved plan. WGSL receives a working-component/index/table source and an explicit
Squeeze axis mode/band or arena offset instead of reconstructing that mapping from an encoded
channel number.
GPU-dependent
dimensions remain bounded by the pre-execution capacities and become authoritative
only after artifact validation. Host planning remains metadata work; pixel transforms,
dictionary search and residual generation stay on GPU. Replacing flags with enums alone
does not establish these boundaries.

`PaletteCapacity` validates GPU counts into `ValidatedPaletteCounts`; header emission
checks those counts against its own planned capacity. Caller policy fields are not
used as execution results. Existing supported combinations retain public behavior,
default bytes, typed rejection and resource lifetime. Extend the plan and its lowering
for a new combination. Keep independently parsed
wire headers, exact/native/GPU comparisons and invalid-input/ownership cases: tests that
reuse the planner's own expected topology cannot replace those independent checks.
Palette interleaving and global/LF/HF transform topology remain roadmap work until
their acceptance gates are met.

## Public API and state model

`WgpuContext` owns shared device/queue handles. `GpuEncodeBackend` is the capability and submission
boundary, and `GpuEncodeJob` is its executor-independent completion object. `FrameSubmission`
implements `std::future::Future` and also offers `wait`; neither API names Tokio, async-std, smol, or
another runtime.

`EncodeSession` assigns monotonically increasing frame indices and permits several returned jobs to
remain in flight. It tracks open/final state separately from completion order. `CodestreamAssembler`
accepts independently completed frame artifacts, orders them by frame index, enforces exactly one
final frame, and produces raw or deterministic container output.

The concrete convenience path is:

```text
LosslessModularEncoder::submit / submit_container
    -> LosslessModularSubmission: Future<Output = Result<Vec<u8>, EncodeError>>

LosslessModularEncoder::encode / encode_container
    -> the same submission through its blocking wait path
```

On native `wgpu`, each context owns a bounded `SubmissionPoller`, and every context clone shares its
single completion worker. `WgpuContext::from_backend` reuses the backend's worker and byte budget,
and inherits its adapter-validated `KernelPolicy`, so encode, decode, and readback use one workgroup
selection contract and do not create per-submission polling threads. The VarDCT keys
`vardct_encode_forward` and `vardct_encode_quantize` accept every linear `KernelVariant`. Single
transforms use resident coefficient/LF buffers and zero workgroup storage; tiled DCT8 uses exactly
2 KiB. Complete parameter, basis, matrix/order, scratch, artifact and readback allocations are
included in the pre-submission byte plan, and per-axis dispatch/device limits are checked. VarDCT control
serialization and the lossless Modular token pass remain fixed scalar kernels because changing only
their workgroup sizes would race sequential predictor and bit-offset state. Poll capacity is
reserved before queue submission and saturation is a typed retryable error. A browser cannot block
`Device::poll`; its synchronous wait returns an error and callers must await.

`EncoderCapabilities::negotiate` is authoritative. A backend must only list profiles and stages it
executes. `LosslessModularBackend` reports `animation = true` and `max_progressive_passes = 1`.
Multi-frame animation sessions are orchestrated via `LosslessModularAnimationSession`.

## Deterministic packet assembly

`FramePacketSet` accepts GPU groups in arbitrary completion order and canonicalizes them to the JPEG
XL TOC order:

```text
DC global, DC groups, AC global, (pass-major, AC-group-minor)
```

The one-group/one-pass optimization collapses this to one fused packet. TOC sizes use the four
normative buckets `(10, 14, 22, 30 bits)` with offsets `(0, 1024, 17408, 4211712)`. All bit writing,
raw/container validation, `jxlc` construction, and auxiliary-box framing come from
`jxl_gpu_bitstream`; the encoder does not duplicate container assembly.

LF global carries the shared Modular tree and selected entropy code (four distributions derived from combined
channel histograms); LF groups and HF global are empty; each PassGroup carries its own group header
and channel token streams inside standard row-major TOC groups. Group payload and TOC are byte
aligned.

### `jwgp` acceleration index

The standard `jxlc` remains the source of truth and must decode without private metadata.
For single-group Gray8 Prefix containers, `encode_container` adds an optional private `jwgp` box containing
only a bounded, hash-bound index into those codestream bits so the project's GPU decoder need not
first implement a fully generic JPEG XL entropy parser. Multi-group, RGB(A), and other bit depths
omit this box and remain standard interoperable containers. Unknown-box-aware decoders, including
`djxl`, ignore it. It never stores pixels or residuals.

The acceleration-index payload is fixed-width and little-endian. Bit offsets are measured from bit
zero of the raw codestream's first byte and bits within a byte are LSB-first:

```text
"JWGP"                         [4]
version = 1                    u16
fixed_header_size = 84         u16
profile = 1                    u16  # gray8/lossless/modular/single-group/prefix
flags                          u16  # bit 0 means LSB-first
codestream_length              u64
SHA-256(codestream)            [32]
width, height                  u32, u32
token_bit_offset               u64
token_bit_length               u64  # excludes group zero padding
sample_count                   u32
predictor, channels, bps, zero u8, u8, u8, u8
raw prefix (nbits, bits)        19 * (u8, u16)
LZ prefix (nbits, bits)         33 * (u8, u16)
```

The final payload is 240 bytes. Parsers must validate the version, fixed sizes, reserved bits,
profile invariants, multiplication bounds, codestream length/hash, prefix-code validity, token range,
and exact `sample_count` termination before dispatch.

## Why not use the internal `jxl` crates as an encoder?

The Rust `jxl` workspace is valuable for decoding and for shared format semantics, but it does not
provide an encoder pipeline. Its useful encoder-adjacent pieces are not a stable public API that can
turn GPU-produced groups into a codestream. Depending on decoder internals would couple this crate to
private data structures without removing the need for GPU token kernels or encoder-side entropy and
TOC decisions.

This project therefore uses focused ordinary dependencies (`jxl_gpu_formats`,
`jxl_gpu_bitstream`, and `wgpu`) and treats decoder crates as conformance oracles in tests. Encoder
execution remains independent of decoder implementation internals.

## Official `libjxl` audit

The read-only reference clone is pinned to commit
[`aea3a06e281fdee13e04815bfbf4f4132e7f59ea`](https://github.com/libjxl/libjxl/commit/aea3a06e281fdee13e04815bfbf4f4132e7f59ea)
(2026-08-21). The relevant primary-code findings are:

- [`enc_frame.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_frame.cc): `ComputeEncodingData`, `ComputeVarDCTEncodingData`, `TokenizeAllCoefficients`, global DC/AC emission, parallel group encoding, streaming and one-shot assembly.
- [`enc_group.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_group.cc): pixel transforms, AC strategy, coefficient quantization, and progressive coefficient splitting are group-local after global choices are fixed.
- [`enc_modular.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_modular.cc) and [`enc_encoding.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/modular/encoding/enc_encoding.cc): Modular tree, transforms, predictor/token production, global info, and per-stream encoding.
- [`enc_ans.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_ans.cc) and [`enc_entropy_coder.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_entropy_coder.cc): histogram clustering and entropy serialization introduce a global barrier between parallel token production and final group emission.
- [`enc_progressive_split.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_progressive_split.cc): progressive passes split spectral coefficients and/or quantized shifts; they are not independent re-encodes.
- [`enc_toc.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/enc_toc.cc) and [`toc.h`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/toc.h): canonical section order and TOC size distributions used by this crate.
- [`encode.cc`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/encode.cc) and [`encode_internal.h`](https://github.com/libjxl/libjxl/blob/aea3a06e281fdee13e04815bfbf4f4132e7f59ea/lib/jxl/encode_internal.h): raw codestream, `jxlc`, `jxlp`, and streaming container boundaries.

The resulting full encoder topology is:

```text
GPU input normalization / color transform
  -> Modular transforms + predictor, or VarDCT transform + quantization
  -> per-group token streams + local histograms
  -> global histogram reduction / clustering barrier
  -> entropy-ready canonical group packets
  -> deterministic host TOC, frame, codestream, and container assembly
```

AC groups can run independently only after global metadata, quant fields, progressive layout, and
entropy clustering policy have been selected. A future implementation should batch many small
images or animation frames in one submission and reuse the existing pipeline and bounded artifact
pool; future multi-stage kernels will also need persistent scratch planning. One image stage per
dispatch would reproduce CPU structure rather than exploit the GPU.

## Other encoder crates reviewed

Versions were checked on 2026-08-30.

| Crate | Finding |
|---|---|
| [`jpegxl-rs 0.15.0`](https://crates.io/crates/jpegxl-rs) / [`jpegxl-sys 0.13.0`](https://crates.io/crates/jpegxl-sys) | `libjxl` CPU FFI; useful as an oracle, prohibited as the production image path. |
| [`jpegxl-src 0.12.0`](https://crates.io/crates/jpegxl-src) | Bundled C++ source, not a GPU encoder architecture. |
| [`jxl-encoder 0.3.1`](https://crates.io/crates/jxl-encoder) / [`jxl-encoder-simd 0.3.0`](https://crates.io/crates/jxl-encoder-simd) | Pure Rust encoder implementation, but AGPL/commercial licensing is unsuitable for code reuse here. |
| [`zune-jpegxl 0.5.2`](https://crates.io/crates/zune-jpegxl) | Permissive MIT/Apache-2.0/Zlib simple Modular encoder. Its fast-lossless prefix/header logic is the implementation reference for the first profile; it is not a production dependency and never receives production pixels. |
| [`jixel 0.2.20`](https://crates.io/crates/jixel) | Permissive pure Rust encoder, but internal packet construction is not a stable public GPU boundary. |
| [`gamut-jxl 0.4.0`](https://crates.io/crates/gamut-jxl) | Native encoder bindings, therefore a CPU path. |
| [`jxl-oxide 0.12.6`](https://crates.io/crates/jxl-oxide) | Decoder only. |

The adapted prefix construction is attributed in source and originates from
[`zune-jpegxl`'s encoder](https://github.com/etemesi254/zune-image/tree/0.5.2/crates/zune-jpegxl),
under its stated MIT, Apache-2.0, or Zlib terms.

## Validation already running

`gpu_tokens_form_a_reference_decodable_lossless_codestream` uploads a 17x13 grayscale image at a
device-aligned non-zero binding base plus a four-byte relative plane offset and a padded row stride.
It encodes through the WGSL kernel and asserts every decoded byte against the original through both
the pure Rust `jxl 0.6.0` decoder and, when installed, official `djxl` (`libjxl 0.12.0`). The test
therefore covers GPU addressing, border prediction, zero-run and non-zero residual tokens, prefix
serialization, frame/TOC assembly, and reference decode equality. It also validates the per-job and
four-job memory accounting, validates the `jwgp` payload against the exact `jxlc`, decodes the
container through both reference decoders, and verifies that two independent GPU submissions
produce identical container bytes. A separate exhaustive test mirrors the WGSL event admission
logic for every zero/non-zero residual stream up to 16 samples, covers several maximum-dimension
patterns, and proves the last possible four-word event write remains inside the allocation.

The deterministic integration fixture is `fixtures/gpu_gray8_lossless.jxl` (609 bytes, SHA-256
`414eb08c62c34d2dd17d0b9f51c3fa1f3c5d750c50fd48d79b76e31f40092ef0`). The test compares newly
encoded bytes directly with this fixture. Set `JXL_WGPU_WRITE_FIXTURE` to an explicit path when an
intentional bitstream-format change requires regeneration.

Run:

```console
cargo test -p jxl_wgpu_encode gpu_tokens_form_a_reference_decodable_lossless_codestream -- --nocapture
cargo clippy -p jxl_wgpu_encode --all-targets -- -D warnings
```

## Implementation slices

### Completed slices

- **Multi-group Modular (Slice 3)**: All four standard PassGroup sizes, multi-group row-major TOC layout,
  two-pass streaming with global histogram aggregation, and out-of-order group completion.
- **Lossless color and alpha inputs (Slice 4 partial)**: Gray/GrayAlpha/RGB/RGBA at integer depths
  `1..=31` or every legal floating precision, with packed/planar/split addressing and declared alpha
  association. RGB(A) can select every GPU-side RCT type or no transform, in global or local headers.
- **Lossless Modular animation (Slice 6)**: Multi-frame `LosslessModularAnimationSession` supporting
  standard timebases, exact durations and timecodes, signed crop rectangles, all 5 blend modes,
  alpha blending, and 4 reference slots with runtime-neutral in-flight futures.
- **VarDCT forward transforms and AC (Slice 5 partial)**: `VarDctEncoder` executes all 27
  standard strategies, and `TiledVarDctEncoder` supports multi-LF/multi-AC-group DCT8 grids with
  checked axes through 16K. Both frontends serialize validated exact-binary16 LF dequantization plus
  LF/HF chroma-correlation metadata. All 27 strategies perform the forward transform, normative LF
  extraction, default-matrix quantization, caller-selected coefficient-order scans and AC bit-fragment serialization
  without exposing pixels or raw/quantized AC to the host. Single transforms use shared resident
  passes; tiled DCT8 keeps each block in 2 KiB workgroup storage. The single `TransformKind`
  alphabet and shared matrix/order metadata are used by both encoder and decoder. Its
  `HfEntropyPlan` currently selects one prefix cluster for all 495 coefficient contexts,
  disables LZ77, and emits 1–11 configured spectral/quantized AC passes. That plan is a stable policy boundary rather than a temporary
  wire format: future adaptive clustering, ANS/LZ, content-adaptive coefficient-order selection, and pass selection can use
  different plans while retaining the GPU artifact contract. Native coefficient/LF fixtures and
  independent f64 compressed-coefficient checks cover every strategy with default/custom correlation.

VarDCT delivery separates coefficient order from physical AC-group order. Validated caller
orders and center geometry remain host metadata. Optional `saliency_first` computes local
RGB edge contrast on the GPU with an integer workgroup reduction, then reads only four words
per group through the existing artifact/map. Host sorting compares those bounded statistics
exactly; it never reads source pixels. The same permutation repeats in each AC pass after
LF/HF metadata. [The encoder contract](../crates/jxl_wgpu_encode/README.md#experimental-vardct-profile)
and [memory layout](WGSL_MEMORY.md#progressive-vardct-encoding) define the heuristic, limits
and cancellation ownership. Local contrast is distinct from content-adaptive transform or
quantizer search and does not establish production perceptual quality.

### Remaining work

The authoritative encoder items, dependencies, priorities, and acceptance gates are the `MOD-E`,
`VDCT-E`, `ENT-E`, `ENC`, and encoder-facing `IO` rows in
[`FULL_JPEG_XL_ROADMAP.md`](FULL_JPEG_XL_ROADMAP.md). After the structural-refactoring gate, the
nearest work remains parallel Modular token production, native YUV/NV12-family ingestion,
broader entropy/progression and the rate/quality control built on top.
Batched codec submission and advanced performance instrumentation stay separate from
format-completeness claims.

Until an item is implemented and validated, capability negotiation must reject it. Benchmarks,
wrappers, and CPU oracles do not expand the advertised production capability.
