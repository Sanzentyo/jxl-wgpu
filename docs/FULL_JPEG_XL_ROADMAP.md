# Full JPEG XL implementation roadmap

Status date: 2026-09-21. This document is the canonical capability and implementation backlog for
the workspace. A crate README may explain a component in more detail, but it must not claim a
broader codec profile than this file.

## What “full” means

JPEG XL is defined by ISO/IEC 18181-1 (codestream), 18181-2 (container), 18181-3 (decoder
conformance), and 18181-4 (reference software). This project uses the following completion gates:

- **Full decoder** means that every feature of the current JPEG XL codestream and container can be
  decoded, composed, color-managed, and returned without a CPU image-codec fallback. A conforming
  input may be rejected only for a checked resource/device limit, malformed data, a future unknown
  extension, or a deliberately unsupported non-portable output-memory layout. Legal current
  codestream features may not remain `UnsupportedFeature` branches.
- **Full encoder syntax coverage** means that the GPU encoder can produce interoperable Modular and
  VarDCT streams, stills and animations, supported metadata/container forms, extra channels, and
  lossless JPEG recompression. An encoder does not have to emit every redundant combination allowed
  by the grammar.
- **Production encoder quality** is a separate gate. It requires useful distance/quality and effort
  controls, rate control, progressive choices, and competitive quality/size/speed evidence. Merely
  emitting a decodable fixed-quality stream does not satisfy it.
- “GPU-only” applies to image-domain codec work: prediction, transforms, coefficient/residual
  processing, filtering, color conversion, and supported entropy jobs. Bounded host parsing,
  scheduling, validation, deterministic bit writing, container assembly, and explicit GPU readback
  remain allowed. Production must never select a CPU pixel codec.

CUDA-specific block-linear surfaces are outside this project because portable `wgpu` cannot expose
them. That product boundary does not reduce JPEG XL codestream conformance: all representable
pitch-linear images must still decode correctly.

## Status and documentation discipline

| State | Meaning |
|---|---|
| **Done** | Production path exists, advertises the capability, and has positive, negative, and interoperability or conformance evidence. |
| **Partial** | Useful implementation exists, but at least one legal variant or required validation gate is missing. |
| **Missing** | No authoritative production path exists. Parsers, models, shaders, or tests alone do not make it supported. |
| **Out of scope** | Deliberate platform boundary, not a codec feature silently omitted from “full” claims. |

Every capability-changing commit must update, in that same commit:

1. the item and status in this roadmap;
2. the root capability summary;
3. the affected crate README;
4. `WGSL_MEMORY.md` when a shader ABI, alignment, workgroup allocation, binding, or memory lifetime
   changes;
5. `CONFORMANCE_CORPUS.md` when coverage changes; and
6. `GPU_BENCHMARKS.md` only when new measurements were actually collected.

A roadmap row moves to **Done** only with the named acceptance evidence. A new parser branch or WGSL
kernel normally moves a row from **Missing** to **Partial**, not directly to **Done**. Checked-in
benchmarks must name the adapter, path, dimensions, concurrency, output target, and validation
contract. Aspirational performance and unexecuted test cases are never reported as measurements.

## Current capability baseline

| Area | State | Authoritative current boundary |
|---|---|---|
| Raw/`jxlc`/`jxlp` transport and header inventory | **Partial** | Bounded transport, ICC reconstruction, feature metadata and explicit GPU preview/main selection exist. Complete previews can open before main input arrives, with independent source and output ownership. Non-accumulating scanners emit arbitrary-chunk raw/`jxlc`, ordered-v0, and bounded out-of-order-v1 `jxlp`, preserve auxiliary events, incrementally parse image/frame headers and TOCs, and route exact physical section ranges. Direct GPU entropy-window integration and full container policy remain incomplete. |
| Modular decode | **Partial** | Lossy XYB and original-sRGB color now join Gaborish/EPF, resampling (also shared by XYB LF producers), spot/alpha handling and composition through the shared GPU color boundary, with a 19-stream corpus. The unfiltered path supports one final Gray/RGB still with independently declared 1–31-bit integer or legal floating color/extra precision, standard/custom 2×/4×/8× color and up to 64× extra resampling, raw or normalized scalar plane selection, all eight orientations, one through eleven passes, standard YCoCg, Prefix/ANS, LZ77, and bounded MA prediction. Every accepted stock MA profile, including Weighted/SelfCorrecting prediction, can resume within one entropy stream through bounded overlapping GPU uploads. Multi-group DC-global Palette/Squeeze reconstructs nonempty global samples into a frame arena, schedules channels with both shifts at least three through LF groups before the header-assigned nonempty pass streams, and executes one frame-wide inverse/finalizer. Modular YCbCr now uses independent integer/floating component grids and GPU JPEG expansion before restoration/resampling, with 474 native streams covering all sampling triples, global/LF/pass ownership, independent extras and progressive color/numeric output. The transform corpus covers all RCT types, global/LF/pass Palette/Squeeze stacks and native geometry for 236 global cases and 664 local substreams. Empty residual RCT retains topology without GPU work. A further 218 patch/noise and mixed-reference cases cover these components through all eight patch modes, restoration, independent extras and frame resampling. Broader mixed MA/transform, spline/LF-producer features and post-transform composition conformance remain. |
| VarDCT decode | **Partial** | The authoritative color-output engine accepts 1–31-bit integer or legal floating XYB/original-sRGB metadata, or 8-bit integer YCbCr input, normalizes all eight orientations for RGB/gray presentation and covers mixed XYB maps containing all 27 regular and special transform strategies, nonzero AC across spectral/refinement passes, opt-in immutable DC/AC images for stills, animations and composed presentations including extra channels and deferred descriptors, plus LF dependency images with independent alpha/extra output, all 13 natural/custom coefficient-order families, stream-defined block contexts, non-default LF dequantization and LF/HF chroma correlation, every normative default and parametric custom strategy matrix, all 3-bit X/B scales, scanline or entropy-permuted center-first pass groups, multiple LF groups with shared or per-substream local MA trees, recursive progressive-DC dependencies, resident Gaborish plus one-to-three-iteration EPF, standard/custom 2×/4×/8× frame resampling, and a checked 2056×256 LF-boundary extent. Sectioned raw mode-7 matrix side images reuse bounded GPU uploads with deferred inverse/overlay completion. Both global- and local-MA raw images, including independent local LF/HF packet trees without a global tree, have actual-adapter evidence through 40-byte windows and fragmented input. Checked non-XYB 4:4:4/4:2:2/4:4:0/4:2:0 JPEG-transcode streams use component-sized LF/AC grids, resident quarter/three-quarter JPEG upsampling, and encoded YCbCr conversion through public `GpuDecoder`. Shifted components also have budgeted expansion before restoration, frame resampling, nonzero noise or retained component references; twenty JPEG restoration/noise streams now validate those expanded planes with independent scalar and CPU references. Another 128 streams cover all 64 component sampling selector triples at aligned and odd extents, including nonzero/zero noise and active adaptive LF for equal factors. Eight additional nonzero-LF-correlation streams fix the equal-nonzero-selector case: LF correlation now uses normalized component shifts. Unequal-factor adaptive LF is rejected as malformed by the shared header parser before TOC/section/GPU work; all four equal triples remain valid. Larger/transformed raw-matrix and broader asymmetric restoration/resampling conformance remain. |
| Lossless Modular encode | **Partial** | Gray/RGB/RGBA with 1–31-bit integers or IEEE binary16/binary32, packed/planar/split buffers, component swizzles and explicit word bit/byte order, 256×256 groups, one pass, fixed Gradient and prefix+RLE/LZ77. Integer RGB uses YCoCg; floating channels retain raw words without a transform. Enumerated RGB/Gray color, all intents and explicit image white are retained, including Replace animations. Exact original-word and native/GPU output evidence covers high depths, IEEE special values, full Replace and finite crop/Add/Multiply animations. |
| VarDCT encode | **Partial** | All 27 strategies perform real forward transforms, normative LF extraction, default-matrix AC quantization, natural-order tokenization and prefix packing on GPU. Caller-selected mixed maps validate exact coverage and AC-group boundaries, batch transforms by strategy, replicate non-DCT8 edges and assemble multiple LF/AC groups. Tiled DCT8 retains its workgroup path and checked 16K axis bound. Independent f64/native coefficient and Rust/libjxl/GPU image tests cover mixed maps and group boundaries under five byte-identical variants. Content-adaptive selection, custom orders/matrices, general distance/quality guarantees, rate control and progressive encoding remain absent. |
| Restoration/render graph | **Partial** | Reusable upsampling, Gaborish, EPF, blend, color, and display kernels exist in `jxl_wgpu`; the bounded stock VarDCT decoder expands only shifted JPEG components before restoration, constructs one full-image sigma plane, and routes Gaborish plus signaled EPF0/EPF1/EPF2 across LF-group boundaries through one resident ping-pong scratch set in the same submission. VarDCT frame 2×/4×/8× upsampling executes after restoration with standard/custom weights and exact output extents. Frame crop/blend composition now executes in original-encoding F32; Modular color and both-mode extra resampling now reuse the same filter after resident normalization; the remaining legal graph is incomplete. |
| Output formats | **Partial** | Native integer Gray/RGB/RGBA and 30 portable VPI pitch-linear outputs exist for the lossless Gray8 conversion path; VarDCT fuses XYB, original-sRGB or YCbCr reconstruction with the shared color-output shader: all 20 color VPI layouts, planar YUV, NV21/NV42, P010/P012/P016, D65 primary conversion, and explicit SDR transfer/range/siting. Orientations 1–8 precede chroma subsampling; grayscale reconstructs luminance. Both coding modes support planar/interleaved F32 RGB/BGR/RGBA/BGRA, with actual-depth Modular normalization, Gray+alpha and independent alpha depth. Modular also exposes selected extras as native integer or normalized scalar F32 planes, with declarations retained in metadata. Explicit Apply/Keep orientation controls also cover mixed animation and recursive DC. Selected gray/RGB components now support native unsigned and scalar F32 output through the common frame surface; Modular direct selection retains exact source codes. Enumerated PQ/HLG now carries image intensity through both codecs and composition. HDR and ICC now connect through explicit image white. Legacy Gray8 numeric layouts in VarDCT/composition remain unsupported. |
| Async/concurrency/memory | **Partial** | Native blocking and runtime-neutral futures, browser compilation, one shared byte budget, leased output lifetime, true aggregate readback, bounded pools, and deterministic budget-adaptive Modular/VarDCT entropy windows exist; codec submission is not yet coalesced across images. |
| Decoder animation and composition | **Partial** | The common executor runs independent Replace and composed Modular, VarDCT, mixed JPEG/Modular and recursive-DC sequences. It retains four accounted post-transform F32 reference slots, handles negative/oversized/off-canvas crops, all five frame blend modes, separate straight or associated integer alpha sources (including Gray+alpha and independent depths), hidden layers and reference-only frames. Initial retry, dependency backpressure, cancellation and both-oracle evidence exist. All supported integer extra channels now retain independent blend/reference/alpha selection in one planar allocation, including shifted/resampled and distributed streams. GPU patch dictionaries and explicitly tagged pre-transform references now execute for both coding modes, with all eight patch modes and independent extras. Patch pass updates and separate LF previews reuse validated component-domain references without committing intermediate images. Both LF producer modes apply patches before publishing their prediction version, with separate extra-plane ownership. Patched producers now defer frame upsampling and noise until ordered patch completion, before prediction/reference publication. A 140-image native corpus covers both modes, LF producers/consumers, independent extras, custom weights and reference overwrites. A further 494-image corpus covers subsampled VarDCT YCbCr patches, all 64 sampling selector triples, mixed Modular/VarDCT and RGB/YCbCr references, filtered/noisy references and four-slot overwrites, including 218 Modular YCbCr and mixed-reference additions. GPU spline entropy, bounded geometry and ordered tile rendering now compose with patches, LF producers/consumers, upsampling, noise and progressive output. The official 60-frame animation spline reference and 80 generated feature/progression streams provide precision and lifetime evidence. Independent plane layouts support unequal color/extra resampling around splines. Modular YCbCr spline/LF-producer features, broader post-transform combinations and non-SDR composition remain. |
| JPEG bitstream reconstruction | **Partial** | Bounded immutable `jbrd` metadata parsing and canonical emission have original-JPEG native interoperability evidence. Actual JXL GPU coefficients now drive qualified sequential/progressive entropy and validated original-byte assembly. All 36 pinned originals, including three official JPEGs, match exactly; broader legal profiles remain open. |

The 24-case checked-in round-trip corpus proves the narrow stock Modular profile across diverse
dimensions up to 15360×8640. It is not evidence of full JPEG XL coverage.

## Required implementation items

Priority is `P0` for a blocker on the decoder’s core image path, `P1` for required complete-format
coverage, and `P2` for production encoder quality, broad product integration, or performance after
correctness. Dependencies name other item IDs in this document.

### A. Unified frontend, transport, and container

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `FRONT-01` | P0 | **Partial** | `GpuDecoder::wgpu` now inventories once and automatically selects Modular or VarDCT while sharing one backend byte budget; one actual-adapter test decodes both modes sequentially through the same decoder and checks pixels and reservation release. The common frontend first constructs `SelectedImageInventory` for Main or Preview; its reconstruction inventory then enters `FrameExecutionPlan`, which separates physical IDs from local node positions and snapshots LF/reference versions. Its lazy production sequence executor covers independent Replace Modular/VarDCT and mixed JPEG-VarDCT/Modular frames, including recursive DC, with dual-oracle and lifetime evidence. Pre-transform references now retain explicitly tagged component surfaces, with patches and deferred frame upsampling/noise before publication. Completion still requires the remaining entropy/side-image and render combinations. Both producers now export one all-channel planar surface to composition, with every integer extra retained and independently blended. Mixed-mode post-transform crop/blend composition now has actual-GPU dual-oracle evidence. | — |
| `FRONT-02` | P0 | **Done** | Inventory preserves each physical section index/range and its logical TOC index after bounded entropy-coded permutation decode. Frontend section vectors normalize to logical group order while retaining those physical ranges. Scanline fixtures, the imported 49-section permutation fixture, a deterministic six-group center-first VarDCT fixture, structural range checks, and an actual-GPU Rust-`jxl`/`djxl` oracle cover the acceptance contract. | — |
| `FRONT-03` | P1 | **Done** | Transport accepts arbitrary owned chunks without collecting a whole stream: apart from the inline signature, ordered raw/`jxlc`/v0-`jxlp` payloads share caller `Arc` storage and gap-blocked v1 fragments use a separately limited retained buffer. `CodestreamStreamScanner` reconstructs only bounded image/frame-header/TOC probes, emits `Arc` frame inventories before section data, and routes physical section ranges in logical codestream order. Public `GpuDecoder::stream` consumes borrowed transport events and passes the resulting inventory plus checked logical span table through the same `GpuSubmissionEngine` boundary as contiguous `open`; no complete host codestream is assembled. A cloneable incremental-input budget atomically admits bytes and one span per nonempty event before scanner mutation. Immutable per-range tokens follow the source into Modular or VarDCT and release after the last owner or source-dependent submission. A complete preview can be taken before main input; preview and main share intersecting tokens once and cancel independently. Later main ranges never become preview-owned. Host input and GPU allocations use distinct budgets because both are simultaneously live during upload. All Modular/VarDCT metadata readers and LF/combined/HF/AC upload paths cross arbitrary spans; VarDCT's temporary whole-stream GPU buffer is initialized directly from them. Every split, byte-drip fragmented animation, retryable admission, cancellation, actual-GPU Modular/VarDCT selector execution, and staged local-HF lifetime are covered. | `CONT-01` |
| `CONT-01` | P1 | **Partial** | The contiguous parser and incremental `ContainerStreamScanner` validate naked streams, `jxlc`, delivery-order v0 and indexed/out-of-order v1 `jxlp`, compact/extended/to-end box sizes, fragment order/completeness, typed input/box/codestream/buffer limits, and end-of-input. Auxiliary start/chunk/end events preserve exact 8/16-byte headers and shared payload ranges; only gap-blocked future fragments are copied and their live/peak bytes are observable. The incremental codestream inventory observes transport events without consuming auxiliary metadata and preserves absolute frame/TOC/section ranges. Explicit `MetadataSelection` and a bounded `MetadataCollector` now retain selected auxiliary payloads independently of decoder input. Completion still requires the full current-version ordering/compatibility rules. | — |
| `CONT-02` | P1 | **Done** | Public opaque Exif/XMP/JUMBF and unknown-box storage preserves original plain or `brob` payloads, duplicate order and selected ownership. Contiguous parsing and a borrowed-event collector share explicit selection, encoded/retained/count limits and atomic replacement/removal. Explicit decoding adds per-box/aggregate output, expansion-ratio and RFC 7932 window limits; writing connects to ordinary `jxlc` and fragmented `jxlp` assembly. Every split, malformed/limit/lifetime cases, 180 Google Brotli bidirectional parameter pairs and six raw libjxl box extractions provide interoperability evidence. Seventeen decoder selections preserve 444 immutable presentations and 1,357,296 F32 words through metadata changes, including orientation, ICC/HDR, animation and previews. Opaque contents are caller-owned documents, not TIFF/XML/JUMBF semantic validation. [Policy and evidence](CONTAINER_METADATA.md). | `CONT-01` |
| `CONT-03` | P1 | **Missing** | Read, validate, generate, and use animation frame indexes (`jxli`) for bounded seeking. Random access must restore the reference-frame dependency chain before presenting a target frame. | `CONT-01`, `FRAME-04` |
| `CONT-04` | P1 | **Partial** | Unknown image, frame, and restoration extension selectors now return typed scope/selector errors before an authoritative inventory, including zero-length payloads. Frame/restoration payload lengths retain checked extension-bit limits. Safe auxiliary container events are independent. Selected unknown auxiliary payloads now preserve their original encoded bytes through metadata collection/editing. Completion still requires all current-version compatibility rules and semantic container extensions. | `CONT-01` |
| `CONT-05` | P1 | **Partial** | Immutable bounded `jbrd` parsing/emission retains 144 independent libjxl original-byte comparisons over 36 inputs / 140 scans. The GPU coefficient API restores exact quantizers and LF/AC/DC/CfL into checked component grids, matching independent libjpeg-turbo extraction. `open_jpeg_reconstruction` now connects those actual GPU coefficients to sequential/progressive entropy, restart/reset/ZRL and shared-padding handling, bounded ICC/Exif/XMP framing and GPU byte assembly. Every byte matches all 36 unchanged originals, including three official sources; five 40-byte-window async cases agree. Raw/escaped size statuses, quantizer consistency and coefficient-bit coverage gate admission and final byte authority. Limit, malformed-state, cancellation, late memory/poller failure and retained-output checks cover the shared-budget lifetime. Wider coefficients/ratios, broader sampling/frame/global-prefix variants and refinement extra-ZRL preservation remain required before full completion. JPEG ingestion is `ENC-05`. [Metadata](CONTAINER_METADATA.md#jpeg-reconstruction-metadata); [byte-output contract](../crates/jxl_wgpu_decode/README.md#gpu-original-jpeg-byte-output). | `VDCT-D03`, `VDCT-E02` |
| `CONT-06` | P1 | **Partial** | Implement bounded encoder output for ordinary progressive order, seek-back TOC assembly, and out-of-order `jxlp` streaming. Generated fragments must reassemble to the same logical codestream, and partial writes must never be reported as a finished container. | `CONT-01`, encoder packet topology |
| `CONT-07` | P1 | **Partial** | Version-zero `jhgm` and exact ISO 21496-1 fractions have bounded parsing/emission, color/ICC metadata reconstruction and native bidirectional interoperability. GPU still reconstruction supports both headroom directions, HDR baselines, requested display headroom and explicit gain reference white in enumerated application primaries with enumerated or ICC output. The shared GPU ICC presentation supports RGB/Gray and complete device components with explicit intent, alpha and orientation; 13 profiles have independent F64 and Little CMS comparisons. The 64 gain-map sources and 48 HDR still sources cover both codecs, original/XYB, Gray/RGB maps, asymmetric resampling, wide primaries, alpha, orientations and transfers. Baseline endpoints preserve ordinary output bits; signed-weight and HDR checks retain existing codec/transfer precision contracts. [Contract and evidence](GAIN_MAP.md). Completion still requires ICC application spaces and broader ICC input/output combinations, animated/progressive/streaming delivery, full numeric/profile conformance, alternate tone-mapping policy, and broader native/official coverage. | `CONT-01`, `COLOR-02` |

### B. Global codestream and frame metadata

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `META-01` | P1 | **Partial** | Execute every image-header dimension, orientation, intrinsic size, preview, animation, original-profile, bit-depth/exponent, tone-mapping, opsin, upsampling, and extension field that currently is only inventoried. VarDCT integer source depths 1–16, orientation 1–8, and grayscale RGB8 presentation have actual-GPU dual-oracle evidence, including resampling and recursive DC. Sixteen depth fixtures plus four high-depth composition cases compare both whole and bounded async output within one code; unsupported integer errors preserve depth and color domain. All legal integer and floating declarations are now admitted; the precision checkpoint below covers native and F32 delivery through 31 integer bits. Explicit GPU preview/main selection now executes independent dimensions, orientation and preview timing through both producers; 48 reproducible streams cover every preview aspect encoding and mixed main LF/animation dependencies. Boundary tests cover defaults and non-default encodings. Bounded original-profile export now borrows embedded bytes or generates enumerated RGB/Gray/XYB ICC metadata, with exact official and native comparisons; image/gain-map parsing shares the implicit-XYB grammar. [Profile contract](ICC_COLOR.md#original-profile-export). | `FRONT-01` |
| `META-02` | P1 | **Partial** | Execute complete frame headers: regular/LF/reference-only frames, names, signed crops, duration/timecode, save-as-reference, is-last, group-size shift, encoding mode, resampling, passes, restoration, and per-channel blend/upsampling state. | `FRONT-01` |
| `META-03` | P1 | **Partial** | Contiguous and incremental inventory preserve LF levels, resolve every `USE_LF_FRAME` read to the exact earlier producer in JPEG XL's four progressive-DC slots, reset dependency state across the preview/main boundary, and reject a missing producer with a typed error before submission. A libjxl `--progressive_dc=2` chain is checked under contiguous and one-byte event delivery. The common physical scheduler executes Modular or VarDCT roots and each single-entry intermediate VarDCT HF-metadata/HF-global/AC continuation with resident XYB handoff and no pixel readback. Actual-GPU `--progressive_dc=1` and `=2` fixtures default to final-only output and match Rust `jxl` within one RGB8 code through blocking and runtime-neutral async completion. Embedded previews now decode as independent stills through both modes, including non-final preview headers, their own dimensions and physical noise counters. Leading main LF frames continue the preview's nonvisible noise count, while LF/reference image state remains separate. A complete preview now opens before main input via `take_preview`, with independent ownership and no whole-file-completion claim. Direct color-only VarDCT stills now publish validated LF/DC/AC refinements through `next_update`, with immutable output leases, original physical IDs, intended detail and unchanged presentation timing. Whole/256-byte windows match native per-pass images and final-only GPU bytes. Composed/animated DC/AC updates now include extras with independent per-pass reconstruction and validation. Modular color/native/scalar F32 and VarDCT numeric-extra presentations now share composed pass refinements with final-only reference commits. Completion still requires broader LF conformance and incomplete-frame-input refinements. LF presentation now retains independent normalized extras separately from prediction slots; complete LF1/LF2 roots and delayed background versions have native/scalar oracle and lifetime evidence. | `FRONT-03`, `FRAME-03` |
| `META-04` | P1 | **Partial** | Enforce checked limits for dimensions, frame count, extra channels, names, groups, passes, tree nodes, histograms, boxes, recursion, and allocations before submission. Fuzzing must show no panic/OOB/unbounded allocation. | all frontend work |

### Embedded preview selection checkpoint

`GpuOutputRequest::with_image_selection` selects Main (default) or Preview before the shared
engine boundary. `SelectedImageInventory` retains the original source inventory and lowers only
presentation metadata: preview extent becomes the canvas, animation timing and intrinsic extent are removed,
and exactly one final still is exposed. Main keeps its physical frame IDs and LF/reference
versions. Entropy ranges and noise seeds are not renumbered. The preview advances the nonvisible
noise counter; leading LF/hidden main frames continue it until the first visible main frame.

The image-header parser now owns the outer metadata grammar, with one bounded walk using
primitive sample/color bundles. Preview width is read only for aspect selector zero, and both
axes are limited to 4096. Whole and incremental parsing switch to main after one preview,
independently of its encoded `is_last`, and reject nonzero preview reference slots.
Forty-eight reproducible streams cover Modular/VarDCT previews, all 16 dimension encodings per
mode, original RGB, JPEG sampling, alpha, F32, resampling, non-final headers, mixed animation and
recursive LF main images. Native libjxl's dedicated preview API verifies pixels. Independent
main controls, native and Rust references verify LF noise and composition; whole/256-byte-window
output is bit-identical. All orientations, numeric alpha, one-byte inventory, malformed selection,
syntax/entropy failure, admission retry and cancellation are covered without a CPU pixel fallback.

`META-01/03`, `FRONT-01`, `RENDER-05` and full decoder conformance remain **Partial**.
Whole `open` and `finish` require authoritative end-of-input. `GpuDecodeStream::take_preview`
now opens the complete embedded preview before any main frame bytes arrive. Its typed
`ImageSourceInventory::PreviewPrefix` preserves original metadata/offsets without claiming whole
codestream or container validity. The frontend continues to main completion with independent
requests and leases. Immutable per-range tokens share byte/span admission once; preview excludes
later ranges and a crossing final range retains its complete charge. Both bounds reject before
scanner mutation and support retry.

All 48 existing fixtures compare early preview and subsequent main GPU output exactly with the
complete-input paths. Raw/jxlc/jxlp two-chunk boundaries, auxiliary relay, delayed/crossing takes,
engine retry, missing/truncated inputs, poisoned continuation and independent cancellation before,
during and after submission are checked. GPU tests include later-main/preview entropy failures,
full-budget retry and separate scalar-extra/orientation requests. Intermediate progressive
presentation, broader render/color combinations and encoder preview syntax remain open.

### C. Common entropy and packet execution

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `ENT-D01` | P0 | **Partial** | The bounded Modular, VarDCT packet, and DCT8 pass-group paths share a typed 12-byte Rust/WGSL `EntropyStreamParams` prefix for token bounds and the LZ ring mask while preserving consumer-specific records and bindings. Whole-range consumers also share exact ANS-state and zero-padding termination. Sectioned shared-global-tree VarDCT packets resume their complete known range from `section_bits.x` without an intermediate map, while their 64/128-byte GPU state retains the LF/HF phase and phase-transition scalars. For an absent global MA tree or a single-entry TOC, the frame engine instead host-packs each LF-local descriptor, executes LF entropy on GPU, maps only aggregate cursor/status records, parses each following HF-local descriptor, and resumes through a separate GPU entry point without host image entropy. Oversized local LF/HF streams execute as ordered bounded windows over the same shared upload; only final LF windows expose aggregate end cursors and one final map validates HF plus downstream work. A single-entry progressive intermediate uses a distinct GPU HF-metadata stop, validates status 31, parses bounded scalar HF-global metadata at its returned cursor, and submits general AC plus downstream resident reconstruction before the next dependency. Conservative fixed AC storage is admitted up front; exact late entropy/order/window buffers use the same budget. Runtime-neutral pending state owns every physical submission/map and publishes one logical final frame. Actual-GPU `--progressive_dc=2` blocking and async output matches Rust `jxl` within one RGB8 code. The low-level AC executor also returns validated unaligned Prefix/ANS cursors, with per-pass termination propagated to every upload window and GPU handoff to a following Modular image. Raw matrix side images now share the source-span window executor with global/distributed extras, retaining matrix and frame ownership through every callback and deferring inverse/overlay until entropy completes. Completion requires lowering the remaining side-image/frame consumers into the same common execution graph and broader recursive corruption/truncation coverage. | `FRONT-02` |
| `ENT-D02` | P0 | **Partial** | Every accepted stock Modular MA profile, combined/global-tree and staged local-tree VarDCT LF/HF packet, and the VarDCT AC pass-group consumer split oversized entropy into ordered GPU work over one reusable upload. Adjacent segments carry 16-byte backward/forward overlap and yield only between complete output tokens. Modular uses 32/48/112-byte aligned resume records. Multi-group Modular also schedules its DC-global zero-symbol Prefix/ANS range through the same bounded executor, exact final-state/padding check, byte budget, and final aggregate status map instead of treating the range as host-validated padding. VarDCT packets define 64-byte generic and 128-byte SelfCorrecting `Pod` states inside each reconstruction allocation; five explicit words retain packet phase, decoded LF/HF counts, first-block count, and extra precision. Local-tree groups conservatively reserve 128 bytes because their HF tree is discovered only after LF completion, then reuse that state sequentially for HF. VarDCT AC uses a 464-byte `Pod` record holding common ANS/LZ state, nested block/channel/coefficient progress, sticky sink failure, and the three-channel 96-word nonzero-neighbour grid. An explicit bounded-mode bit keeps middle packet windows distinct even when neither FIRST nor FINAL is set. Channel boundaries reset predictor-local state; only a final segment performs exact ANS/padding or packet-tail termination. Final combined/global or local-HF windows share the first downstream submission. The caller cap is bounded by device limits and both coding modes adapt it against a per-frame budget. VarDCT planning is deterministic against total budget capacity, searches four-byte-aligned caps down to the 40-byte overlap/sentinel minimum, exposes the resolved cap, and returns typed `MemoryBudgetTooSmall` before submission when no minimum layout fits; current live reservations instead cause retryable submit backpressure. Host scheduling tests cover unaligned starts, overlap mapping, lane isolation, budget adaptation, undersized caps, and segmentation of every oversized Modular profile. Actual-adapter fixtures force 193×97 Prefix+RLE/LZ77 Gradient, libjxl 193×197 ANS Weighted, a generated 32×32 staged single-entry packet through a 40-byte cap, libjxl 2056×256 shared-global and local-tree packets, libjxl 438×589 global packet plus nonzero/custom-order AC, and a recursive single-entry intermediate with late HF-global/AC through fixed and budget-resolved caps. Blocking and runtime-neutral async outputs match their source/Rust-`jxl` oracles, concurrent sessions report typed budget pressure, abandonment releases reservations, and late-window damage returns typed group-specific GPU entropy failure from the final aggregate map for Modular and combined/local packet paths. Static memory stats expose the resolved cap, packet state, reusable packet/AC peaks, and known initial packet batches without double-counting; dynamically discovered local HF/global batches and exact submission counts are sampled after completion. Global extra-channel side images now reuse the same window geometry and 48/112-byte Modular state, stop at their validated sample count/ANS state before the packet's upper bound, and defer inverses until completion. Six every-plane/cursor fixtures and seven public color/alpha fixtures match whole input under 40-byte or 1024-byte caps; zero-bit Prefix input and constant-space geometry near the u32 bit limit are checked. Raw matrix side images now share lazy window geometry, overlap/sentinel uploads, early termination and deferred inverse/overlay. Global/local-MA fixtures, including independently local LF/HF packets, prove exact matrices and cursors under 40/44/64-byte caps, unchanged resources before completion and on one-bit truncation, public fragmented decode against both oracles, budget-driven shrinking and cancellation at three stages. Completion still requires larger/transformed raw-matrix combinations and broader side-image corruption/truncation fuzz coverage. | `ENT-D01` |
| `ENT-E01` | P1 | **Partial** | Add GPU ANS token serialization, histogram clustering, context clustering, hybrid-uint selection, general LZ77 search/distances, and canonical entropy metadata. `djxl`/`jxl` must accept every generated family. | — |
| `ENT-E02` | P2 | **Missing** | Select entropy configurations by effort and workload with deterministic modes. Report density and speed independently; no heuristic may change lossless pixels. | `ENT-E01`, `QA-05` |

The common entropy plan for `ENT-D02` now stores O(coded streams) packed runs and oversized-stream
geometry, including Modular DC-global/groups and VarDCT LF/HF/combined packets and AC. Budget
selection computes counts, peaks and lane occupancy without per-window tables. Packet/AC
submission derives parameters and fills one reusable host upload from retained compressed spans;
GPU commands are recorded at that same boundary. Near-u32-bit tests cover more than 134 million
windows, exact first/middle/last ABI records, shared LF-to-HF capacity and bounded Modular budget
selection. This closes eager host window/parameter/upload accumulation; the remaining side-image,
frame integration and conformance requirements above still apply.

### D. Modular decode

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `MOD-D01` | P0 | **Partial** | Direct integer-to-F32 RGB/RGBA reconstruction now preserves legal signed working values; the untouched official alpha_triangles reference checks this path. Codestream color/extra-channel counts are independent of output pixel formats. Gray/RGB plus integer extras at independent 1–31-bit depths use the bounded entropy/inverse pipeline; original declarations survive in public metadata. GPU finalization selects raw integer or normalized scalar F32 planes, expands Gray+alpha, and selects/rescales the first alpha for color output. Six libjxl fixtures cover eight types, multiple alpha planes, a legal one-leaf MA tree and transformed multi-group input, with exact source codes and Rust/libjxl F32 agreement. Modular color and extras now support standard/custom 2×/4×/8× interpolation, including dimension shifts and complete group assembly. All legal sample precisions are admitted; lossy XYB and original-sRGB Modular now execute through the shared color/restoration tail described below; completion requires broader original color domains and complete sample-range/render conformance within checked device limits. | `ENT-D01`, `COLOR-01` |
| `MOD-D02` | P0 | **Partial** | The stock path parses a bounded MA configuration independently for each pass group when `use_global_tree` is false, including its tree, Prefix/ANS tables, hybrid configs, context map, LZ77 contract, and custom weighted-predictor header. Self-contained descriptors are rebased into one GPU metadata buffer, equal locals are deduplicated, and a checked 256-byte `Pod` parameter selects each group's metadata base. Memory/state planning takes the maximum per-group LZ77 and Weighted/SelfCorrecting requirement, reports mixed Prefix/ANS frames explicitly, and retains fixed-kernel specialization only when every resolved tree proves the same contract. A public encoder policy emits shared-global or complete local-per-group descriptors. An actual six-group 515×259 stream is byte-exact through this GPU decoder, Rust `jxl`, and `djxl`; offset-rebase tests cover multiple distinct packed descriptors without shader-source substring checks. VarDCT packets now accept previous-channel properties through explicit consumer-specific geometry/storage access: LF uses component extents with zero Modular shifts; HF filters by correlation versus strategy/sharpness shifts and uses capacity-strided rows. Seventy native geometry cases provide 41,616 exact GPU property values, and 72 native custom-tree streams preserve complete output through whole and seven-byte-fragmented input with 40-byte GPU windows. No buffer/uniform/resume-state allocation grows. Completion requires every legal MA-tree property, decision, leaf multiplier/offset, shifted-channel reference, and custom weighted-predictor header without implementation-specific tree caps beyond advertised resource limits, plus generated tree differential tests. | `ENT-D01` |
| `MOD-D03` | P0 | **Partial** | The production metadata path parses arbitrary ordered stacks with all 42 validated RCT types, Palette fields including delta storage and predictors, and explicit/default horizontal/vertical Squeeze. A bounded typed IR meta-applies exact channel order, odd average/residual geometry, shifts, bit depth, and meta-channel boundaries. Its geometry/layout and entropy-consumer records are 32-byte `Pod` ABIs with checked WGSL-u32 offsets. One checked header identifies descriptor/reference/final-plane tables, entropy words, maximum width, and arena high-water; channel records add cumulative sample ranges and prefiltered same-geometry/shift references for MA properties 16+. A descriptor-specialized WGSL path reconstructs unequal channels directly at arena offsets, resumes bounded windows by cumulative range, uses flattened MA references, and sizes LZ77/Weighted scratch from the maximum transformed width without adding descriptor branches to direct kernels. Reverse traversal reconstructs prior topology with only two live channel vectors and a cumulative work bound. Resident Squeeze executes odd-tail reconstruction with normative `i64` tendency; resident RCT executes all 42 exact wrapping-`i32` operation/permutation types in place after loading three non-overlapping arena views. Both expose Scalar/32/64/128/256 linear policy variants and 64-byte aligned `Pod` uniforms. Resident Palette uses a checked 128-byte `Pod` uniform, implements explicit entries, negative implicit delta entries, both normative implicit color cubes, and all predictors. Predictor zero dispatches in portable 2D; serial predictors use bounded 262,144-sample chunks, with resident row/error state for exact SelfCorrecting continuation. A best-fit lifetime planner emits arbitrary RCT/Palette/Squeeze compositions in exact inverse order, leaves in-place RCT spans live, lifetime-colors Palette/Squeeze outputs, and reuses one predictor scratch span across Palette output channels. Production accepts single- and multi-group compositions. RCT-only multi-group streams preserve the per-group inverse/finalizer fast path. For nonempty DC-global Palette/Squeeze, the global entropy prefix reconstructs into a separately budgeted frame arena. LF groups first run local inverse plans for channels whose horizontal and vertical shifts are both at least three; pass groups then process every remaining channel, including asymmetric shifts. All subimages copy edge-aware rows into disjoint full-frame transformed views, and one shared inverse plan plus one checked 176-byte finalizer execute after the final pass group. Arena high-water, global decoded samples, LF-group stream count, total inverse and Palette dispatch counts, actual 64/128-byte job uniforms, and the finalizer uniform share the frame byte budget and public memory statistics. Naga, scalar, malformed-plan, and actual-adapter tests cover extremes, odd/zero residual geometry, padded placement, all RCT types, every Palette predictor, explicit/implicit delta entries, bounded SelfCorrecting continuation, native packing, NV12, exact-widened F64, out-of-range rejection, asymmetric channel shifts, and arbitrary compositions without shader-source string tests. Optional real `cjxl` fixtures cover single-group Palette, 515×259 six-group local transforms, a 515×259 six-pass-group DC-global Palette stream, and a 2051×259 DC-global Squeeze stream with two LF groups, all with byte-exact GPU/Rust-`jxl` output. A real optional progressive-DC fixture fixes 13 parameters, 40 entropy channels, 37 jobs, three full outputs, and a two-times arena bound. Lossy/XYB Modular and LF producers now share normalization, restoration and resampling before color conversion or dependency retention. Completion still requires broader Global/LF/HF side-image and progressive-reference integration plus additional libjxl pixel conformance. | `MOD-D01` |
| `MOD-D04` | P0 | **Partial** | Standard Modular group geometry derives and validates every 128/256/512/1024 size from `group_size_shift`, including edge-group origins and extents. Transformed multi-group execution supports shared DC-global RCT/Palette/Squeeze, arbitrary local RCT/Palette/Squeeze stacks, shared-global or local per-subimage MA/entropy configurations, and exact GPU validation of zero- or nonzero-sample DC-global entropy in the final aggregate status map. One through eleven passes are lowered through the normative downsampling/last-pass shift brackets: each non-LF channel is owned by exactly one pass, empty physical sections are zero-validated without dispatch, nonempty streams execute in pass/group order, and the public profile/stats retain the declared count. Cross-group transforms use a frame-resident transformed arena: LF-group planes with both shifts at least three are assembled before pass-group planes and one global inverse/finalizer. A generated `cjxl` 2051×259 two-pass Squeeze stream is byte-exact against source and Rust `jxl` on an actual adapter. Frames whose image samples reside entirely in DC-global now use zero subimage lanes and run the shared inverse/final validation in the last global submission; the checked recursive DC-plus-quantized-AC stream exercises this root on GPU. Recursive LF dependencies now execute through the common physical graph, including filtered/resampled Modular roots with extra channels. LF sources validate every extra and retain three reconstructed XYB planes for prediction; LF patch features and presentation own normalized extras separately. LF-consuming VarDCT frames now decode distributed LF extras before HF metadata through the common GPU subimage executor. Two 2051×33 Squeeze fixtures cover both LF producer modes, every extra and two LF groups. Completion requires broader Global/LF/HF image streams, transform combinations and legal group permutations. Composed color, native integer and scalar F32 updates preserve final-only bytes and presentation metadata. Every exposed intermediate output must converge exactly to final output. Optional typed global/LF and residual-pass images reuse final output conversion on immutable arena copies, with on-demand continuation and pass-local validation; all 1–11 pass counts now execute, including equal-count downsampling boundaries and empty leading/interior/trailing passes. Header and shift ownership tests enumerate every representable boundary schedule; 40 plain/Squeeze arrangements match native source codes and libjxl F32 prefix images. Images begin only after samples exist. Eleven-pass 40-byte continuation covers exact-budget retry, cancellation, late corruption and final-only switching. Broader transform/side-image combinations and composition conformance remain open. | `MOD-D03`, `FRONT-02` |
| `MOD-D05` | P1 | **Partial** | Shared Modular substream execution now drives raw quantization matrices and public VarDCT global extra channels. GPU entropy/inverses finish before the checked bit cursor resumes LF parsing; first declared alpha stays resident through fused color output. Six internal fixtures prove every original plane and seven public fixtures prove color/alpha, progressive multi-entry TOCs, Apply/Keep, fragmented async input, corruption and cancellation/admission. Independent arena/transient permits cover stage transitions. Global extras now resume over one input/parameter pair with 16-byte overlap, a four-byte sentinel and a 40-byte minimum; validated early termination avoids uploading the later packet suffix, and inverse transforms wait for completion. Lazy window geometry is constant-space, budget capacity can shrink the upload, and canceled middle-window maps retain their permits. Whole/bounded comparisons prove all original planes and exact unaligned cursors, including zero-bit single-symbol Prefix input. All 32 global extra planes in the seven public fixtures now also return native unsigned or scalar F32 through individual output selection, with full color entropy validation and no color reconstruction surfaces. Public LF/AC distribution now assembles local inverse results into a GPU frame arena before one global inverse/output tail. Five fixtures cover all 13 native/F32 extra planes, two LF groups, progressive Squeeze, bounded fragmented input, cancellation, budget pressure and malformed extra entropy. All selected extra views can now remain resident together for frame composition, sharing their arena reservation once; nine-extra distributed and resampled animation fixtures exercise that boundary. Raw matrices now use that same bounded source-span executor, with inverse/overlay only after checked completion, exact late admission and callback-owned frame/image lifetimes. Global/local-MA raw images, including independently local LF/HF packets, match whole execution and independent public RGB8 oracles. Completion requires all embedded LF/quant images, larger/transformed raw matrices and broader cross-feature conformance. | `MOD-D03`, `VDCT-D01` |

Integer source checkpoint for `META-01`, `MOD-D01/03/05`, `FRAME-03`, `COLOR-03` and `IO-01`:
Modular primary/extras and XYB VarDCT now admit every 1–31-bit integer declaration. The public
`native_modular_pixel_format` constructor represents the full range with 8/16/32-bit storage.
Exact unfiltered source codes, independent alpha rescaling and final F32-to-integer quantization
use portable GPU integer arithmetic. Scalar output uses storage width rather than rounded-up
valid bytes, and native packing writes all four bytes for 17–31 bits. Broader color requests
route through the existing accounted F32 presentation surface; no image readback is added.

The reproducible libjxl corpus adds 42 exact-source/predictor/RCT/alpha fixtures and 40 rendering
cases, including every 25–31-bit XYB declaration, 24–31-bit extras, real Squeeze, progressive DC,
shifted 2×/4×/8× resampling and seven nine-layer animations. Exact native delivery and F32
normalization are separate contracts: rendering and lossy reconstruction retain their existing
F32 precision, with final rounding preserving the information remaining at presentation.
`MOD-D01`, `COLOR-03` and `IO-01` remain **Partial** because additional original
color domains, the remaining render graph and full conformance gates are still open.

Lossy Modular color checkpoint (`MOD-D01`, `RENDER-02`, `COLOR-01/03`, `FRAME-01`): the production
frontend now admits XYB and restored original-sRGB Modular frames. One frame arena precedes
sample/XYB interpretation, resident Gaborish/EPF, resampling and the shared `color_output` packer.
The frame-constant Modular EPF sigma uses the existing 80-byte filter uniform and no image
allocation. XYB dequantization, inverse opsin, grayscale projection, alpha, spots and original-sRGB
reference composition stay on GPU. Unreferenced XYB remains linear until final presentation.
Nineteen reproducible streams cover all EPF iteration counts, all orientations, 2×/4×/8× factors,
distributed Squeeze, floating extras and four nine-layer animations. Independent planes and
whole/bounded output have libjxl evidence; Rust independently covers all stills and initial
Replace presentations. These rows remain **Partial** until the remaining original color,
render/reference and conformance requirements are proved.

Raw matrix window checkpoint (`ENT-D01/02`, `MOD-D05`, `VDCT-D05`): mode-7 images now use the
same source-span Modular executor as global and distributed extras. The duplicate resident-copy
path is removed. One bounded input/parameter pair carries entropy and predictor state; inverse
transforms and overlay execute only after validated completion. Late admission can shrink the
upload to 40 bytes, and map callbacks retain both matrix and frame allocations through cancellation.
Checked global/local-MA images, including independent local LF/HF packets, prove bit-exact
matrix/cursor agreement, unchanged destinations on one-bit truncation, public RGB8 agreement with
Rust jxl and libjxl, and cancellation during initial, continuation and finalization submissions.
Both local-MA fixtures are reproducible without changing image tokens. The fully local case also
fixes a final-validation mismatch: expected HF termination now follows the entry point actually
submitted, preserving status 31 across the late raw stage. These rows remain **Partial**:
larger/transformed raw matrices, broader embedded-image combinations and full conformance still
need evidence.

Composed progression checkpoint (`META-03`, `MOD-D04`, `VDCT-D06`, `FRAME-03`, `API-03`):
the common compositor now forwards progressive requests for native integer and scalar F32 output
as well as color. Three native two-pass Modular animations cover RGB8, Gray16/associated alpha5,
and floating Gray16/exponent5 with associated alpha24/exponent7. They exercise nine physical layers,
six presentations, signed crops, all five blend modes and independent alpha references. Native
standalone prefix images plus independent F64 reference blending verify each update; scalar finals
also agree with the original coalesced animation. Existing VarDCT associated-alpha layers verify
numeric extra progression through the same path. Across 48 output/orientation/window configurations,
936 GPU images preserve metadata, retained-image immutability, whole/40-byte equality and exact
final-only convergence. Modular color and all numeric F32 comparisons use a 3e-6 normalized-error
bound; VarDCT color retains its separate 1e-3 composition bound. Native integers differ by at most
one code. Public cancellation/final-only transitions and late-pass/hidden-frame corruption cover
24 and eight cases; 48 deterministic submission-stage cases additionally prove allocation failure
and retirement. Composed F64, legacy Gray8 numeric layouts, broader LF/transform/render conformance
and incomplete-frame input remain open. These rows remain **Partial**, and this checkpoint does not
close encoder syntax or quality gates.

Numeric color checkpoint (`IO-01`, `COLOR-03`, `VDCT-D06`, `FRAME-03`, `API-03`):
`NumericChannel` unifies exclusive color/extra selection. Gray defaults to its only component;
RGB requires an explicit red/green/blue index. The common engine routes VarDCT scalar color
requests through unquantized GPU frame surfaces and the shared original-encoding packer. Codec
inverse transforms, restoration, resampling and reference composition precede scalar selection;
presentation alpha/spot policies leave numeric values unchanged. Native unsigned clamps/rounds
once at the source depth, while normalized integer and native floating requests retain F32 values.
Lossless Modular direct selection preserves exact source codes without a floating round trip.

`tests/numeric_channels/main.rs` covers 17/31-bit exact Modular selection; libjxl original components for
integer/float, RGB/gray, associated alpha, spots, recursive DC, resampling and composed frames;
original-RGB and JPEG reconstruction with noise; native prefix images under both orientation
policies; cropped LF previews with retained references; and exact-rational integer packing.
Whole/bounded fragmented outputs are identical. Cancellation, late entropy corruption, retained
image immutability and allocation-failure retry use the common lease/validation contract. Scalar
outputs match the existing RGB reconstruction within a separate 5e-6 bound, while independent
native comparisons retain each corpus's reconstruction tolerance. All listed rows remain
**Partial**: legacy Gray8 numeric layouts across both modes (including composed F64), original
non-sRGB/ICC domains and the remaining render/progressive/conformance gates are still open.

### E. Modular encode

The [wide-integer encoding checkpoint](CONFORMANCE_CORPUS.md#wide-integer-modular-encoding)
extends packed Gray/RGB/RGBA to every 1–31-bit depth. GPU comparison-based Gradient and
modulo-32-bit residual arithmetic feed a 33-symbol raw prefix alphabet; source pixels stay on
GPU. Ninety high-depth stills, nine three-frame Replace animations, and streamed 16K×1 and resident
257×9 RGBA31 images have exact original-word and native normalized oracles, whole/bounded GPU comparison,
canonical-artifact rejection and admission/cancellation/retention evidence. The old Gray8 fixture
and low-depth prefix policy are unchanged. This advances `ENC-01`, `IO-02` and `QA-03/06` while
those items remain **Partial**; other layouts, independent extras, adaptive
predictors/transforms and parallel token production remain separate work.

The [IEEE floating-point checkpoint](CONFORMANCE_CORPUS.md#ieee-floating-point-modular-encoding)
adds packed native binary16/binary32 Gray/RGB/RGBA with exponent-aware capability negotiation,
headers, alpha precision and animation descriptors. GPU integer prediction preserves raw IEEE
words without a float conversion or RCT. Fifty-two still streams cover shared/local trees, every
binary16 word, both signs at every binary32 exponent, NaN payloads, infinities, signed zero and
subnormals. Retained jxl-oxide working words and native libjxl F32 output provide independent
exact comparisons; whole/bounded GPU decoding checks all components, alpha and color output.
Six three-frame Replace animations check timing and retained output, while two four-frame finite
animations check crops, Add/Multiply and independent color/alpha reference fields against Rust
`jxl` and libjxl. The serializer now follows each channel's own blend mode when deciding whether
to write its reference field. Resident and streamed RGBA32 retain exact budget admission,
cancellation, deterministic blocking/Future assembly and pool reuse. `ENC-01`, `IO-02`,
`FRAME-05` and `QA-03/06` remain **Partial**; other floating precisions, independent extra planes,
embedded ICC, other input normalizations, learned transforms/predictors and progressive encoding
remain open.

The [buffer-layout checkpoint](CONFORMANCE_CORPUS.md#lossless-modular-buffer-layouts) advances
`ENC-01`, `IO-02`, `FRAME-05` and `QA-03/06` with direct GPU addressing of packed, planar and split
color/alpha buffers. Bijective swizzles, shared words, arbitrary integer bit placement and both
word endiannesses preserve canonical codestream bytes. Four independently bounded source bindings
avoid plane-gap exposure and intermediate normalization storage. The matrix covers 102 still
layouts plus resident/streamed lifetime cases and four animations whose layouts change per frame,
with independent retained integer/IEEE words, native output and whole/bounded GPU output. All
affected items remain **Partial**: arbitrary extra-channel declarations, embedded ICC,
YUV/textures, other floating precisions and the remaining encoder syntax/quality gates stay open.

The [source-color checkpoint](CONFORMANCE_CORPUS.md#lossless-modular-source-color) retains
full-range enumerated RGB/Gray primaries, white and transfer declarations, all four intents and
positive exact binary16 image white. Independent native ICC bytes, original integer/IEEE words,
requested F64 color conversion, Replace animation and exact admission/cancellation checks cover
the contract. Source words and GPU memory layout are unchanged. Embedded ICC and the remaining
input/encoder contracts keep `ENC-01`, `IO-02`, `FRAME-05` and `QA-03/06` **Partial**.

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `MOD-E01` | P1 | **Partial** | Replace the scalar per-group token scan with correct parallel predictor scans, compaction, and hierarchical histogram reduction. Preserve bit-exact artifacts and prove no races at workgroup sizes 32/64/128/256. | — |
| `MOD-E02` | P1 | **Missing** | Encode all group sizes and transform stacks: local/global RCT selection, Palette/delta palette, Squeeze, and the corresponding global/LF/HF topology. Round trips cover every transform and composition order. | `ENT-E01`, `MOD-D03` |
| `MOD-E03` | P1 | **Partial** | Add all predictors, learned MA trees, weighted-predictor parameter search, previous-channel properties, and bounded effort tiers. Validate compression choices against the fixed Gradient baseline. | `ENT-E01`, `MOD-D02` |
| `MOD-E04` | P1 | **Missing** | Add progressive/responsive Modular and lossy Modular with an explicit error/quality contract. Intermediate passes and final output must satisfy the appropriate exact or bounded comparison. | `MOD-E02`, `MOD-D04` |
| `MOD-E05` | P2 | **Partial** | Stream/batch many independent images or frames through shared GPU passes and artifact pools. A batching claim requires fewer codec submissions than logical images and no hidden per-image map. | `API-04`, `MOD-E01` |

### F. VarDCT decode

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `VDCT-D01` | P0 | **Partial** | One LF-global/multiple-LF-group/HF-global topology executes nonempty pass groups across spectral/refinement passes in scanline or arbitrary entropy-permuted physical order, including center-first. LF groups own independent packet/HF workspaces but scatter into shared full-image resident atlases and one output. Sectioned shared-global-tree packets resume their complete LF/HF range through one reusable upload without an intermediate map; the final packet command is co-submitted with downstream work. Actual-GPU 2056×256 fixtures force more than two packet batches and validate one final aggregate status map, exact submissions, async completion, cancellation release, and typed late-window corruption. Ordinary `cjxl` per-substream local trees execute through bounded LF submissions, one aggregate LF map, host descriptor packing, bounded HF submissions, and one final aggregate map after downstream work. Their final HF command is likewise co-submitted with the downstream prefix. A generated 2056×256 effort-7 stream with a forced 256-byte cap exercises exact dynamic submission accounting, blocking, runtime-neutral async, typed early-handoff refusal, late-HF corruption, and cancellation memory release. Recursive LF chains also execute a single-entry intermediate's GPU HF metadata, cursor-discovered HF-global tables, general AC, and resident output before scheduling the next dependency; `cjxl --progressive_dc=2` is checked in four physical submissions behind one logical final frame. Independent per-pass entropy/order descriptors and shifts now feed bounded accumulation before one final reconstruction. Single-entry packets now use staged LF and HF-metadata cursors for general HF-global/AC parsing regardless of global-tree presence. The host no longer infers a uniform transform from image dimensions; checked custom-resampling and single-sample fixtures execute arbitrary small extents. Completion requires broader legal packet topologies and intermediate presentation. | `FRONT-02`, `ENT-D01` |
| `VDCT-D02` | P0 | **Partial** | Sectioned packets now GPU-decode mixed strategy maps for all 27 regular and special strategies, capacity-strided quant fields, `hf_mul`, extra precision, every 3-bit X/B quant-matrix scale, per-frequency-cell HF chroma correlation, and arbitrary valid block-context maps selected from stream-defined quant-field and signed X/Y/B LF thresholds. Global LF/correlation/resource origins cover cross-group addressing. An actual-GPU scalar context matrix, a mixed libjxl image, two standard multi-LF-group images, and ordinary local-tree `cjxl` output cover the production path without lowering a legal strategy to DCT8. Completion requires required Modular side images and broader mixed-strategy/correlation conformance coverage. | `VDCT-D01`, `MOD-D05` |
| `VDCT-D03` | P0 | **Partial** | The coefficient executor decodes real nonzero counts, multiple HF presets, contexts, Prefix/ANS tokens, signs, and all 13 natural/custom coefficient-order families, then scatters coefficients in the regular or special transform's required layout entirely on GPU after the small order permutation is expanded on the host. A 438×589 DCT8 fixture and a 257×257 mixed-strategy/custom-order libjxl fixture match Rust `jxl` and `djxl` within one RGB8 code. The one-pass restriction is removed: each of up to eleven declared passes owns its entropy/order tables, coefficient shift, bounded resume state, LZ77 history, and validation status while atomically accumulating into shared resident coefficients. Checked-in libjxl three-pass spectral and two-pass quantized fixtures cover odd extents, center-first permutation, multiple LF groups, independent tables, blocking/async completion, 37-byte transport chunks, 256-byte GPU windows, late-pass corruption, and cancellation. Their final RGB8 agrees with Rust `jxl` and `djxl` within one code on Apple M5. Completion requires broader sparse/dense/order combinations and coefficient-layer conformance. | `VDCT-D01`, `ENT-D01` |
| `VDCT-D04` | P0 | **Partial** | The stock decoder dispatches every one of the 27 compact strategy buckets through the resident regular or special inverse-transform kernels, with one artifact/scratch plan per LF group and shared output planes. Kernel tests cover every strategy, a libjxl mixed map covers odd padded edges, and a two-LF-group fixture covers repeated renderer dispatch into one image. A spectral stream transplanted with a raw JPEG DCT8 matrix exposes extended linear values: final-only and progressive GPU output are byte-identical, but direct native-linear comparison reaches 0.001915 absolute error (0.000676 normalized by max(1, abs(reference))). This remains an explicit precision gap, not a conformance pass. Completion requires libjxl coverage of every strategy in mixed maps and ISO 18181-3 precision evidence. | `VDCT-D02`, `VDCT-D03` |
| `VDCT-D05` | P0 | **Partial** | Every normative default strategy matrix and all parametric custom encodings 0 through 6, all 3-bit stream-selected X/B scales, stream-selected global/LF/HF scales, non-default LF channel dequantization, LF and per-cell HF chroma correlation, extra precision, quant bias, and DC prediction execute on GPU for the bounded global-, local-tree, and progressive-intermediate profiles. Bounded scalar matrix parameters are expanded with normative orientation for regular, wide, and special coefficient layouts and overwrite the resident matrix region before AC/render; no CPU coefficient or pixel decode is used. Mode 7 parses each complete three-channel Modular side-image header, global/local MA-tree selection, transform topology, exact entropy stream index, denominator, inverse plan, and resumable HF-global tail. Execution reserves exact buffers from the shared byte budget, reuses one four-byte-aligned input window from source spans under caller/device and available-budget limits, runs common GPU entropy plus resident Palette/RCT/Squeeze inversion, rejects invalid weights with typed status, overlays one canonical raster into every aliased resource target, and rebases/resumes repeated raw matrices before AC. Later scalar-metadata uploads exclude those GPU-resident raw ranges; oriented grayscale and 4:2:0 JPEG cases guard against overwriting them with placeholder defaults. Staged HF metadata and AC traversal retain the MCU-padded edge block grid. Local-tree packets enter that state only after every LF cursor and bounded HF-local metadata stream validates. Checked-in cjpeg-to-cjxl JPEG-transcode codestreams fix the wire contract and execute the DCT8 matrix primitive plus complete public 4:4:4/4:2:2/4:4:0/4:2:0 presentation on an actual adapter. LF and HF consumers use exact Y/Cb/Cr dimensions, the resident resource/task ABI carries channel-specific bases, strides, offsets, masks, and destinations, and output fuses normative quarter/three-quarter edge-replicating upsampling with encoded BT.601 conversion. For signaled restoration, shifted components instead expand into separately budgeted full-resolution resident planes before Gaborish/EPF; horizontal, vertical, fused two-axis, and odd-edge interpolation execute on an actual adapter. Rust `jxl` and optional `djxl` differ by at most one RGB8 code for the checked restoration-disabled codestreams. Two reproducible local-MA fixtures preserve the original tokens while embedding descriptors in the raw image alone or in every LF/HF/raw substream without a global tree. All three forms match exact DCT8 matrices/cursors and public RGB8 output under whole and bounded input; truncation, late admission and cancellation have actual-GPU checks. Another 128 independent-entropy streams cover all 64 component sampling selector triples at 272×32 and 257×17, including nonzero/zero noise and equal-factor adaptive LF. Eight nonzero-LF-correlation streams additionally require equal nonzero factors to preserve LF correlation; the engine now uses normalized component shifts for that decision. Whole and bounded F32 outputs match exactly and agree with native/Rust references. The shared parser classifies unequal-factor adaptive LF as malformed before TOC/section delivery; public inventory negotiation uses the same validation. Completion requires larger/transformed raw-matrix conformance, other required Modular side images, broader asymmetric restoration/resampling and correlation/dequantization corpora. Multiple LF groups scatter into full-image LF/correlation atlases; adaptive LF smoothing runs once globally, while the skip flag writes directly or uses a resident smoothing buffer as required. The frontend enforces libjxl-compatible dequant/base-correlation/matrix bounds, bit-level tests cover parametric and malformed encodings, actual-GPU LF/HF probes verify the parameter ABI, and generated custom-header plus `--progressive_dc=2` streams agree with Rust `jxl` and optional `djxl` within one RGB8 code where applicable. | `VDCT-D02`, `VDCT-D03` |
| `VDCT-D06` | P1 | **Partial** | Spectral and quantized progressive AC passes accumulate on GPU before dequantization/restoration. Opt-in stills, animations and composed presentations, including extra channels, publish immutable DC images using normative image-header 8× interpolation, followed by complete image-wide AC passes and the final frame. Deferred HF-global/raw-matrix continuations begin after DC publication; `next_frame` remains final-only. Per-pass status snapshots validate only completed entropy, and all updates share one logical frame slot without advancing animation time. A checked-in three-frame recursive DC plus quantized-AC stream also executes its global-only Modular root, intermediate VarDCT frame, and final AC passes without pixel readback. Six independent alpha/depth streams now cover Modular/VarDCT LF roots and LF2-to-LF1 consumers whose global Modular extras must complete before HF metadata; tracked XYB leases survive every bounded cursor continuation. Two further Squeeze streams prove LF-consumer extras distributed into two LF groups: a typed coefficient/extra/HF entry contract starts at known section offsets, validates GPU extra cursors and admits late HF descriptors without a fabricated LF stream. Whole and bounded output match both independent decoders, with per-group corruption and cancellation coverage. Eager HF-only LF consumers now share the typed LF/HF/combined window plan, exact predictor-state allocation and cap accounting. Packet and AC commands are recorded immediately before submission, including deferred AC; a 1024×128 recursive DC+AC stream matches whole input through 40-byte windows and fragmented async input without exhausting Metal command resources. Eleven descriptor passes and shifts 0–3 have bounded parser/truncated-tail coverage. Twenty-three oracle cases cover DC, spectral/refinement passes, JPEG sampling/raw matrices including noisy asymmetric and equal-factor MCU-padded odd edges, grayscale, custom 4×/8× weights, orientation, multiple LF groups, 37-byte transport, 256-byte GPU windows and retained images. Single-entry/deferred paths additionally cover thin/one-sample extents, late corruption, cancellation and budget admission. Native flush comparisons stay within one RGB8 code; final GPU bytes equal final-only decoding. Completed Modular/VarDCT LF dependencies now publish immutable full-canvas images after their exact background reference validates and before terminal consumer admission; four color/gray/custom/noisy LF2 chains and scalar expansion through LF4 cover the new typed physical boundary. Only exact dependencies of the presentation publish; unused versions still validate and reused slots are not republished. Seven animation families now publish DC/AC snapshots only from the terminal physical producer after hidden layers validate. Every composed snapshot reads committed references and packs a separate output; references change only on complete physical frames. Independent native layer flushes plus scalar composition, coalesced native final checks, exact timing/IDs, Apply/Keep orientation, whole/40-byte fragmented input and staged cancellation/pressure/final-only switching have adapter evidence. Composed LF images now use canonical physical-layer surfaces and retain queued planes until the exact terminal background reference validates, even after LF predictor-slot expiry. Original and deferred-reference sequences compare LF1 against independent Rust layer flushes plus scalar composition, with finite normalized linear error below 0.000285 under the existing 0.001 composition regression bound. Whole/40-byte fragmented input, Apply/Keep, hidden-reference corruption and 21 queued/render/blend/pack lifecycle cases have adapter evidence; LF2 has no native per-level pixel-oracle claim. Integer/floating extra subimages now validate at each coefficient boundary; independently owned assembly copies undergo global inverse and resampling before alpha/spot/color output. Eight fixture families cover native prefix/final precision, exact whole/40-byte fragmented bytes, retained images and memory release. All alpha policies, orientation and planar/linear output have 24 configuration checks. Independent native layers plus F64 associated-alpha composition cover nine layers and six presentations, with coalesced-native final checks. Corruption in each extra pass, cancellation inside a bounded extra subimage, initial/late pressure and final-only switching preserve earlier images and release budgets. Completion requires broader LF conformance, broader composed precision conformance, incomplete-frame readiness and broader per-level conformance precision evidence. LF images now include alpha/depth and native/scalar F32 extra output for Modular and VarDCT roots, including nested LF2 and distributed groups. Independent small native producers plus F64 pre-opsin expansion validate this presentation policy. | `VDCT-D03`, `API-03` |
| `VDCT-D07` | P1 | **Partial** | The public VarDCT decoder executes JPEG reconstruction's 4:4:4, 4:2:2, 4:4:0, and 4:2:0 component layouts with normative quarter/three-quarter weights and replicated odd borders inside the resident output pass. The same horizontal/vertical kernels, with a fused two-axis form, now expand shifted components before a signaled full-resolution restoration sequence without readback. Ordinary VarDCT frames now use their encoded sample grid for entropy/restoration and the exact presented grid for 2×/4×/8× resampling. Standard and custom compact weights expand into one shared phase-major kernel, with three resident 5×5 filter dispatches before color conversion. Seven checked libjxl fixtures cover all factors, nearest-neighbor weights, spectral AC plus 4× resampling, odd extents, one-sample axes, and a 4111×17 two-LF-group output; whole and bounded async results differ from Rust jxl and djxl by at most one RGB8 code on Apple M5. Exact plane/weight/uniform costs share the frame budget, and an undersized budget is rejected before submission. Twenty 257×17 JPEG streams now combine Gaborish, effective EPF1/2/3 and noise with all four ordinary sampling layouts plus gray under whole and 256-byte fragmented input. Another 128 streams exercise all 64 component sampling selector triples at aligned and odd extents, including smaller Y, independent chroma shifts, padded equal factors, nonzero/zero noise and active equal-factor adaptive LF. Every whole/fragmented F32 output is identical and agrees with both native and Rust references. Completion requires broader asymmetric restoration/resampling combinations and conformance precision. | `MOD-D05`, `RENDER-01` |

Subsampled adaptive-LF smoothing remains an unresolved compatibility/conformance item, not a
proven malformed-header classification. Native libjxl 0.12 still rejects it, while Rust `jxl` 0.6
removed that restriction in [jxl-rs PR861](https://github.com/libjxl/jxl-rs/pull/861), merged on
2026-08-16. The related [libjxl PR4932](https://github.com/libjxl/libjxl/pull/4932) is still open and
calls for a specification clarification/change. Keep the typed unsupported boundary until the
normative behavior and an interoperable corpus are established.

Modular predictor arithmetic for `MOD-D02/03` now covers the full signed 32-bit working-word domain.
Entropy and Palette share all 14 predictors with portable 64-bit intermediate arithmetic, while
committed error rows and bounded-input resume layouts retain their normative 32-bit storage.
Implicit Palette color scaling supports all working depths through 32 bits. Direct GPU comparisons
cover arithmetic boundaries, binary32 bit patterns, all predictor outputs and error updates, and
all implicit color components at every depth. The floating sample checkpoint below connects
representation conversion and delivery for `MOD-D01` and `IO-01`.

Floating source checkpoint for `META-01`, `MOD-D01/03/05`, `FRAME-03`, `COLOR-03`, and `IO-01`:
all 154 legal precision combinations now decode into binary32 on GPU. Original sample encoding
is independent of transform geometry and resident state; `SampleBitDepth` is public, and scalar
floating output uses `NativeFloat`. Integer bit assembly preserves signed zeros, subnormals,
infinities and NaN payloads for unfiltered, uncomposed delivery. No-op RGB F32 output copies those
words as well. Integer and floating extra planes coexist and convert before resampling, alpha,
spot rendering and reference/crop/blend operations. VarDCT retains floating metadata while its
XYB and progressive-DC working values remain in their own reconstruction domain. Color integer
output quantizes at presentation. Multi-entry Modular frames with transform-free DC-global-only
samples now receive a frame arena even when every pass group is empty.

Checked-in libjxl 0.12 references cover all 154 precisions and 27 rendering cases: mixed extras,
associated alpha, 2×/4×/8× reconstruction, shifted channels, global-only and distributed streams,
Squeeze, progressive DC, orientation, and five nine-layer animations. Whole and 256-byte bounded
fragmented GPU output agree exactly; reference comparison separates sample-bit preservation from
the existing tolerance for filtered/VarDCT/composed arithmetic. These rows remain **Partial**:
broader original color domains,
pre-transform references and progressive output still have independent completion gates.

Progressive-DC checkpoint for `MOD-D03/04` and `VDCT-D01`: the common physical frame plan executes
each LF node exactly once and retains its slot version through the final consumer. Modular LF
producers share presentation normalization, Gaborish, constant-sigma EPF and resampling, then stop
before color conversion. Both Modular and VarDCT retain the final pre-color-transform planes with
restored/upsampled geometry; independently leased final planes outlive transient scratch. A
single-entry intermediate executes GPU HF metadata to a validated cursor, then general
HF-global/AC and resident reconstruction before the next dependency. Fixed and cursor-discovered
allocations share one byte budget; only complete presentations are published.

Actual-GPU `cjxl --progressive_dc=1` and `=2` fixtures cover blocking and runtime-neutral async
paths. Eleven reframed libjxl Modular root configurations exercise default/custom Gaborish,
EPF1/2/3, custom sigma and 2×/4×/8× upsampling; both independent decoders agree within one RGB8
code with whole input and 256-byte windows. Separate allocation tests cover all pass parities and
retention after scratch release. Six alpha/depth streams now exercise staged global Modular
reconstruction in LF1 and recursive LF2 chains, including both LF producer modes, Gaborish,
bounded windows, unused-extra corruption and cancellation. Two 2051×33 Squeeze streams also
validate LF-group extras before HF parsing, preserve every plane through the frame arena and
reject either truncated group even for scalar selection. Conservative HF history/predictor/upload
capacity and exact late descriptor permits share the byte budget. Ten additional native-accepted
chains cover actual LF1–LF4 with independent integer/floating color and extra precision, associated
alpha, grayscale, EXIF orientations and independent 2× color/8× extra root resampling. Independent
small producers and F64 pre-opsin expansion validate native and composed LF presentations, including
16/20-bit native extras, scalar F32 output, all alpha-output policies and Apply/Keep orientation.
Cancellation at every deep-LF boundary and root/intermediate/background/consumer corruption retain
only previously validated output leases.

Eight additional LF1–LF3 chains cover intrinsic extra-channel shifts 1–3, signed floating source
depth and values above one, independent alpha/depth precision, and both LF root coding modes.
Sixteen cropped consumers use negative horizontal or vertical canvas origins and smaller block
rectangles than their LF producers. The frame plan accepts a containing producer and rejects an
undersized one. Prediction packs a top-left view with the original row strides and tracked leases;
LF presentation clips that same local grid before each recursive Up8 stage, then composites at the
signed origin. No pixel copy/readback is introduced for these views. Native final images and
independent small-producer/F64 presentation oracles cover color, alpha, depth, Apply/Keep orientation
and final-only convergence. Native 16/20-bit output, signed F32, cancellation, final-only drain and
root/intermediate/background/consumer corruption exercise the same ownership contract. Poisoned
padding and surplus producer rows/columns test clipping with both default and custom Up8 weights.
These rows remain **Partial**: broader LF crop/filter/color-domain combinations and incomplete-frame
input still require implementation or conformance.
Embedded preview/main selection is covered by the checkpoint above.

### G. VarDCT encode

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `VDCT-E01` | P0 | **Partial** | All 27 strategies now perform real forward transforms and normative LF extraction on GPU. `VarDctStrategy` is the shared `TransformKind`, with no duplicate strategy alphabet or fixed diagnostic kernel. Regular transforms use separable storage passes; special transforms use bounded strategy-only bases. Correct LF comes from the raw LLF rectangle and resampling factors. All 667 pinned native coefficient/LF cases, including complete 8x8 impulse bases, match under five byte-identical linear variants. Fifty-four textured single-transform cases cover default/custom correlation, independent f64/native AC checks and native/Rust/GPU decode agreement within one RGB8 code. Tiled DCT8 retains odd-edge replication, 2048-pixel LF and 256-pixel AC groups through the checked 16K axis bound, with 2057x2057 nonzero AC and exact-black 16K panoramas. Caller-selected mixed maps now validate coverage, overlap, bounds and AC-group ownership before dispatch; per-strategy batches share image-wide arenas and variable-capacity AC slots. Native/f64 AC checks and three-decoder agreement cover all 27 strategies in a 512x512 map, a 2057x17 LF boundary and 13x21 non-DCT8 replicated edges, with five byte-identical variants. Batched native transform/LF tests cover disjoint/reordered ranges and poisoned guards. Completion still requires broader scalar/ISO conformance and bounded execution beyond this fixed RGB8 profile; content-adaptive strategy search remains VDCT-E04. | `VDCT-D04` |
| `VDCT-E02` | P0 | **Partial** | All 27 strategies, including caller-selected image-wide mixed maps, and tiled DCT8 quantize real AC with shared default matrices, natural orders and one prefix cluster for all 495 contexts, with LZ77 disabled. Native metadata verifies every order exactly and all AC matrix entries within 3e-6 relative error; the largest rectangular Y/B defaults are corrected in both encoder and decoder. General transforms retain raw/quantized coefficients on GPU and emit one checked AC fragment per transform, with exact strategy-specific capacities; tiled DCT8 keeps coefficients in 2 KiB workgroup memory and emits one fragment per block. Exact memory plans retain every parameter, resident plane, basis, matrix/order, artifact and readback through validation/cancellation. Tests cover all 54 strategy/correlation streams under five variants, dense maximum 256x256 fragments, malformed/missing artifacts, independent storage limits, exact/one-byte-deficient budgets and abandoned nonzero work. Existing f64 boundary checks include 2057x2057. Completion requires custom orders/matrices, adaptive clustering or ANS/LZ policy, multiple passes, broader coefficient conformance and bounded larger-image execution. Stateful entropy must remain on GPU. | `ENT-E01`, `VDCT-E01`, `VDCT-D03` |
| `VDCT-E03` | P1 | **Partial** | Exact global/LF quantizers and caller-selected per-transform HF multipliers now share typed configuration, GPU computation and serialized metadata. Raw 33-symbol prefix alphabets cover signed 32-bit values; GPU quantizers report overflow instead of clipping. The ineffective fixed-distance API is removed. Completion still requires adaptive quant selection, distance/quality control, quant bias, chroma-from-luma search, and a bounded rate-control loop with size/quality distributions. | `VDCT-E02`, `QA-05` |
| `VDCT-E04` | P1 | **Missing** | Select strategy maps and coefficient orders by content and effort, including all special transforms. Decisions must be deterministic when requested and must improve a declared objective over DCT8-only. | `VDCT-E02`, `VDCT-D02` |
| `VDCT-E05` | P1 | **Missing** | Encode spectral, quantized, and DC progressive modes plus center-first/saliency group ordering. Every partial stream must remain decodable at its declared progression point. | `VDCT-E02`, `VDCT-D06` |
| `VDCT-E06` | P2 | **Missing** | Add perceptual optimization tiers, including iterative adaptive quantization and quality feedback. Compare against `cjxl` at matched distance/size with Butteraugli and additional artifact-sensitive metrics. | `VDCT-E03`, `QA-05` |

### H. Rendering features and restoration

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `RENDER-01` | P0 | **Partial** | VarDCT uses the existing normative GPU 5×5 filter through a resident pipeline for 2×/4×/8× frame upsampling after restoration and before inverse opsin. The host expands only bounded standard/custom weights; the GPU mirrors borders, clamps to the source neighborhood, and writes the exact possibly odd output extent across group boundaries. Seven libjxl image fixtures match both Rust jxl and djxl within one RGB8 code under whole and bounded async input. Both-mode integer extras and Modular color now use resident normalization plus this filter at 2×/4×/8×, with effective dimension shifts, independent rates, accounted scratch and post-filter packing. Twenty fixtures and ten custom-header variants cover both producers, native/F32 delivery, bounded fragmented input, admission and cancellation. Two nine-layer sequences now verify resampled RGB plus nine extras with color factor 2 and effective extra factor 8 in both producers. Completion requires broader resampled crop/blend cross-products, channel recombination and the remaining render graph. | `FRONT-01` |
| `RENDER-02` | P0 | **Partial** | The bounded VarDCT path applies signaled default/custom Gaborish, constructs one full-image inverse-sigma plane from each LF group's stream `global_scale`/`hf_mul`/sharpness, and executes the one-to-three-iteration EPF0/EPF1/EPF2 sequence through one resident ping-pong scratch set. Shifted JPEG components use exact separately budgeted full-resolution destinations and 32-byte interpolation uniforms before that cursor; an actual-GPU differential covers horizontal, vertical, fused two-axis, odd extents, and edge replication. Odd 257x17 EPF2/EPF3 libjxl fixtures plus a 2056x256 LF-boundary fixture cover mirrored whole-image borders, cross-group neighborhoods, typed malformed sharpness, exact shared-budget accounting, and Rust `jxl`/`djxl` error at most one RGB8 code. Twenty JPEG streams now cover Gaborish and effective EPF1/2/3 after component expansion, before signaled noise, with custom sharpness/sigma parameters and exact whole/fragmented output. Scalar Gaborish and a pinned third CPU oracle distinguish documented vertical-subsampling defects in the other references. Completion requires broader custom-parameter and extreme sigma/edge corpora, full filter-graph composition, and 18181-3 precision coverage. | `VDCT-D05`, `RENDER-01` |
| `RENDER-03` | P1 | **Partial** | GPU Prefix/ANS/hybrid/LZ77 dictionary parsing uses bounded windows, a count/validate pass and exact-size resident command emission; only 16 control bytes are mapped. Both coding modes retain explicitly tagged codec-component surfaces. Ordered patch rendering handles all eight modes, bounds, signed offsets, independent alpha/extra selection, association, clamping and color-alpha override before inverse color conversion. Eighteen native fixtures cover empty/nonempty dictionaries, Gray/RGB, integer/F32, XYB, restoration and straight/associated alpha. Scalar extras and whole/fragmented outputs agree; separate tests cover 40-byte windows, 1100 occurrences, four slot versions, malformed fields, exact admission and staged cancellation. Pass refinements now apply the same dictionary to fresh component surfaces before display conversion, with eight native prefix fixtures covering both codecs, Gray/RGBA and nine mixed-precision extras. Whole/fragmented updates and final-only output agree; staged cancellation, allocation failure and later entropy corruption preserve earlier images and committed references. Separate LF patch previews now retain component-domain color and all extras until the hidden references and dictionary validate; eight single/nested LF fixtures have native component/scalar presentation oracles and exact cancellation/admission coverage. Both LF producer modes now apply the shared patch program to all components before committing prediction, including overwritten unused producers. Fourteen native fixtures cover single/nested dependencies, empty/overlapping dictionaries and VarDCT padded destinations; independent scalar plane checks and staged cancellation/admission verify separate prediction/extra lifetimes. Patched producers now retain coded geometry through patches, then complete frame upsampling and noise before prediction/reference publication. A 140-image native corpus covers both modes, original/XYB color, 2×/4×/8× factors, custom weights, equal-rate/early extras, noisy LF producers/consumers, padded destinations and overwritten reference chains. Pass/LF preview features preserve held images and reference identities; cancellation, final draining and allocation failures verify exact reservations. Retained VarDCT YCbCr surfaces now expand shifted JPEG components before patches even when filters and noise are absent, using exact existing transient admission. A 494-image corpus covers all 64 selector triples, noise controls, Gaborish/EPF, mixed Modular/VarDCT sources and LF roots, original RGB/YCbCr component crossings, all four overwritten reference slots and padded destinations. It includes 218 Modular YCbCr patch/noise and mixed-reference cases with global/local transforms, floating extras and resampling. Preserved native sRGB and an independent f64 transfer provide direct output references; 18 filtered sequences require agreement on independently expanded equivalents from libjxl and jxl-oxide. Case classification, controls, oracle selection and precision metrics are explicit. It records native fast-renderer/CMS limitations through independent references. Whole/fragmented progression, held images, final-only equality, post-transform reference rejection, exact admission and cancellation pass. Spline interactions now have a dedicated 68-stream feature corpus and twelve native progressive-prefix streams. Completion still requires broader original-color conformance, Modular YCbCr spline/LF-producer interactions and remaining ISO precision conformance. Resampled patch-bearing frames retain the normative equal-extra-factor requirement. | `FRAME-01`, `MOD-D05` |
| `RENDER-04` | P1 | **Partial** | GPU Prefix/ANS/hybrid/LZ77 spline parsing shares the bounded feature executor with patches, preserving exact LF-global cursors and resident quantized coefficients/control points. Bounded geometry applies quantization adjustment, base color correlation, centripetal Catmull–Rom interpolation, equally spaced continuous-DCT samples, signed thickness and clipping. Count/replay admits exact 32×32 tile caches; ordered per-pixel raster batches avoid floating-point atomics. Both coding modes execute patches, splines, late resampling and noise before LF prediction/reference publication. The official 60-frame animation_spline F32 reference uses per-frame maximum channel RMSE 0.0001 and peak 0.004 without clipping; whole and bounded fragmented output agree bit-for-bit. A further 68 feature streams and twelve native progressive-prefix streams cover both modes, original/XYB color, extras, custom upsampling, LF chains, noise, correlation and subsampled JPEG restoration. Tests cover malformed control points, coordinate/delta/work/output limits, multi-dispatch geometry/raster, initial retry, replay/body admission failure and staged cancellation. LF/pass updates preserve held images and committed reference versions. Explicit per-plane layouts and a shared single-filter schedule now support unequal color/extra factors, with native 2/4, 2/8 and 4/8 progressive references, integer/float LF roots, custom kernels and feature admission/cancellation coverage. Remaining gates are broader original-color profiles and the rest of the official conformance corpus. | `COLOR-01` |
| `RENDER-05` | P1 | **Partial** | XYB and original-sRGB Modular/VarDCT, plus JPEG YCbCr VarDCT, parse the bounded 80-bit model and share portable GPU SplitMix64/Xorshift128Plus generation, mirrored 5×5 convolution and luma-dependent addition after restoration/upsampling and before color conversion. Physical-frame inventory retains visible/nonvisible seeds across producer projection. Sixty-four fixtures plus zero-model variants cover both modes, all four Modular group dimensions and ordinary JPEG sampling layouts, RGB/gray, 16-bit and F32 sources, orientation, 2×/4×/8× upsampling, visible/nonvisible frames, independent custom base/LF correlations and a single-channel implicit palette. Whole and 256-byte fragmented output is bit-identical; dual F32 references and checked native linear snapshots distinguish documented oracle defects without loosening tolerances. Scalar-u64 GPU comparison covers partial row tails and wrapped seeds. Subsampled components expand before noise and final color conversion follows the actual plane geometry. Eleven public admission cases include original Modular normalization/render storage and VarDCT component expansion, retry, zero-model elision and cancellation. Twenty JPEG restoration cases require an observable EPF effect and verify filtering before noise. Seven LF chains cover Modular/VarDCT roots, individual models in nested levels, Gaborish, progressive AC and unchanged alpha/depth; physical seeds, reduced LF extents, exact bounded output and intermediate cancellation are checked. The preview corpus adds both-mode noise, all preview aspect encodings, non-final headers and leading noisy LF state; whole/bounded output and native preview/main comparisons agree. The patch-feature corpus now covers reference-only noise, overwritten reference chains, both LF root modes and 2×/4×/8× resampling after patches, including alpha/depth preservation and zero-model controls. The spline corpus now covers ordered patch/spline/resampling/noise combinations in both modes, LF chains and custom correlation. Completion requires broader LF restoration cross-products and remaining 18181-3 precision coverage. | `VDCT-D05` |
| `RENDER-06` | P2 | **Missing** | Encoder detection/selection for patches, dots, noise models, Gaborish inverse sharpening, and EPF signaling. Each tool needs an on/off corpus and an objective improvement gate. | decode counterparts, `QA-05` |

### I. Frames, animation, and composition

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `FRAME-01` | P0 | **Partial** | Numeric non-XYB ICC composition now retains explicitly tagged original components without interpreting opaque profile bytes; requested color conversion still validates the profile. Production crops use wide signed intersections and GPU Replace/Add/Blend/Mul/MulAdd in original-encoding F32 before output conversion/orientation. Empty references, negative/oversized/fully off-canvas frames and independent color/alpha source slots execute across both modes. Twelve libjxl fixtures include native/F32, Gray16+Alpha5, RGB12+Alpha5, mixed JPEG/Modular, recursive DC and layered stills; eleven match both oracles, while a clamped-Multiply case has analytic/libjxl evidence and a documented Rust-oracle defect. Associated source-over preserves premultiplied reference values until final output conversion; three nine-layer sequences cover both modes, Gray16/RGB12 with Alpha5, independent source slots and all blend modes against both decoders. All-channel planar surfaces now retain every integer extra through its own blend/reference/alpha selection. Seven nine-layer sequences cover nine declarations with two differently associated alpha planes, all five modes, Gray/RGB, both codecs, distributed groups and shifted resampling; libjxl verifies all six presentations and every extra, with Rust evidence for unshifted initial Replace presentations. Integer/floating channels now share the frame boundary. D65 BT.709/BT.2020/Display-P3 with Linear/sRGB/BT.709 and gray now compose in original encoding; 74 six-frame native sequences cover RGB/XYB/YCbCr in both modes, independent alpha and reference overwrites. Completion requires the remaining original color domains and render combinations. | `FRONT-01`, `COLOR-03` |
| `FRAME-02` | P0 | **Partial** | Four post-transform reference slots retain shared GPU buffer leases and are replaced only after physical reconstruction/composition. Hidden and reference-only frames execute without presentation; re-serialized, dual-oracle Modular fixtures exercise slot 3. Mixed JPEG/Modular and VarDCT/DC references execute with initial admission retry, explicit dependency backpressure, and cancellation/lifetime tests. Pre-transform references are never mislabeled as RGB; consuming that domain as a post-transform background is rejected. Every supported integer extra is now retained beside RGB in one accounted allocation, with validated plane offsets and independent reference selectors. Multi-extra admission retry and abandoned distributed/resampled work have actual-GPU lifetime evidence. Explicitly tagged pre-transform surfaces now retain reconstructed components and every extra for GPU patches; inverse color conversion occurs after patch rendering and before ordinary frame blending. Intermediate patch rendering and display conversion reuse the dictionary and committed references; cancellation or final-only switching never commits an intermediate version. Completion requires remaining patch combinations and broader cross-feature/lifetime conformance. | `FRAME-01` |
| `FRAME-03` | P0 | **Partial** | Full-canvas Replace and real crop/blend/reference sequences coalesce into presentations with exact rational timing, timecodes, names and loop counts through blocking/poll/Future APIs. Every physical color/extra/LF node now validates exactly once in the common execution plan, including unused or overwritten LF producers. Four versioned LF slots reuse reconstructed planes across presentations, retain exact byte reservations through the last consumer, and release expired planes before the next admission. LF flags, levels, current producer versions and sample/block extents are checked before execution. Modular or VarDCT LF roots and LF-dependent SkipProgressive frames have dual-oracle GPU evidence. VarDCT handoff preserves restored and frame-upsampled planes, including Gaborish and 2× LF fixtures. Replace releases overwritten outputs before admitting the next producer and preserves the final native integer codes; hidden Modular/VarDCT/DC truncation is rejected before presentation. Gray31 stills with 2/17/129 layers fit the first producer's GPU footprint, count every submission, and release source/GPU reservations on staged cancellation. All-channel composition retains independent planes until final color or scalar native/F32 presentation. Whole and bounded fragmented async output match, with dual-oracle and lifetime evidence. VarDCT DC/AC refinements, including extra channels, and LF previews with independent extra planes preserve presentation timing and committed references across animation and composition. Terminal DC/AC updates follow hidden-producer validation; LF updates wait for every selected color/extra background version, retaining queued planes across predictor-slot expiry. Patch consumers additionally wait for the validated dictionary and render queued LF components before inverse color conversion while preserving the admitted codec body. LF producer patches now complete before publishing prediction slots or queuing previews; unused producers still validate and execute their features, and extras never share the prediction allocation. Hidden-reference corruption, cancellation and final-only switching have actual-adapter evidence. Modular color/native/scalar F32 and VarDCT numeric-extra updates now preserve the same committed-reference and presentation contract. Patch-bearing Modular and VarDCT pass updates now apply the resident dictionary before display conversion, with native prefix color/extra snapshots and cancellation, final-only and later-corruption coverage. Completion requires remaining source color modes, broader LF conformance and broader precision conformance. LF extra images now survive predictor-slot expiry before a late hidden background and use the common blend/pack path without committing references; native scalar composition and cancellation/failure cases verify this lifetime. | `FRAME-02` |
| `FRAME-04` | P1 | **Missing** | Add non-coalesced frame/layer output, skip/progressive controls, and bounded seek/restart semantics without losing required references. | `FRAME-03`, `CONT-03` |
| `FRAME-05` | P1 | **Partial** | Extend the implemented Modular animation encoder to VarDCT/mixed frames, arbitrary extra-channel blends, hidden/reference frames, names, previews, and frame indexing. `djxl` verifies composed output and timing. | encoder mode items, `CONT-03` |

### J. Color, HDR, extra channels, input, and output

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `COLOR-01` | P0 | **Partial** | The generic GPU image-output path executes standard/custom RGB primary matrices and explicit Bradford/absolute white-point conversion; parameterized Gamma/DCI; source Linear/sRGB/BT.709/PQ/HLG EOTFs; target Linear/sRGB/BT.709/PQ/HLG/BT.2020 OETFs; and BT.601/709/2020 NCL/2020 constant-luminance YCbCr. Without image intensity, PQ is explicitly normalized to `1.0 = 10,000 nit` and HLG is scene-linear; image-owned conversion supplies display-relative PQ and the HLG OOTF, and unsupported or incomplete color metadata returns a typed error before submission. Actual-adapter scalar-oracle tests cover every new path. The decoder inventory now admits D65 BT.709/BT.2020/Display-P3 with Linear/sRGB/BT.709 and gray through both codecs, original-domain composition and numeric/color packing. The private RGB tag carries actual primaries and transfer. Primary conversion and display share matrices with one exact D65 white and host F64 multiplication. A 148-stream native corpus and 74 jxl-oxide still comparisons cover integer/F32 RGB/XYB/YCbCr, plus independent f64 output conversion. Both decoders now admit enumerated PQ/HLG with explicit image intensity through original-domain composition and numeric/color packing; the HDR checkpoint below records 56 native streams and 48 independent Rust stills. HDR and ICC now connect through explicit image white under all four intents. All four original enumerated RGB/Gray intents now admit standard/custom whites; 320 declarations retain native original words and GPU composition/progression, with independent requested Bradford/absolute output checks. Bounded original ICC export has exact official/native byte evidence independently of GPU CMS admission; see [the profile contract](ICC_COLOR.md#original-profile-export). Completion still requires remaining ICC/HDR/CMYK policies, broader rendering-intent semantics and full conformance. Never relabel unsupported pixels. | `META-01` |
| `COLOR-02` | P0 | **Partial** | The shared 304-byte image-output uniform now carries image intensity and both primary-luminance/OOTF vectors. Modular and VarDCT supply the same context to RGB/XYB reconstruction, original reference/blend surfaces, LF presentation and numeric/color packing; XYB inverse intensity must agree with the output context. PQ scales absolute light against the declared unit white. HLG executes forward/inverse display OOTFs with the codec threshold and an explicit native-compatible negative-luminance extension. Identity output retains source F32 bits. Fifty-six native streams present 80 images across both codecs, RGB/XYB, four intensities, three primary sets, gray, alpha, progression and bounded input; 48 stills also have independent Rust component evidence. The HDR corpus documents precision before and after nonlinear transfer. HDR↔ICC now connects through explicit image white with native/independent references and all four intents. Explicit tone mapping now resolves minimum light and protected absolute/relative thresholds against requested display white. Explicit target-linear RGB gamut mapping now follows the curve, with protected light taking precedence and native/F64 whole/bounded image evidence. Completion still requires broader profile/display policy and full-range/feature/LF conformance. The generic display/render graph's existing luminance contracts remain distinct. | `COLOR-01` |
| `COLOR-03` | P0 | **Partial** | Full-resolution 1–31-bit integer Modular alpha, depth, selection mask, spot color, CFA, thermal, black and optional planes now execute and can be selected as native unsigned or scalar normalized F32 output. Names, depths and type-specific declarations survive in public metadata; first-alpha color selection handles Gray+alpha, multiple extras and different alpha depths. Six real fixtures compare every plane against exact source codes and two decoders, including transformed multi-group and bounded fragmented input. VarDCT now consumes global extra-channel streams, fuses the first declared alpha into color output, and independently delivers any global extra plane as native unsigned or scalar F32. All 32 planes in seven public fixtures have source-code and dual-oracle evidence. Scalar requests retain full LF/HF/AC validation while omitting color inverse transforms, restoration and color surfaces. Five distributed VarDCT fixtures additionally deliver all 13 extra planes through native/F32 selection, plus color/alpha after complete global/LF/AC assembly. Both modes now reconstruct shifted/resampled integer extras with 2×/4×/8× effective factors, including dimension_shift 1–3. Associated integer alpha now executes in both decoders and post-transform frame composition. Public Unassociated/Preserve/Associated output policies apply after requested color conversion, with a finite 2^-26 alpha floor; numeric planes are unchanged. Fourteen stills and three nine-layer sequences cover independent depths, non-leading alpha, shifted resampling, Squeeze, all blend modes, native/F32 output, orientation and bounded input. Twenty libjxl fixtures cover odd/thin axes, independent color/extra rates, LF/pass boundaries and progressive Squeeze; native/F32 outputs agree across whole and bounded asynchronous input. Standard/custom filtering shares GPU F32 normalization and final packing. Post-transform composition now retains all integer planes with independent blend/reference/alpha selectors; seven nine-layer fixtures cover every plane, two alpha associations, Gray/RGB, distributed groups and shifted resampling. Floating channels now share representation conversion, filtering, alpha, spots and composition. Completion requires reserved/unknown semantics, broader original color domains and conformance. | `MOD-D05`, `META-01` |
| `COLOR-04` | P1 | **Partial** | VarDCT normalizes orientations 1–8 before target chroma subsampling in its final shared word-owned color packer, after restoration/resampling, with explicit unrotated input extent and transposed output extent. Eight non-square multi-group spectral fixtures, six grayscale cases, an oriented odd 4:2:0 JPEG case, and both one-pixel axes have actual-GPU evidence; RGB8 matches Rust jxl and djxl within one code under whole and bounded async input. Grayscale XYB projects linear luminance through the inverse matrix, while recursive Modular DC roots retain three internal XYB channels. Modular now uses the same orientation helpers in direct, ordinary, group-inverse, and frame-inverse output. Twenty-three native fixtures cover all directions, high-depth RGBA, Palette/Squeeze and one-pixel axes with exact source/Rust jxl/djxl color evidence; twelve gray cases exercise every VPI format with byte-identical whole and bounded fragmented async output. Atomic byte ownership handles rotated group boundaries and odd packed-4:2:2 tails. `GpuOutputRequest::with_orientation_policy` now selects Apply or Keep in both producers and the common frame planner. Nine F32 frame sequences verify codestream-coordinate output, timing, mixed coding modes and recursive DC. Default Render now mixes all integer spot planes on the GPU after reference storage and before target color/alpha conversion and packing; Preserve returns base color, and numeric selection retains raw spot data. An explicit private RGB-domain tag keeps unreferenced XYB linear while composition/reference storage uses the validated original encoding. Ten multi-spot stills cover declaration order, extended RGB/solidity, associated alpha, Gray/RGB, thin axes, resampling and distributed transforms; existing nine-layer sequences cover final-only spot presentation. Native/F32 and YUV output, bounded async equality and exact ink-table admission have actual-GPU evidence. Completion still requires the remaining original color domains and full output conformance. | `COLOR-01`, `COLOR-03` |
| `IO-01` | P1 | **Partial** | The generic resident/readback session supports native RGB/BGR/RGBA and every classified pitch-linear color layout, including NV12-family, packed 4:2:2, P010/P012/P016, odd extents, range, chroma siting, BT.601/709/2020 NCL/2020 CL matrices, wide-gamut primaries, and SDR/HDR transfer conversion. Explicit target-linear RGB gamut mapping precedes transfer, association, YUV subsampling and quantization, including ICC-to-RGB output. Explicit full-range Gray/GrayAlpha U8/F32 now projects target-linear luminance after tone/gamut mapping and before target transfer and alpha association. Independent F64 checks cover three primary sets, eight transfers, both storage forms and all orientations; eight native both-codec sources verify nonopaque alpha, bounded input and retained leases. ICC device Gray and numeric Gray retain their own semantics. VarDCT now uses the same color/layout lowering and shader fragment as the render graph with no intermediate RGB storage or extra submission. Thirty integer layout/transfer cases cover 20 color VPI forms plus high-depth YUV, planar layouts, and linear BGRA under whole and bounded fragmented async input; float Rust jxl/djxl references differ by at most one code at 8–12 bits and three at 16 bits on Apple M5. Dedicated Display-P3/BT.2020 djxl requests, gray resampling, recursive DC, JPEG component upsampling, padding/alpha/guard regions, and typed malformed/HDR contracts are covered. Modular orientation is covered for all 30 VPI layouts, exact native 12/16-bit output, and both inverse-stage topologies; color-code tolerance is one and numeric output is exact. Nine additional F32 layout/transfer cases preserve extended RGB values and validate linear-light precision against both CPU oracles, including direct linear djxl output. Modular F32 normalizes supported 1–16-bit sources after inverse transforms with independent alpha and explicit BT.709 primaries. Gray+alpha, multiple extras, arbitrary first-alpha position and mixed integer depths now use selected GPU views; native RGBA rescales alpha independently. Any admitted integer extra plane can return native unsigned or normalized scalar F32 without a color transfer. Shifted/resampled extras first use shared resident normalization and 2×/4×/8× interpolation; native output rounds at the declaration depth after filtering, while F32 preserves fractions. Unresampled still-image native planes retain exact codes and reject unrepresentable signed working samples; F32 retains negative/overshoot normalization. Composed extra output clamps/rounds the normalized result once for native unsigned delivery, or preserves extended F32 values. The VarDCT scalar packer uses a 64-byte uniform and a four-byte status appended to final validation, without color image allocations. Color outputs now select Unassociated/Preserve/Associated alpha after color conversion, including omitted-alpha RGB, while raw numeric planes retain their source semantics. Selected gray/RGB components now use native unsigned or scalar F32 output in both modes; VarDCT uses unquantized frame surfaces and the shared original-encoding packer, including progressive/composed images. HDR↔ICC now uses explicit image-white connections in both directions. Completion requires every Modular/VarDCT source color domain, general multi-plane extra delivery, both-mode legacy numeric layouts and broader HDR output conformance. | `COLOR-01`, both decoders |
| `IO-02` | P1 | **Partial** | Gray/RGB/RGBA buffers now accept packed, planar and split components with bijective RGB/BGR/alpha swizzles, arbitrary integer bit positions, 8/16/24/32-bit storage words and Native/Little/Big byte order. Every 1–31-bit integer depth and IEEE binary16/binary32 retain exact source words. Independent unaligned offsets/pitches are validated and rebound per plane; source and artifact limits both split GPU batches. Independent exact/native references, canonical byte equality, whole/bounded GPU output, mixed-layout animations and admission/cancellation tests cover the contract. Enumerated full-range RGB/Gray metadata now retains standard/custom primaries/white, seven transfers, four intents and explicit image white, with native profile and original-sample evidence. Other floating precisions, embedded ICC, YUV and textures remain; image-domain normalization stays on GPU. | `COLOR-01`, encoder modes |
| `IO-03` | P1 | **Partial** | Same-queue display now converts pitch-linear D65 BT.709/BT.2020/Display-P3, Linear/sRGB/BT.709/BT.2020/PQ/HLG, and BT.2020 NCL/constant-luminance input into explicitly tagged linear-BT.709 textures. SDR U8 BT.709 accepts `Rgba8Unorm`; F32 RGB, wide-gamut and HDR require `Rgba16Float`, preserving negative and greater-than-one values rather than silently clipping. `DisplayTexture::luminance_encoding` distinguishes relative SDR, normalized absolute PQ (`1.0 = 10,000 nit`), and scene-linear HLG before display OOTF. Both generated storage-texture shaders are Naga-validated and actual-GPU scalar-oracle tests read back the float result. Completion requires alpha-association policy, tone/gamut mapping/HLG OOTF, requested encoded-HDR output, and direct surface-format/capability negotiation. | `COLOR-02`, `IO-01` |
| `IO-04` | P2 | **Partial** | Use native shader `f64` when the backend exposes it and an operation benefits from it; otherwise require an explicit precision policy. F64 storage capability must not be confused with JPEG XL conformance or silently widened arithmetic. | capability negotiation |
| `IO-05` | — | **Out of scope** | CUDA/VPI block-linear and block16-linear layouts remain typed unsupported because portable `wgpu` cannot represent their memory contract. All 30 VPI pitch-linear formats remain in scope. | — |

### K. API, scheduling, and resource safety

ICC matrix/TRC checkpoint (`COLOR-01`, `IO-01`, `API-05/06`, `QA-03/06`): bounded original-byte
profile metadata and a reusable resident RGB/Gray F32 conversion primitive now preserve exact
colorants and independent identity/gamma/sampled/parametric curves. GPU execution covers relative
XYZ matrix/TRC semantics, inverse plateaus/gaps, checked plane bindings and explicit program/80-byte
dispatch allocation. The 100-pair corpus checks 176,120 components against independent f64
equations and retains native Little CMS references with documented precision/boundary differences.
See [the contract and evidence](ICC_COLOR.md). All affected feature rows remain **Partial**:
embedded-ICC decoder admission, original/XYB/reference integration, requested ICC outputs, exact
numeric bypass, LUT/MPE/Lab/CMYK, other intents and HDR/unbounded policies are not completed here.

ICC connection/ownership checkpoint: profile↔linear RGB programs now combine exact ICC colorants
and shared f64 CIE/Bradford geometry before GPU lowering. The linear endpoint preserves signed
and above-one components; ICC device curves retain their bounded contract. 100 additional native
and independent connections validate 182,410 components, while the original 121 reference files
remain unchanged. `ColorSpecification::Icc` owns shared original profile bytes/tag metadata;
inventory clones share reconstructed ICC bytes. Explicit Gray color/alpha storage and device-space
validation prevent numeric or YCbCr relabeling. This prepares original/XYB/reference integration;
stock decoder admission, profile output packing, display and per-image program/budget integration
remain open. All affected rows remain **Partial**.

Embedded ICC numeric checkpoint: Modular reconstruction/restoration/LF settings now describe codec
components independently of color output. Unfiltered original Modular numeric samples and independent
extras from supported single-frame paths in both codecs admit embedded ICC without interpreting
profile curves or inventing RGB metadata.
Eight native RGB/Gray original/XYB streams cover exact alpha and original color samples; metadata-only
ICC substitution additionally covers 17/31-bit integer codes and 5/16/24/32-bit floating representations.
Complete/fragmented input and common/standalone engines retain the existing byte-budget contract.
This numeric checkpoint did not establish ICC color surfaces or conversion. All affected rows
remain **Partial**.

Original ICC execution checkpoint: common decoding of original (non-XYB, non-YCbCr) ICC RGB/Gray
now uses explicitly tagged non-color codec components and copies them into their real device
domain. Gray has one color plane; reference blending and extra offsets follow that count. Color
conversion returns both the actual layout and its buffer. Relative matrix/TRC execution supports
enumerated SDR and other RGB/Gray ICC targets, with exact-profile U8/F32 packing, orientation and
alpha association. Same-profile F32 output preserves Modular IEEE words without curve evaluation.
Image-owned programs upload once with retryable exact admission and completion-owned resources.
The eight-stream corpus retains all previous bytes and adds native original VarDCT pixels,
Little CMS references and independent f64 color results; a metadata-only Gray composition case
retains values above one across blending and orientation. Broader ICC XYB/YCbCr conformance,
spot rendering, standalone color admission, CMYK image plumbing, complete profile/range
conformance and HDR/display integration remain open. Later LUT/MPE, intent and enumerated-RGB
checkpoints extend this initial execution scope. This is progress toward the original
full JPEG XL objective; no feature row or completion gate is marked complete by this checkpoint.

ICC YCbCr checkpoint: inverse codec reconstruction now carries either the actual enumerated RGB
encoding or the image-owned ICC device profile. It writes one Gray or three RGB color planes
before reference storage and composition, without selecting a CMS method for reconstruction.
102 Modular sources cover independent JPEG grids, sample precision, filters, resampling and
extras; 44 both-codec original-color sources cover stills and composed sequences. Complete input
and bounded fragmented input are word-identical, and retained outputs survive session release.
Five source cases additionally check linear/sRGB and opposite RGB/Gray ICC targets against
independent scalar intervals propagated from unchanged codec bounds and checked with Little CMS.
Broader ICC XYB conformance, spot rendering, standalone color admission, CMYK image plumbing,
complete profile/range conformance and HDR/display integration remain open. Later checkpoints
cover LUT/MPE, intent and enumerated-RGB execution. All affected rows and completion gates
remain **Partial**.

ICC XYB checkpoint (corrected by the September 14 reference-validity audit): inverse opsin
reconstructs unbounded linear D65 BT.709, and requested output selects only the necessary ICC
program. Four native stills and seven LF/patch substitutions retain their independent evidence.
The four additive sequences formerly cited here are invalid reference inputs, as detailed in
[the F.2 audit](ICC_COLOR.md#xyb-reference-validity); their old pixel calculations do not establish
conformance. Shared GPU programs, exact admission, completion ownership and independent numeric
selection retain their separate unit coverage. Wider legal combinations, full methods/intents,
HDR and all other completion gates remain **Partial**.

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `API-01` | P1 | **Partial** | One capability query must report exact decode, encode, color, output, precision, memory, workgroup, and platform limits. No versioned aliases or compatibility shims are required; breaking APIs should model current semantics directly. | all feature rows |
| `API-02` | P1 | **Partial** | Keep native blocking plus runtime-neutral `Future`/`Poll` completion for stills, animation, progressive delivery, encode, decode, display, and readback. Browser paths may reject blocking waits but not require a named runtime. | feature state machines |
| `API-03` | P1 | **Partial** | `GpuOutputRequest::with_progressive_output` and native/poll/async `next_update` expose validated, immutable pass images with original physical frame IDs, typed complete LF levels or completed/total coefficient or Modular passes, intended downsampling and explicit presentation finality. A pending presentation and its updates share one logical frame slot; only the final image advances frame index/time, while every packed output retains separate byte ownership. VarDCT stills, animations and composed presentations, including extra channels and deferred HF descriptors, have actual-GPU DC/AC pass-oracle, corruption, cancellation and budget-admission evidence; direct and composed LF-dependent presentations with extra channels also expose LF images after their exact background references validate, with queued and submitted cancellation, final-only switching and LF1 scalar-composition oracle coverage; generic session tests cover metadata, ordering and final-only engine adaptation. Modular global/LF and residual-pass images now resume on demand with copied inverse arenas, independent completed-stream validation and pre-admitted output/scratch leases; integer/float/extra, numeric F32/F64, whole/40-byte windows, corruption, cancellation and exact-budget retry have native/GPU evidence. Composed Modular color/native/scalar F32 and VarDCT numeric extras now share the same validated snapshots, reference versions, immutable packing and final-only convergence. Completion requires broader LF conformance, broader Modular transform/header combinations, composed precision coverage, progressive/ROI targets, selective dirty regions and incomplete-frame resumable input. LF intermediates now retain alpha and independent normalized extras, use native/scalar F32 output, and preserve final-only bytes across queued, rendered, blended and packed cancellation boundaries. | `FRONT-03`, progressive mode rows |
| `API-04` | P2 | **Partial** | Coalesce multiple small stills/frames into shared codec command buffers and bounded maps. Distinguish codec batching from host-thread fan-out and the existing aggregate readback. | stable feature graph |
| `API-05` | P1 | **Partial** | Preserve one shared byte budget across encode/decode/display/readback; account scratch, pooled physical bytes, output clones, reference slots, ICC/box buffers, and abandoned futures. Device loss and cancellation must release exactly once. | all allocations |
| `API-06` | P1 | **Partial** | Keep all Rust/WGSL ABI records `repr(C)` + `bytemuck::Pod` where valid, with compile-time sizes, explicit padding/alignment, checked dynamic offsets, bounded u32 addressing, and no string-content shader tests. Validate shaders by parse/compile/execute and semantic outputs. | every shader change |
| `API-07` | P1 | **Partial** | Define direct-map exclusivity by the underlying resource identity/accounting owner, keep it instance-scoped, and reject conflicting mappings. Document raw-handle escape as outside lease accounting. | readback API |
| `API-08` | P1 | **Partial** | Keep public failures as structured `thiserror` enums with source chaining and distinct malformed-input, unsupported-feature, resource-limit, capability, device, cancellation, and retryable-pressure variants. Tests assert typed variants and fields, not display strings. | every public path |

### L. Encoder input decisions and product controls

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `ENC-01` | P1 | **Partial** | Lossless Modular accepts all 1–31-bit integer and IEEE binary16/binary32 Gray/RGB/RGBA sources with exact original-word conformance and unassociated alpha. Packed/planar/split layouts, swizzles, word byte order and integer bit positions are read directly on GPU. Floating samples retain special values and payloads through shared/local trees, animations with per-frame layout changes and bounded streaming. Enumerated RGB/Gray source color, four intents and explicit image white survive still and Replace animation serialization with independent native profile/pixel checks. Other floating precisions, alpha association, arbitrary extra-channel planes, embedded ICC, YUV and textures remain; all input normalization must stay on GPU. | `IO-02`, `COLOR-03` |
| `ENC-02` | P1 | **Missing** | Add explicit lossless/lossy, distance/quality, effort, decoding-speed, resampling, progressive, metadata-preservation, deterministic, and memory/latency policy controls with typed invalid combinations. | encoder mode rows |
| `ENC-03` | P2 | **Missing** | Implement automatic Modular versus VarDCT selection and per-feature decisions. Evidence must show the selected path and compare it with forced alternatives. | complete baseline encoders |
| `ENC-04` | P1 | **Missing** | Encode all extra channels with independent distance, upsampling, dimensional shift, and blend contracts; preserve invisible color according to policy. | `ENC-01`, `COLOR-03` |
| `ENC-05` | P1 | **Missing** | Implement GPU JPEG entropy/coefficient ingestion and lossless JPEG recompression, optional chroma-from-luma, metadata preservation, and `jbrd` emission without a CPU JPEG coefficient or pixel codec. | `CONT-05`, `VDCT-E02` |

### M. Conformance, robustness, and performance evidence

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `QA-01` | P0 | **Partial** | Import or generate positive and negative fixtures for every roadmap feature before advertising it. The corpus manifest records expected profile, dimensions, source, oracle, precision, and stock/future status. | each feature |
| `QA-02` | P0 | **Partial** | All 40 descriptors across 27 unique inputs in pinned libjxl/conformance revision b1d0f990b03e57bf6d137c365cd5dc8b470b9191 now have original per-channel pixel-bound coverage. The common target compares 37 descriptors / 25 inputs, including 48-frame and 36-frame animations, large progressive output, noise, upsampling, patches, signed F32, alpha and original ICC components. Identity checks bind three more descriptors to the existing 60-frame spline and five-channel CMYK GPU families. Whole and bounded fragmented output agree exactly; retained frames outlive sessions and release reservations. Input, descriptor, profile, reference hashes and metadata/shape are checked, including distinct alternate NPYs. Zero-error F32 requires exact IEEE words; numeric ICC does not imply an accepted color profile. A separate GPU byte-output target now matches all three official original JPEGs exactly. Original ICC export now matches every declared original profile object (19 descriptors), including generated profiles; all 27 inputs and 761 generated declarations also match the public native original-profile API byte-for-byte. Remaining ISO/IEC 18181-3 acceptance evidence stays open; these checks do not establish full decoder conformance. | full decode path |
| `QA-03` | P0 | **Partial** | Differentially compare decode with libjxl and encode with both libjxl/`djxl` and independent Rust oracles (`jxl`, or retained jxl-oxide working planes for exact high-depth integer/IEEE words). Cover cross-products of tools, not only isolated single-feature fixtures. | each feature |
| `QA-04` | P1 | **Partial** | Fuzz raw/container parsing, entropy metadata, MA trees, transforms, coefficients, frame references, ICC/brob/jbrd, cancellation, and resource limits. Add truncation and corruption at every bounded stream layer; no panic, hang, OOB, or partial authoritative result. | parsers and engines |
| `QA-05` | P1 | **Missing** | Build lossy encode evaluation with Butteraugli plus PSNR/SSIM and artifact-focused cases, bitrate matching, deterministic source hashes, and images from tiny through 16K across square, portrait, panorama, and extreme one-pixel axes. | `VDCT-E02` |
| `QA-06` | P1 | **Partial** | Expand animation, HDR/wide-gamut, ICC/CMYK, extra-channel, progressive, JPEG-reconstruction, metadata, and mixed Modular/VarDCT corpora. Include 1/255/256/257/2048 boundaries and 16K where memory permits. | feature rows |
| `PERF-01` | P2 | **Partial** | Add in-process libjxl decode/encode baselines. Keep external-process `djxl`/`cjxl` results labelled separately; they are not fair warm-library comparisons. | stable correctness |
| `PERF-02` | P2 | **Partial** | Record GPU timestamps, host submit/wait/map time, pipeline compilation, upload/readback, peak active/pooled/driver memory, submissions, maps, and output hashes for isolated, warm, concurrent, batched, animation, display, and CPU-readback paths. | `API-04`, stable correctness |
| `PERF-03` | P2 | **Partial** | Autotune validated 32/64/128/256 workgroups by adapter/profile/output/workload, including encoder kernels. Reject choices exceeding workgroup storage or device limits and persist profiles with typed version checks. | each tunable kernel |
| `PERF-04` | P2 | **Missing** | Establish release gates for CPU-readback parity/wins, GPU-resident/display wins, concurrent throughput, encode speed, quality/size, and peak memory on multiple native adapters plus browser WebGPU. No universal win is claimed from one Apple M5 snapshot. | `PERF-01`, `PERF-02` |

### Analytic profile progress

The shared RGB model now carries custom CIE primaries, reference white and validated gamma.
Both codecs admit nonsingular enumerated SDR RGB/gray with D65/E/DCI/custom whites,
Linear/sRGB/BT.709/Gamma/DCI transfer and all four original enumerated rendering intents.
GPU output offers explicit Bradford or absolute-XYZ conversion; display accepts the new profiles.
The shared image/display uniforms are 240/160 bytes. Original reconstruction and general CMS
curves remain explicit, including the native Gamma/DCI black floor and calibrated XYB RGB source.

Eighty additional streams retain all previous fixtures and error bounds, covering both modes,
RGB/XYB/YCbCr, independent alpha, integer/F32, gray, stills and composition. Forty independent
stills and pre-OETF references protect non-invertible Gamma/DCI output. These are milestones in
`COLOR-01`, `COLOR-02`, `IO-01`, `IO-03` and `QA-06`; each remains **Partial**. Full ICC matrix/TRC
and LUT profiles, CMYK, broader rendering policies, HDR luminance/OOTF, wide-gamut feature/LF
cross-products, encoder completion and all existing conformance gates remain required.

The [original-intent corpus](CONFORMANCE_CORPUS.md#enumerated-original-rendering-intents) adds
320 test-time declarations from the unchanged 80 analytic sources. Native original output is
bit-identical across intents; GPU reconstruction, composition, immutable pass output, requested
Bradford/absolute conversion and 156 exact-word metadata variants retain the existing bounds.
Original intent does not override a requested output policy. `COLOR-01` and `QA-06` stay Partial.

### Original SDR metadata and composition evidence

The `original_color` corpus adds 148 independently referenced streams (74 stills and 74 six-frame
sequences) for both codecs, RGB/XYB/YCbCr, three D65 RGB primaries and three SDR transfers, gray,
12-bit/8-bit/F32 color and independent 10-bit alpha. Native reference generation explicitly separates
36 RGB component sources from their YCbCr header recipes; metadata validation verifies every actual
stream before comparison. Whole and fragmented progressive updates retain immutable images and
match final-only output. Original RGBA F32, linear BT.709 F32, explicit sRGB8, original native12 and
scalar color/alpha use independent references; source reconstruction bounds propagate through an
f64 chromaticity/transfer oracle for converted output. Native composition packing now carries the
original transfer in an 80-byte uniform and never reapplies an OETF to original Linear samples.

Color metadata alone no longer sends native or numeric Modular samples through F32. A regression
test re-encodes only the color declarations of three exact-word integer fixtures across all nine
RGB profiles: 27 variants preserve 17/31-bit color and independently declared 5/24/31-bit alpha
exactly through native RGB and each scalar selection, under whole and bounded fragmented input.
Four additional analytic RGB profiles add 12 variants (39 total), including custom white/primary
coordinates and Gamma/DCI declarations with the same byte-identical frame and exact-word checks.

jxl-oxide independently checks all 74 stills; XYB uses unbounded linear BT.709 plus independent f64
conversion to avoid that decoder's CMS clipping. Its extra-channel source-selector bug prevents these
sequence headers from parsing; all 74 sequences retain their original native libjxl references.
BT.709's negative extension explicitly follows libjxl and jxl-oxide's linear toe, with the differing
Rust jxl reflected curve documented. This evidence does not complete color management: ICC and complete rendering intents,
HDR luminance mapping, wide-gamut spot/patch/spline/LF combinations and complete
ISO precision/conformance remain open. See the
[generator and precision notes](../crates/jxl_wgpu_decode/test-data/original_color_generator/README.md).

## Critical implementation order

The order below minimizes temporary formats and unlocks the largest amount of conformance work per
stage. Performance work continues only where it does not freeze an incomplete packet contract.

1. **Unify the frontend and finish common entropy**: `FRONT-01/02`, `ENT-D01/02`.
2. **Make VarDCT real rather than zero-AC**: `VDCT-D01/02/03/04/05`, then `VDCT-E01/02`.
3. **Complete the render graph**: `RENDER-01/02`, `COLOR-01/02`, and both-mode `IO-01`.
4. **Complete frame semantics**: `COLOR-03`, `FRAME-01/02/03`, then patches/splines/noise.
5. **Complete Modular**: `MOD-D01..05`, followed by `MOD-E01..04`.
6. **Complete progressive and streaming behavior**: `FRONT-03`, `VDCT-D06/07`, `META-03`,
   `API-03`.
7. **Complete containers and JPEG reconstruction**: `CONT-02..07`, `ENC-05`.
8. **Raise encoder quality and breadth**: `VDCT-E03..06`, `ENC-01..04`, `RENDER-06`.
9. **Close conformance and performance gates**: `QA-01..06`, `PERF-01..04` across native and
   browser adapters.

The common frame plan now executes frame-local producers and real post-transform GPU reference,
crop and blend composition, including mixed coding modes, recursive DC, Gray+alpha and independent
integer alpha depths. Full-resolution Modular extra planes now have independent output selection.
The shared Modular substream executor now has public VarDCT global-extra/alpha and cursor
evidence, including bounded continuation and individual native/F32 scalar delivery. The AC executor
now has GPU-verified unaligned Prefix/ANS continuation into a Modular substream, and both coding
modes share global/LF/pass channel ownership. Five distributed fixtures now prove public GPU
assembly, color/alpha and every native/F32 extra plane, including empty globals and LF Squeeze.
Shifted/resampled integer extras now use common resident normalization and 5×5 interpolation.
Associated integer alpha now executes through both color producers and post-transform composition;
references preserve their original association until requested output conversion.
All integer extra channels now participate in composition through independently addressed planar
surfaces, including distributed and shifted/resampled sources. Seven nine-layer sequences exercise
independent blend/reference/alpha selectors and native/F32 delivery.
Integer spot presentation now shares the all-channel boundary and final GPU packer across both coding modes.
Floating sources now use the same decoded F32 surfaces for color, extras, alpha and spots.
Pre-transform references now retain codec components after ordered patch, upsampling and noise
completion. Mixed coding-mode and subsampled VarDCT YCbCr patches now have independent conformance
evidence. GPU spline parsing, bounded geometry and ordered rendering now precede resampling/noise
and share completed caches with LF/pass updates. Independent plane layouts now support unequal
color/extra resampling with one full filter per channel. Remaining integration includes broader
original color domains, Modular YCbCr spline/LF-producer interactions and post-transform composition. Modular YCbCr
now reconstructs independently sized integer/floating components through the common GPU renderer;
474 native streams cover all sampling selectors, global/LF/pass ownership, resampling,
restoration, extras and immutable pass output. Transform evidence includes 236 global cases,
140 local cases and independent source/target geometry for all 664 nonempty local substreams.
A separate 494-case patch-reference corpus now includes 218 Modular YCbCr and mixed-reference
combinations; direct sRGB, independent linear conversion, numeric extras, all-slot overwrites and
whole/fragmented immutable progression have actual-adapter evidence.
Empty Squeeze residuals retain their channel positions and shifts; RCT on empty residuals emits
no GPU work. The inverse planner owns job counts and allocation invariants; the obsolete duplicate
transform-admission gate and unreachable unsupported-transform API are removed. Short entropy
uploads use their actual size for byte admission. Initial admission is retryable; resource failure after a staged presentation begins is
terminal. Independent Replace validates overwritten color/extra producers, releases each hidden
output before the next admission, and presents the final native buffer. LF sequences now visit
every physical node once, including unused or overwritten versions, instead of reconstructing a
dependency closure for each color producer. Four slots retain exact producer versions and actual
plane byte reservations across presentations through their last consumer. A partition of the
exclusive producer reservation transfers plane ownership without releasing/re-admitting bytes.
Tests cover unused LF corruption, shared LF versions, exact retained bytes, VarDCT roots
with Gaborish and 2× frame upsampling,
LF-dependent SkipProgressive output, hidden entropy truncation, Gray31 under one producer's
budget, submission totals, retry before initial admission, terminal late pressure and cancellation.
Modular LF restoration/resampling and staged global/LF-group extra channels in LF consumers
now have independent conformance coverage. Remaining `FRAME-01/02`, `FRONT-01` and
intermediate progressive-output gates still apply, including broader reconstruction color domains.

The 48 ICC XYB alpha sequences are now negative reference fixtures. Their earlier classification
as positive composition evidence was incorrect; see [the reference-validity correction](ICC_COLOR.md#xyb-reference-validity).
Independent physical-layer and blend calculations remain diagnostic records. The shared-program
allocation/cancellation tests remain useful unit evidence, separate from codestream conformance.
All legal colour/frame requirements and the full JPEG XL goal remain open.

### Structural refactoring gate

After the first bounded DCT8 nonzero-AC milestone, feature additions pause until the five largest
implementation units are split by responsibility. The layout uses Rust 2018+ `name.rs` plus
`name/` submodules and never introduces `mod.rs`. Internal visibility is minimized; preserving an
awkward public boundary is not a goal, so workspace call sites are migrated when a clearer API
requires a breaking change. Each split must independently pass formatting, warning-free
`cargo clippy --all-targets`, and the complete test suite before the next feature slice.

- `lossless_modular.rs`: types, grid, memory, dispatch, streaming, serializer, tests.
- `vardct_engine.rs`: types, pipeline, window planning, source, restoration, execution, tests.
- `wgpu_engine.rs`: types, pipeline, session, lifetime, execution, tests.
- `vardct_encoder.rs`: types, entropy, AC fragments, bitstream, dispatch, tests; bounded/scalable shaders share co-located colour/DCT/quantizer primitives.
- `scheduler.rs`: validation, pipeline, color/filter/blend/I/O nodes, tests.

The structural gate is complete. All five implementation units now use responsibility modules with
explicit production imports and scoped internal visibility; no `mod.rs` or source-inclusion shim is
used. Feature work may resume from the next roadmap item after the complete workspace validation
gate passes.

The coding-mode selector, shared typed entropy-stream ABI, bounded Modular/VarDCT-AC/staged-LF/HF
stream resume, nonzero-AC mixed/multi-group decode,
local per-substream MA-tree frame execution, non-default LF dequantization/correlation, bounded
resident Gaborish/EPF restoration chain, logical/physical TOC-order normalization, and the
multi-LF-group tiled-DCT8 encoder are implemented. Both bounded and tiled DCT8 now emit
real GPU-generated AC coefficients using a simple one-cluster prefix policy, with checked
fragment concatenation across single and multiple AC/LF groups.
Incremental transport, bounded
image/frame/TOC inventory, public event-fed decode, shared-span metadata/upload paths, and
source-lifetime admission now connect through the same engine boundary without whole-input
assembly. Recursive progressive-DC HF-global/AC resume and parametric custom matrices now use that
engine boundary. The immediate P0 gap remains the common frontend and entropy work in `FRONT-01`
and `ENT-D01/02`: lower the remaining side-image and frame consumers into one bounded
backend-neutral execution graph. The remaining `VDCT-E02` work must extend the all-strategy AC
path to custom orders/matrices, adaptive entropy and progression, while `VDCT-E04` must select
image-wide strategies by content and effort;
broader render-graph composition remains required beside those format-completeness gates.

## Claims explicitly prohibited before their gates pass

- Executing all 27 inverse-transform kernels is not full VarDCT support while nonzero AC entropy,
  quant fields, mixed strategies, and passes remain missing.
- Executing all MA predictors and resident transforms is not full Modular support while
  any legal transform/entropy/render combination or progressive delivery remains missing.
- Emitting one standards-compatible fixed Gradient stream or one single-cluster DCT8 AC stream is
  not a production-complete encoder; broad normative output ability and quality/rate-control search
  are separate gates.
- Parsing `jxlc`/`jxlp` is not complete container support without metadata, `brob`, `jxli`, streaming,
  unknown-box policy, and JPEG reconstruction.
- Having blend kernels and animation metadata types is not decoder animation support without GPU
  reference slots, hidden frames, dependency ordering, coalescing, and lifetime accounting.
- A CPU oracle, external wrapper, generated fixture, or CPU fallback cannot expand the production
  GPU capability claim.

## Matrix/TRC rendering-intent checkpoint

All four matrix/TRC intents now share one f64 PCS affine program: media-relative conversion,
fully adapted absolute media white and v4 perceptual/saturation black compensation. Linear RGB
is an unbounded ideal adapted v4 endpoint; selected LUT/MPE precedence remains authoritative.
The existing 80-byte program header stores offsets in the matrix rows' fourth lanes with no
additional allocation or CPU pixel work. All 2,704 profile pairs and 1,040 linear connections
have native/independent references. Public decoding adds 192 original/XYB connections and 768
presentations, preserving codec precision, alpha, transport and completion ownership.
All 1,015 earlier reference files remain unchanged. See [the ICC contract](ICC_COLOR.md).
`COLOR-01/02`, `IO-01`, `QA-03/06` and the full JPEG XL goal remain **Partial**; LUT/MPE/Lab/CMYK,
other profile policies, gamut/HDR mapping and the remaining conformance requirements are open.

## Primary references

- [JPEG XL format overview](https://github.com/libjxl/libjxl/blob/main/doc/format_overview.md)
- [libjxl encoder API and feature settings](https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h)
- [libjxl codestream metadata and extra-channel types](https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/codestream_header.h)
- [libjxl encoder effort/tool selection](https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md)
- [libjxl codec architecture overview](https://github.com/libjxl/libjxl/blob/main/doc/xl_overview.md)

The ISO text and official conformance material remain normative. The public libjxl sources above
are implementation and audit references, not permission to substitute its CPU codec in production.

ICC MPE execution checkpoint: selected floating-point processing elements now use one ordered
GPU stage program shared with matrix/TRC and linear RGB connections. `matf`, segmented `cvst`,
`clut`, `bACS`/`eACS`, physical Lab/XYZ conversion, all four tag selections and mixed absolute-PCS
connections are implemented. Metadata bounds and whole-element sharing precede allocation;
exact identity elimination and bit-ordered breakpoint selection preserve subnormal branch
ownership. The native/f64 MPE corpus adds 135,864 resident components and 22,032 original-decoder
components, with 192 requested-output presentations. See [ICC color processing](ICC_COLOR.md).
`COLOR-01/02`, `IO-01`, `QA-03/06` and the full goal remain **Partial**: legacy LUT methods,
complete CMYK image plumbing, arbitrary float-range MPE conditioning/overflow, hybrid-profile
CMM policy conformance, remaining embedded MPE/XYB coverage and HDR/display mapping remain open.

MPE curve range checkpoint: the GPU retains separate significands and exponents across curve
intermediates, avoiding overflow in powers, exponential multipliers and sampled interval widths.
Logarithmic increments and differences retain their contribution before a large outer scale;
computed subnormal inputs survive normalization. Stored subnormals are rejected, empty sampled
segments retain the next segment's implicit endpoint, and logarithmic metadata endpoints avoid
overflowing powers. Thirteen independent scalar profiles cover 10,959 components across F32 extrema
and boundaries (65,754 GPU checks across directions/variants). Earlier native/scalar references
are unchanged. The feature rows remain **Partial**: entire-segment formula validation, arbitrary
conditioning and full-range matrix/CLUT/Lab arithmetic still require further work.

MPE affine-power checkpoint: two scaled terms retain the base's product/sum remainder before
nonlinear amplification, with integer arithmetic independent of GPU `fma` contraction. Near-unit
logarithms and exponential increments preserve final offset cancellation, including tiny powers
of far-from-unit bases. Metadata-only implicit sampled endpoints use compensated f64 sums and
`log1p`/`expm1`. Twenty-two additional independent profiles contribute 72,864 scalar components
and 437,184 directional/kernel GPU checks; all 160 earlier MPE reference files remain unchanged.
No feature row is completed by this checkpoint. Complete formula-range validation, arbitrary
conditioning, remaining ICC methods and all broader full JPEG XL gates remain required.

Legacy ICC LUT checkpoint: `mft1`, `mft2`, `mAB` and `mBA` lower to the shared ordered GPU
program with explicit curve/matrix clipping, normalized XYZ/legacy-Lab boundaries and typed
CLUT interpolation. Metadata preflight validates counts, named offsets, complete/suffix curve
sharing, storage overlap and stage combinations before allocation. Forty-one v2/v4 profiles
provide 202,436 independent/native components and 607,308 GPU directional/kernel comparisons.
Twenty-four compressed-ICC RGB/Gray Modular/VarDCT streams add 96 LUT-to-LUT references and
384 decoded presentations, with propagated codec precision, exact alpha, bounded fragmented
input and completion-owned memory. All earlier reference data remain unchanged.
The source-black checkpoint below extends automatic v2 connections. `COLOR-01/02`, `IO-01`,
`QA-03/06` and the full goal remain **Partial**: CMYK image plumbing, other profile classes,
broader MPE/XYB and HDR/gamut policy, full-range arithmetic and the remaining conformance/encoder
gates are still required. See [ICC processing](ICC_COLOR.md).

V2 LUT source-black checkpoint: the selected source program and normalized darker-colorant
endpoint execute in a separate one-invocation GPU metadata pass. Per-dispatch storage holds
the derived PCS affine and status while curves/CLUTs retain shared immutable storage.
Nonfinite connection coefficients suppress image writes and return typed precision failure
before authoritative output; no CPU pixels or CPU LUT evaluation are involved. Exact memory
admission includes the 320-byte dispatch and optional four-byte map. Concurrent reuse, failed
wait/poll and cancellation release their accounted resources.

Twenty v2 LUT profiles add 120 native black references and 318,240 independently bounded/native
color components, checked 954,720 times across three GPU kernels with 576 validated preparations.
Twenty-four embedded-profile Modular/VarDCT streams add 384 presentations and 117,504 color
checks, exact alpha and bounded fragmented input. All 718 new files reproduce twice exactly;
all 581 prior LUT files also regenerate unchanged. CMY/device-Lab endpoint metadata is covered,
with native image conformance for those and other uncommon spaces still open. No feature row
or full JPEG XL completion gate is closed by this checkpoint.

Enumerated RGB-to-ICC checkpoint: selected profile programs now connect to `RgbColorEncoding`
through an explicit GPU transfer stage and exact CIE/Bradford geometry. Linear endpoints omit
the transfer stage. The common decoder routes requested ICC output through these programs for
original RGB/Gray, YCbCr, direct XYB and original-domain reference composition. Presentations
are selected by actual source encoding, deduplicating originally linear color. Programs and
intermediates retain exact admission, retry and completion ownership; alpha stays outside CMS.

All 228 existing enumerated streams are paired with nine matrix/TRC, LUT and identity-MPE targets
under four intents. The independently derived/native 4,049,280 reference components are checked
16,197,120 times in 9,120 presentations across layouts and whole/fragmented input. Held progressive
output agrees with final-only output. The independent oracle is a normal shared test-support
module; no path inclusion or additional warning suppression is introduced. Existing source
fixtures and profile bytes remain unchanged. See the [RGB-to-ICC recipe](../crates/jxl_wgpu_decode/test-data/rgb_icc_generator/README.md).
This progresses `COLOR-01/02`, `IO-01`, `QA-03/06`; those rows and the full goal remain **Partial**.
Complete CMYK/HDR plumbing, wider profile/method/range conformance and the
remaining codestream/container/encoder requirements are still required.

ICC spot presentation checkpoint: declaration-order ink rendering now precedes the selected ICC
connection in the actual original or linear source domain, after reference storage. A dedicated
GPU copy retains the actual one/three color planes and independent extras, with offsets derived
from their layouts. Its storage and ink table share exact admission and completion ownership
with ICC packing; Preserve and numeric requests omit the stage. Same-profile output retains
extended rendered values. Gray ICC connections consume their single device component.

The spot corpus has 28 supported streams and four negative reference cases. Positive checks
compare 1,002,456 colour components in 2,808 final presentations and 71,604 numeric extras,
with independent intervals and retained output ownership. The four ICC XYB sequences were
incorrectly described as conformance evidence; they now require early rejection. Their bytes
and diagnostic calculations remain unchanged. See [the recipe](../crates/jxl_wgpu_decode/test-data/icc_spots_generator/README.md).
`COLOR-01/02/03/04`, `IO-01`, `QA-03/06` and the full JPEG XL goal remain **Partial**.

Original CMYK checkpoint: an explicit CMYK surface owns its profile and Black extra index while
retaining three complemented CMY planes. ICC presentation borrows the independently composed
Black plane and normalizes all four components on GPU. Source-domain spot rendering precedes this
connection. YCbCr inversion now has an explicit three-component packing contract, and VarDCT
uses the shared legal integer/floating precision validation for YCbCr as well as XYB/RGB. ICC image sample
normalization leaves metadata black probes unchanged; the storage ABI is 320 bytes.

The unchanged official `cmyk_layers` reference is checked across all five channels with its
original RMSE/peak bounds and whole/bounded input. Eighteen native/assembled three-frame streams
exercise RGB/Gray output, mft1/mft2/A/B, XYZ/Lab PCS, four intents, both original codecs, YCbCr,
Black in extra slot 0 or 2, distinct color/Black blend references and independent F32 alpha/spot.
Native and F64 ICC oracles provide 132,192 components, checked 528,768 times in 1,728 GPU
presentations; scalar original components and retained memory ownership are checked separately.
[Recipe and precision](../crates/jxl_wgpu_decode/test-data/cmyk_generator/README.md).
`COLOR-01/02/03/04`, `IO-01`, `QA-03/06` and the full JPEG XL goal remain **Partial**. Requested CMYK
layouts, CMYK-suggested XYB output and numeric selection conformance, wider sampling/profile/range conformance, HDR and
all remaining codestream/container/encoder gates still require work.

ICC device output checkpoint: `ColorModel::IccDevice` and explicit `Channel::Device(index)` /
`Channel::Alpha` packing replace the four-component assumption for profile-owned output.
U8/F32 planar/interleaved layouts support up to fifteen profile components plus alpha, arbitrary
component order, all orientations and padded byte addressing. CMYK output uses unit ink amounts;
same-profile requests preserve original F32 values with only the declared sample convention
conversion. The 320-byte packer uniform and all intermediate/device resources participate in
exact admission, retry, concurrent completion and cancellation ownership.

Forty-two unchanged JPEG XL sources provide 403,920 independent F64/native output references.
CMYK covers each 1/2/3/4/5/15-component target through Modular, VarDCT and YCbCr, all four intents
and both spot policies. Public decoder checks compare 3,231,360 converted components in 4,224
presentations and 323,136 same-profile components in 624 presentations. Twenty enumerated sources
add 800 presentations / 1,417,248 independently bounded components across all reconstruction
modes. The resident packer checks 1,152 guarded dispatches / 931,040 bytes, including alpha,
orientation, component permutations and unaligned pitches. Existing source/profile/reference
fixtures remain unchanged. See [the device output recipe](../crates/jxl_wgpu_decode/test-data/device_output_generator/README.md).

This completes the requested CMYK layout gap described above. `COLOR-01/02/03/04`, `IO-01`,
`QA-03/06` and the full JPEG XL goal remain **Partial**: CMYK-suggested XYB output and numeric selection conformance,
uncommon device-space image conformance, broader profile/range/conditioning and sampling,
HDR, container, encoder and the other roadmap gates still require work.

Reference-colour validity checkpoint: the shared header parser and public frame planner now
reject forbidden ICC XYB post-transform reference storage, with a typed slot error. All 56
previously accepted invalid streams remain unchanged as negative fixtures. Whole input and
1-/43-byte fragments reject before frame/section publication and release retained input. A
192-case header matrix preserves legal pre-transform references, ordinary output, LF and preview
behaviour. The corpus audit separates 56 negative inputs from 2,516 accepted inventories;
acceptance alone is not a pixel-conformance claim. No completion gate closes at this checkpoint.
[Specification basis and corrected evidence](ICC_COLOR.md#xyb-reference-validity).

CMYK-suggested XYB checkpoint: numeric color reconstruction now runs in the ICC output stage,
using the suggested original profile's header intent. Complete device storage keeps generated K
separate from the encoded Black extra; numeric CMY complements only the selected generated color
component. Extra selections do not request an inverse ICC method. RGB/Gray share this boundary.
Independent sequences now preserve the caller's retained-frame window instead of copying a
single physical VarDCT producer's one-slot capacity; byte admission remains per allocation.

Twelve legal three-frame full-canvas F32 sequences cover six unchanged CMYK LUT profiles,
both codecs, both PCS domains and Black at extra index 0 or 2. Native decoding verifies the
linear basis and exact extras. The 88,128 independent/native ICC components are compared
352,512 times in 576 GPU presentations, and numeric selection adds 66,096 sample comparisons.
Four patched LF substitutions also check Alpha/Depth selection across whole and bounded input.
Exact device-plus-extra storage, numeric uniform/program admission, retry and cancellation are
covered. [Recipe and precision](../crates/jxl_wgpu_decode/test-data/cmyk_xyb_generator/README.md).
`COLOR-01/02/03/04`, `IO-01`, `API-05`, `QA-03/06` and the full JPEG XL goal remain **Partial**:
broader source precision/sampling, legal ICC crop/blend combinations, profile methods/ranges,
HDR and every remaining roadmap gate still require completion.

Extended extra sampling checkpoint: shared header validation and both producers now admit
effective extra factors 16/32/64. The common resident reconstruction applies 8× followed by
2×/4×/8× with standard/custom weights. A distinct, fully admitted intermediate retains all
first-stage samples until final cropping; integer normalization precedes filtering.

A 139-file corpus supplies 68 Modular images with independently verified source words and two
native VarDCT controls whose presentation scaling preserves every encoded grid and entropy
section. Independent F64 filter intervals, 35 eligible Rust Modular cases, 21 native Modular
8× controls, 16 Rust VarDCT cases and four native VarDCT controls anchor 968 whole/bounded GPU
outputs. Thin-grid limitations of the Rust oracle and libjxl's effective-factor-eight limit are
explicit. Exact memory admission, one-byte pressure/retry, held outputs, cancellation, strided
cropped views and a premature-cropping counterexample cover storage and edge behavior.
[Recipe, arithmetic bounds and oracle eligibility](../crates/jxl_wgpu_decode/test-data/extra_upsampling_generator/README.md).

`META-01/02`, `MOD-D01`, `RENDER-01`, `COLOR-03`, `QA-03/06` and the full JPEG XL goal remain
**Partial**. Broader image-header dimension shifts, independent factors between extras, native
quantization and feature/LF/composition combinations at these extended rates still need evidence;
all remaining codestream/container/encoder and quality gates remain in scope.

### Enumerated HDR intensity and reconstruction checkpoint

`COLOR-01/02` and `IO-01` now connect image intensity to enumerated PQ/HLG in both decoders,
XYB restoration, original-domain reference/blend surfaces, LF presentation and final numeric/color
packing. The shared image uniform is 240 bytes; native composition packing is 96 bytes. No
production pixel processing moves to the CPU. Same-encoding F32 words remain unchanged. The
following checkpoint extends this enumerated HDR support to relative image-white ICC connections.

The [HDR corpus](../crates/jxl_wgpu_decode/test-data/hdr_generator/README.md) contains 56 native
libjxl 0.12.0 streams and 80 presentations. Both coding modes and RGB/XYB cover PQ/HLG at
100/255/1000/4000 nits, D65 BT.709/BT.2020/Display-P3, gray, independent alpha, 257-pixel group
boundaries and six-layer sequences. Whole input, bounded fragmented progression, retained outputs,
final-only equality, original numeric color/alpha and reservation release are checked. All 48
stills also compare against jxl-oxide raw components with independent F64 reconstruction; 1728
GPU transfer triples separately check both directions, primary conversion, near-unity HLG,
identity, black and negative samples.

Apple M5 Metal results give maximum normalized linear error `0.0005264366` against native
libjxl and `0.000047218684` against the independent Rust path. PQ magnifies tiny near-black
reconstruction differences: native original encoded error reaches `0.0181252823`, so acceptance
propagates the declared linear reconstruction interval through the transfer, instead of claiming
a uniform encoded epsilon. The independent Rust linear budget is `1e-4`, native linear uses
`1/1024`, and isolated F64 transfer comparisons use `5e-5`. Native original non-XYB, composition
and alpha retain their separate existing bounds. The corpus recipe documents native CMS black/
highlight policies and the signed-bit HLG reconstruction extension.

This is a bounded HDR integration checkpoint. `COLOR-01/02`, `IO-01/03`, `QA-06` and the full
JPEG XL goal remain **Partial**. Display peak/min-nit and relative-to-max-display policy,
tone/gamut mapping, broader HDR/LF/feature combinations, complete numeric-range conformance and
all remaining decoder/container/encoder/quality gates still require work.

### HDR and ICC image-white connection checkpoint

`COLOR-01/02` and `IO-01` now connect PQ/HLG and ICC through an explicit relative image white:
PCS Y=1 represents `intensity_target` nits. The backend-neutral `DisplayIntensity` validates
finite positive luminance. RGB ICC endpoints retain complete encoding/intensity metadata, and
the shared WGSL transfer executes PQ absolute scaling or the coupled HLG OOTF around the PCS
matrix. The decoder passes this context through reconstruction, composition and final packing.
All four profile rendering intents retain their selected media-white and black compensation.
Equal reconstruction/presentation programs share exact admission; no CPU pixel fallback or
additional transfer image is introduced.

The [HDR/ICC corpus](../crates/jxl_wgpu_decode/test-data/hdr_icc_generator/README.md) reuses all
56 native HDR streams unchanged. Its 225 files contain 489,536 native/independent components for
nine paired profiles and four intents. GPU checks cover 1,958,144 components in 1,280 presentations,
plus independent PCS comparisons. Eight existing embedded RGB/Gray ICC sources at four
intensities produce another 512 PQ/HLG outputs; only image tone headers change, with compressed
ICC and frame bytes retained exactly. Direct resident checks cover 11,664 HDR components across
three variants, nine intensities and both directions. Existing codec precision budgets propagate
through the nonlinear connection before comparison, and the extracted native evaluator must
reproduce the existing 913-file SDR corpus exactly.

`COLOR-01/02`, `IO-01/03`, `QA-06` and the full JPEG XL goal remain **Partial**. This connection
does not infer physical display luminance from ICC `lumi` or implement peak/min-nit adaptation,
relative-to-max-display, or tone/gamut mapping. Wider HDR/CMYK/feature/LF combinations, complete
numeric-range/ISO precision and every remaining decoder/container/encoder/quality gate remain open.

## Explicit tone-mapping checkpoint

`COLOR-02` and `IO-01` now accept a requested display luminance range. The shared GPU BT.2408
curve resolves image intensity/minimum light and protects absolute/relative `linear_below` light.
Source and target PQ/HLG intensities remain distinct. Mapping follows reference storage,
composition and spot rendering; numeric channels retain their sample domain. Requested ICC
output maps PCS after intent connection and before device curves, including same-profile output.

[Policy and evidence](TONE_MAPPING.md): 4,986 pinned-libjxl samples, 44,874 actual-GPU primitive
components, dense protected/degenerate boundary tests, all 56 HDR streams/80 images producing
960 mapped outputs and 1,681,344 checked components, plus 32 embedded-ICC/XYB outputs. Whole and
bounded input, both layouts, progressive ownership, exact alpha and released budgets are checked.
Image output is 288 bytes, codec source/output 448 bytes, and resident ICC dispatch remains 320
bytes. Program admission/retry/cancellation also covers tone mapping with dynamic black detection
and complete CMYK/15-channel device output. No production CPU pixel path is introduced.

`COLOR-01/02`, `IO-01/03`, `QA-06` and the full goal remain **Partial**. Broader combined requested
ICC profile/intent pixel references, gamut mapping, automatic display adaptation, extreme numeric
ranges, HDR/LF/features, display textures and all remaining codec/conformance/encoder gates are open.

## Explicit RGB gamut checkpoint

`COLOR-02` and `IO-01` now expose target-linear RGB gamut mapping after composition, spots and
tone mapping, before transfer, alpha association and YUV packing. Saturation preference is
validated; protected light takes precedence, including thresholds which reach either white.
The image uniform is 304 bytes and codec source/output uniforms total 464 bytes. ICC input
joins the same final RGB stage; ICC device targets retain selected profile/intent semantics.

[Policy and evidence](GAMUT_MAPPING.md): 5,000 native libjxl records, 45,000 actual-GPU primitive
components across three workgroups, subnormal/cube-face/extended-light cases, 1,920 HDR
presentations with 3,362,688 independent/native-bounded color comparisons, and 64 embedded-ICC
outputs. Both layouts, whole/bounded input, progressive ownership, exact alpha and budget release
are checked. Exact ICC/spot admission and cancellation tests also select gamut mapping. No
production CPU image path is introduced, and existing source/reference assets remain unchanged.

The affected color/output/conformance rows and full goal remain **Partial**. Requested-profile
tone/gamut combinations, automatic display adaptation, HDR feature/LF and CMYK combinations,
full numeric/ISO precision, display textures and every remaining decoder/container/encoder and
quality requirement still require completion.

## Opaque container metadata checkpoint

`CONT-02` now has production Exif/XMP/JUMBF and unknown-box retention, atomic editing, canonical
container writing and bounded standard Brotli compression/decompression. `CONT-01/04` gain explicit
selection and original-payload preservation while their complete compatibility policies remain
**Partial**. Metadata ownership is independent of codestream spans and GPU memory; rendering
continues to use codestream fields. Native interoperability and 444 exact GPU presentations are
recorded in [container metadata](CONTAINER_METADATA.md). Existing fixture bytes remain unchanged.

The [published 2026 Part 2 contents and revision summary](https://cdn.standards.iteh.ai/samples/iso/iso-iec-18181-2-2026/8fe37de68af84f79a5df779b89a66a83/iso-iec-18181-2-2026.pdf)
include the HDR Gain Map box. Its required implementation is explicitly tracked as `CONT-07`;
preserving `jhgm` bytes does not satisfy it. `jxli`, JPEG reconstruction, full container compatibility,
remaining image/render paths, encoder syntax/quality and all other incomplete roadmap gates remain
part of the active full JPEG XL goal.

## Alternate gain-map still checkpoint

`CONT-07` now has bounded version-zero bundle/ISO fraction parsing and writing, borrowed auxiliary
codestreams, bounded alternate ICC metadata reconstruction, and an actual GPU alternate-still
entry point. Primary and map images use the normal selected-image/frame pipeline. Gain samples
stay in their original domain; baseline samples convert to the chosen linear application space.
The fused gain/output shader shares existing orientation, luminance, alpha and packing semantics.
Generic Modular color requests outside the narrow direct-output profile now enter the common
presentation surface, including 8-bit sources requested in wide linear F32.

[Gain-map evidence](GAIN_MAP.md) includes 64 native streams, 128 oriented/plain-or-compressed
reconstructions (78,336 RGBA comparisons), 32 HDR/layout/association outputs (19,584 comparisons),
64 native bundle rewrites and 128 ISO fraction rewrites. Current libavif supplies the ISO oracle;
the initial libultrahdr draft grammar has been removed. Records use individual denominators and
derive direction from headroom order. Six compatible-writer reads and 11 invalid-record rejections
have native agreement; Rust retains compatible opaque extensions within the bundle's 16-bit length.
Twenty-one fixture metadata payloads are corrected while all 128 codestreams and 256 native pixel
planes remain exact. Output lifetime, output/uniform byte
admission, cancellation, exact fractions and portable WGSL layout have separate tests. Reference
CMS negative extensions and rounded native primary coefficients are isolated explicitly; the
GPU/F64 and native gain tolerances remain fixed. New fixtures are reproducible from pinned sources.

This initial rendering checkpoint required a forward map with zero baseline headroom. The following
checkpoint extends that profile; complete ICC/animation/streaming support, alternate tone mapping,
extreme numeric/combined-feature conformance and official coverage remain open.

## Gain-map headroom and HDR-baseline checkpoint

`GpuDecoder::decode_gain_map` now accepts either headroom ordering, an exact alternate or a requested
display headroom, and explicit gain reference white. Exact rational cross-products and an F64 FMA
preserve endpoint direction/residuals before lowering the signed weight to F32. Exact baseline
selection skips the unused auxiliary decode and preserves ordinary output bits. Equal headrooms
follow libavif's identity policy; positive weights that underflow to F32 zero still apply offsets.
GPU gain math converts baseline light into the selected reference-white units and back, preserving
the ordinary output unit and PQ/HLG intensity contract. The fused gain uniform grows to 176 bytes;
no additional image, binding or dispatch is introduced.

Eighty bidirectional selections check 48,960 values at the earlier fixed linear/alpha bounds.
Forty-eight unchanged native HDR stills add 384 outputs / 774,656 values across PQ/HLG sources,
four intensities, original/XYB and both codecs, Gray/RGB, widths 17/257 and three reference whites.
Both headroom directions have three nonzero weights preceding Linear/PQ/HLG output and an exact
baseline comparison. The independent oracle propagates pre-existing baseline and auxiliary codec bounds
through bilinear sampling, inverse gamma, signed gain products and output transfer. Eight direct
gain-image decodes verify 4,528 original components against those bounds. All 464 headroom selections
also run pristine libavif weight selection and libultrahdr weighted application on supplied linear
pixels at the existing native formula tolerance. Earlier fixtures and 160 output checks remain
unchanged. `CONT-07` and the full JPEG XL goal remain **Partial** pending the remaining profiles,
numeric/official conformance and all other incomplete decoder/container/encoder/quality gates.
