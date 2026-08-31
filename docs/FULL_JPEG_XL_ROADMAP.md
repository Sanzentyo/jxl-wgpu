# Full JPEG XL implementation roadmap

Status date: 2026-08-31. This document is the canonical capability and implementation backlog for
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
| Raw/`jxlc`/`jxlp` transport and header inventory | **Partial** | Bounded still-image transport, frame/TOC inventory, ICC reconstruction, and feature metadata exist; streaming box delivery and the complete container API do not. |
| Lossless Modular decode | **Partial** | One final Gray/RGB/RGBA integer still, 1–16 bits, one pass, standard YCoCg, Prefix/ANS, LZ77, and bounded MA prediction. |
| VarDCT decode | **Partial** | A separate authoritative 8-bit XYB engine covers nine regular zero-AC single transforms and tiled DCT8 with single-pass nonzero AC, natural/custom DCT8 order, one LF group, and extents through 2048×2048. |
| Lossless Modular encode | **Partial** | Gray/RGB/RGBA integer input, 1–16 bits, 256×256 groups, one pass, fixed Gradient/YCoCg and prefix+RLE/LZ77 profile; standard animation is implemented. |
| VarDCT encode | **Partial** | All 27 strategy identifiers execute, but only in a fixed distance-25 LF-only profile with every AC coefficient quantized to zero; tiled DCT8 is limited to one LF group. |
| Restoration/render graph | **Partial** | Reusable upsampling, Gaborish, EPF, blend, color, and display kernels exist in `jxl_wgpu`; the stock decoders do not yet route the full legal feature graph through them. |
| Output formats | **Partial** | Native integer Gray/RGB/RGBA and 30 portable VPI pitch-linear outputs exist for the lossless Gray8 conversion path; VarDCT currently returns packed RGB8 only. |
| Async/concurrency/memory | **Partial** | Native blocking and runtime-neutral futures, browser compilation, one shared byte budget, leased output lifetime, true aggregate readback, and bounded pools exist; codec submission is not yet coalesced across images. |
| Decoder animation and composition | **Missing** | Metadata/session contracts exist, but the stock decoder still rejects animation/reference-frame streams. |
| JPEG bitstream reconstruction | **Missing** | No `jbrd` reconstruction or JPEG coefficient transcode path exists. |

The 24-case checked-in round-trip corpus proves the narrow stock Modular profile across diverse
dimensions up to 15360×8640. It is not evidence of full JPEG XL coverage.

## Required implementation items

Priority is `P0` for a blocker on the decoder’s core image path, `P1` for required complete-format
coverage, and `P2` for production encoder quality, broad product integration, or performance after
correctness. Dependencies name other item IDs in this document.

### A. Unified frontend, transport, and container

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `FRONT-01` | P0 | **Partial** | `GpuDecoder::wgpu` now inventories once and automatically selects Modular or VarDCT while sharing one backend byte budget; one actual-adapter test decodes both modes sequentially through the same decoder and checks pixels and reservation release. Completion still requires lowering every frame into one bounded backend-neutral execution graph plus mixed-mode referenced-frame tests. | — |
| `FRONT-02` | P0 | **Partial** | Preserve exact bit ranges and logical/physical TOC order for global, LF-group, HF-global, and pass-group sections, including entropy-coded permutations and arbitrary group order. Test scanline, center-first, and permuted fixtures. | — |
| `FRONT-03` | P1 | **Partial** | Implement incremental bounded input: headers, boxes, fragments, frame sections, and end-of-input may arrive in arbitrary chunks without collecting an unbounded codestream copy. Test every byte split around signatures, box headers, TOCs, and entropy words. | `CONT-01` |
| `CONT-01` | P1 | **Partial** | Fully validate naked streams, `jxlc`, ordered and out-of-order `jxlp`, large-box sizes, unknown boxes, ordering constraints, and truncation. Expose a streaming box/event API and preserve requested unknown boxes byte-for-byte. | — |
| `CONT-02` | P1 | **Missing** | Expose, preserve, replace, and encode the opaque payloads of Exif, XMP (`xml `), and JUMBF (`jumb`) boxes, and decompress/compress `brob` boxes with explicit size/decompression-ratio limits. Rendering must continue to follow codestream metadata precedence. | `CONT-01` |
| `CONT-03` | P1 | **Missing** | Read, validate, generate, and use animation frame indexes (`jxli`) for bounded seeking. Random access must restore the reference-frame dependency chain before presenting a target frame. | `CONT-01`, `FRAME-04` |
| `CONT-04` | P1 | **Missing** | Preserve container extensions and current-version compatibility rules; reject future unsupported rendering extensions before authoritative output while allowing safe skippable boxes. | `CONT-01` |
| `CONT-05` | P1 | **Missing** | Implement `jbrd` parsing/emission and bit-identical JPEG reconstruction, including required JPEG coefficient/marker/metadata state. Verify byte equality on a diverse JPEG corpus, not only decoded pixels. | `VDCT-D03`, `VDCT-E02` |
| `CONT-06` | P1 | **Partial** | Implement bounded encoder output for ordinary progressive order, seek-back TOC assembly, and out-of-order `jxlp` streaming. Generated fragments must reassemble to the same logical codestream, and partial writes must never be reported as a finished container. | `CONT-01`, encoder packet topology |

### B. Global codestream and frame metadata

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `META-01` | P1 | **Partial** | Execute every image-header dimension, orientation, intrinsic size, preview, animation, original-profile, bit-depth/exponent, tone-mapping, opsin, upsampling, and extension field that currently is only inventoried. Boundary tests cover defaults and non-default encodings. | `FRONT-01` |
| `META-02` | P1 | **Partial** | Execute complete frame headers: regular/LF/reference-only frames, names, signed crops, duration/timecode, save-as-reference, is-last, group-size shift, encoding mode, resampling, passes, restoration, and per-channel blend/upsampling state. | `FRONT-01` |
| `META-03` | P1 | **Missing** | Decode previews and recursively referenced LF frames without corrupting the main-frame state. Expose preview/progressive results as explicitly non-final outputs. | `FRONT-03`, `FRAME-03` |
| `META-04` | P1 | **Partial** | Enforce checked limits for dimensions, frame count, extra channels, names, groups, passes, tree nodes, histograms, boxes, recursion, and allocations before submission. Fuzzing must show no panic/OOB/unbounded allocation. | all frontend work |

### C. Common entropy and packet execution

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `ENT-D01` | P0 | **Partial** | The bounded Modular, VarDCT packet, and DCT8 pass-group paths share a typed 12-byte Rust/WGSL `EntropyStreamParams` prefix for token bounds and the LZ ring mask while preserving consumer-specific records and bindings. The single-pass DCT8 consumer now executes real nonzero coefficient contexts through the same Prefix/ANS, hybrid-uint, context-map, histogram, alias-distribution, and LZ77 primitives. Completion requires every remaining VarDCT LF/HF consumer plus differential symbol/termination tests for every distribution form. | `FRONT-02` |
| `ENT-D02` | P0 | **Partial** | Decode arbitrarily large legal group streams through bounded windows while retaining enough history for LZ77 and context state. Prove exact results across window boundaries and cancellation. | `ENT-D01` |
| `ENT-E01` | P1 | **Partial** | Add GPU ANS token serialization, histogram clustering, context clustering, hybrid-uint selection, general LZ77 search/distances, and canonical entropy metadata. `djxl`/`jxl` must accept every generated family. | — |
| `ENT-E02` | P2 | **Missing** | Select entropy configurations by effort and workload with deterministic modes. Report density and speed independently; no heuristic may change lossless pixels. | `ENT-E01`, `QA-05` |

### D. Modular decode

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `MOD-D01` | P0 | **Partial** | Remove the fixed stock profile restriction: execute all legal channel counts, integer/floating original sample metadata, bit depths, channel shifts, and lossy Modular parameters within checked device limits. Compare raw channel values or required precision bounds with libjxl. | `ENT-D01`, `COLOR-01` |
| `MOD-D02` | P0 | **Partial** | Execute every legal MA-tree property, decision, leaf multiplier/offset, channel reference, predictor, and custom weighted-predictor header without implementation-specific tree caps beyond advertised resource limits. Add generated tree differential tests. | `ENT-D01` |
| `MOD-D03` | P0 | **Partial** | Implement arbitrary ordered Modular transform stacks: all 42 reversible color transforms, Palette including delta/lossy palette and meta-channels, and horizontal/vertical Squeeze with exact inverse channel geometry. | `MOD-D01` |
| `MOD-D04` | P0 | **Missing** | Support 128/256/512/1024 groups, Global/LF/HF streams, all legal group permutations, Squeeze pyramids, and up to three progressive Modular passes. Progressive output must converge exactly to final output. | `MOD-D03`, `FRONT-02` |
| `MOD-D05` | P1 | **Missing** | Decode Modular streams embedded inside VarDCT for LF images, quant fields, and all extra channels, with the required interleaving and pass dependencies. | `MOD-D03`, `VDCT-D01` |

### E. Modular encode

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
| `VDCT-D01` | P0 | **Partial** | One LF-global/LF-group/HF-global topology now executes nonempty single-pass DCT8 pass groups in standard TOC order. Completion requires multiple LF groups, recursive LF frames, arbitrary group order, additional passes, and images beyond 2048 pixels per axis. | `FRONT-02`, `ENT-D01` |
| `VDCT-D02` | P0 | **Partial** | Decode the block context map, strategy map, all 27 regular and special strategies in mixed images, quant field, sharpness, chroma-from-luma factors, and required Modular side images. No legal strategy may be lowered to DCT8. | `VDCT-D01`, `MOD-D05` |
| `VDCT-D03` | P0 | **Partial** | Single-pass DCT8 now decodes real nonzero counts, contexts, Prefix/ANS tokens, signs, natural or custom DCT8 order, quant bias/dequantization, and coefficient placement entirely on GPU after the small order metadata permutation is expanded on the host. A 438×589 libjxl fixture with six groups and 4,070 tasks matches Rust `jxl` and `djxl` within one RGB8 code. Completion requires all order masks/strategies, multiple presets, pass refinement, sparse/dense/edge fixtures, and corruption at the coefficient layer. | `VDCT-D01`, `ENT-D01` |
| `VDCT-D04` | P0 | **Partial** | Execute inverse transforms for every strategy and mixed block map with exact edge extension/cropping. Existing resident kernels must be reached from the stock decoder, with per-strategy precision tests. | `VDCT-D02`, `VDCT-D03` |
| `VDCT-D05` | P0 | **Partial** | Default DCT8 dequantization, stream-selected global/LF/HF scales, quant bias, adaptive LF smoothing, and DC prediction execute on GPU for the bounded one-LF-group profile. Completion requires all default strategy matrices, custom matrices, nonzero chroma-from-luma factors, and cross-LF-group behavior. | `VDCT-D02`, `VDCT-D03` |
| `VDCT-D06` | P1 | **Missing** | Sum spectral and quantized progressive AC passes and expose DC/LF/pass progression without publishing an invalid final frame. Every intermediate level is compared with libjxl and the final image meets conformance bounds. | `VDCT-D03`, `API-03` |
| `VDCT-D07` | P1 | **Partial** | Support color/extra-channel resampling and normative 2×/4×/8× upsampling weights for color and extra channels, including odd borders and group halos. | `MOD-D05`, `RENDER-01` |

### G. VarDCT encode

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `VDCT-E01` | P0 | **Partial** | Generalize the existing 27-strategy forward-transform path from single/fixed blocks to mixed strategy maps, arbitrary image extents, multiple LF groups, and all edge blocks. Forward/inverse scalar oracles cover every strategy. | `VDCT-D04` |
| `VDCT-E02` | P0 | **Missing** | Quantize and serialize real nonzero AC coefficients with correct orders, contexts, signs, histograms, and pass groups. Generated streams must decode with libjxl across all strategies and group boundaries. | `ENT-E01`, `VDCT-E01`, `VDCT-D03` |
| `VDCT-E03` | P1 | **Missing** | Implement adaptive quant fields, distance/quality control, DC/LF/AC quant selection, quant bias, chroma-from-luma search, and a bounded rate-control loop. Report size and quality distributions, not only one fixture. | `VDCT-E02`, `QA-05` |
| `VDCT-E04` | P1 | **Missing** | Select strategy maps and coefficient orders by content and effort, including all special transforms. Decisions must be deterministic when requested and must improve a declared objective over DCT8-only. | `VDCT-E02`, `VDCT-D02` |
| `VDCT-E05` | P1 | **Missing** | Encode spectral, quantized, and DC progressive modes plus center-first/saliency group ordering. Every partial stream must remain decodable at its declared progression point. | `VDCT-E02`, `VDCT-D06` |
| `VDCT-E06` | P2 | **Missing** | Add perceptual optimization tiers, including iterative adaptive quantization and quality feedback. Compare against `cjxl` at matched distance/size with Butteraugli and additional artifact-sensitive metrics. | `VDCT-E03`, `QA-05` |

### H. Rendering features and restoration

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `RENDER-01` | P0 | **Partial** | Route exact 2×/4×/8× upsampling, halo extension, crop, and channel recombination through the unified render graph. Tile boundaries must equal whole-frame reference results. | `FRONT-01` |
| `RENDER-02` | P0 | **Partial** | Route Gaborish and EPF0/1/2/3 with signaled strengths, sharpness, sigma, border handling, and cross-group halos. Meet 18181-3 precision bounds on each stage and their composition. | `VDCT-D05`, `RENDER-01` |
| `RENDER-03` | P1 | **Missing** | Decode and render patches with reference-frame lookup, all blend modes, alpha/extra-channel behavior, clipping, and mixed Modular/VarDCT sources. | `FRAME-01`, `MOD-D05` |
| `RENDER-04` | P1 | **Missing** | Decode and rasterize splines with normative quantization, Catmull–Rom geometry, color, thickness, and clipping. Include stress tests for count/length limits. | `COLOR-01` |
| `RENDER-05` | P1 | **Missing** | Decode and synthesize noise from the signaled luma-dependent model with deterministic seed/state and correct ordering relative to filters/color. | `VDCT-D05` |
| `RENDER-06` | P2 | **Missing** | Encoder detection/selection for patches, dots, noise models, Gaborish inverse sharpening, and EPF signaling. Each tool needs an on/off corpus and an objective improvement gate. | decode counterparts, `QA-05` |

### I. Frames, animation, and composition

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `FRAME-01` | P0 | **Partial** | Implement decoder crops with negative origins/oversized frames and Replace/Add/Blend/Mul/MulAdd composition for color and each extra channel, including associated/unassociated alpha. Compare every mode and off-canvas edge with libjxl. | `FRONT-01`, `COLOR-03` |
| `FRAME-02` | P0 | **Missing** | Maintain four reference slots, save-before/save-after semantics, LF/reference-only and hidden frames, dependency validation, and bounded lifetime accounting on GPU. Mixed Modular/VarDCT references must work. | `FRAME-01` |
| `FRAME-03` | P0 | **Missing** | Decode animation timebase, duration, timecode, zero-duration layers, looping, and default coalescing into ordered GPU frames. Blocking, poll, and `Future` APIs must share one state machine. | `FRAME-02` |
| `FRAME-04` | P1 | **Missing** | Add non-coalesced frame/layer output, skip/progressive controls, and bounded seek/restart semantics without losing required references. | `FRAME-03`, `CONT-03` |
| `FRAME-05` | P1 | **Partial** | Extend the implemented Modular animation encoder to VarDCT/mixed frames, arbitrary extra-channel blends, hidden/reference frames, names, previews, and frame indexing. `djxl` verifies composed output and timing. | encoder mode items, `CONT-03` |

### J. Color, HDR, extra channels, input, and output

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `COLOR-01` | P0 | **Partial** | Execute every enumerated color encoding and compressed ICC profile, including grayscale/RGB/XYB/CMYK cases, white point, primaries, rendering intent, and transfer functions such as linear, sRGB, PQ, and HLG. Never relabel unsupported pixels. | `META-01` |
| `COLOR-02` | P0 | **Partial** | Execute custom opsin inverse matrices/biases, intensity target, min nits, relative-to-max-display, tone mapping, gamut mapping, and requested output color conversion with declared precision. | `COLOR-01` |
| `COLOR-03` | P0 | **Partial** | Decode and expose all extra-channel types: alpha, depth, spot color, selection mask, black, CFA, thermal, reserved/unknown, and optional; preserve names, bit depths, exponent bits, dimensional shifts, spot metadata, and alpha association. | `MOD-D05`, `META-01` |
| `COLOR-04` | P1 | **Partial** | Apply orientation 1–8 and spot-color rendering at the correct graph stage. Provide explicit keep-orientation/keep-spot-channel controls without changing default normalized output. | `COLOR-01`, `COLOR-03` |
| `IO-01` | P1 | **Partial** | Make native Gray/RGB/RGBA and every classified pitch-linear RGB/BGR/YUV format available from both Modular and VarDCT, including NV12-family, packed 4:2:2, P010/P012/P016, odd extents, range, matrix, transfer, and chroma siting. | `COLOR-01`, both decoders |
| `IO-02` | P1 | **Partial** | Accept GPU buffers and textures for encoder input in the same portable pitch-linear RGB/BGR/YUV family. Normalize planar/semi-planar formats and bit alignment in WGSL; no host pixel conversion. | `COLOR-01`, encoder modes |
| `IO-03` | P1 | **Partial** | Present SDR/HDR outputs directly as sampleable/renderable `wgpu::Texture` objects with explicit texture format, alpha, transfer, and surface compatibility. Add PQ/HLG/wide-gamut display paths and GPU-resident validation. | `COLOR-02`, `IO-01` |
| `IO-04` | P2 | **Partial** | Use native shader `f64` when the backend exposes it and an operation benefits from it; otherwise require an explicit precision policy. F64 storage capability must not be confused with JPEG XL conformance or silently widened arithmetic. | capability negotiation |
| `IO-05` | — | **Out of scope** | CUDA/VPI block-linear and block16-linear layouts remain typed unsupported because portable `wgpu` cannot represent their memory contract. All 30 VPI pitch-linear formats remain in scope. | — |

### K. API, scheduling, and resource safety

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `API-01` | P1 | **Partial** | One capability query must report exact decode, encode, color, output, precision, memory, workgroup, and platform limits. No versioned aliases or compatibility shims are required; breaking APIs should model current semantics directly. | all feature rows |
| `API-02` | P1 | **Partial** | Keep native blocking plus runtime-neutral `Future`/`Poll` completion for stills, animation, progressive delivery, encode, decode, display, and readback. Browser paths may reject blocking waits but not require a named runtime. | feature state machines |
| `API-03` | P1 | **Missing** | Add progressive/ROI decode targets, dirty-region metadata, cancellation, and resumable input state. A partial output must carry its exact quality/resolution/finality contract. | `FRONT-03`, progressive mode rows |
| `API-04` | P2 | **Partial** | Coalesce multiple small stills/frames into shared codec command buffers and bounded maps. Distinguish codec batching from host-thread fan-out and the existing aggregate readback. | stable feature graph |
| `API-05` | P1 | **Partial** | Preserve one shared byte budget across encode/decode/display/readback; account scratch, pooled physical bytes, output clones, reference slots, ICC/box buffers, and abandoned futures. Device loss and cancellation must release exactly once. | all allocations |
| `API-06` | P1 | **Partial** | Keep all Rust/WGSL ABI records `repr(C)` + `bytemuck::Pod` where valid, with compile-time sizes, explicit padding/alignment, checked dynamic offsets, bounded u32 addressing, and no string-content shader tests. Validate shaders by parse/compile/execute and semantic outputs. | every shader change |
| `API-07` | P1 | **Partial** | Define direct-map exclusivity by the underlying resource identity/accounting owner, keep it instance-scoped, and reject conflicting mappings. Document raw-handle escape as outside lease accounting. | readback API |
| `API-08` | P1 | **Partial** | Keep public failures as structured `thiserror` enums with source chaining and distinct malformed-input, unsupported-feature, resource-limit, capability, device, cancellation, and retryable-pressure variants. Tests assert typed variants and fields, not display strings. | every public path |

### L. Encoder input decisions and product controls

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `ENC-01` | P1 | **Partial** | Support integer and floating-point source precision, alpha association, arbitrary extra-channel planes, color/ICC metadata, row padding, planar/interleaved buffers, and textures without CPU pixel normalization. | `IO-02`, `COLOR-03` |
| `ENC-02` | P1 | **Missing** | Add explicit lossless/lossy, distance/quality, effort, decoding-speed, resampling, progressive, metadata-preservation, deterministic, and memory/latency policy controls with typed invalid combinations. | encoder mode rows |
| `ENC-03` | P2 | **Missing** | Implement automatic Modular versus VarDCT selection and per-feature decisions. Evidence must show the selected path and compare it with forced alternatives. | complete baseline encoders |
| `ENC-04` | P1 | **Missing** | Encode all extra channels with independent distance, upsampling, dimensional shift, and blend contracts; preserve invisible color according to policy. | `ENC-01`, `COLOR-03` |
| `ENC-05` | P1 | **Missing** | Implement GPU JPEG entropy/coefficient ingestion and lossless JPEG recompression, optional chroma-from-luma, metadata preservation, and `jbrd` emission without a CPU JPEG coefficient or pixel codec. | `CONT-05`, `VDCT-E02` |

### M. Conformance, robustness, and performance evidence

| ID | Pri | State | Requirement and acceptance gate | Depends on |
|---|---:|---|---|---|
| `QA-01` | P0 | **Partial** | Import or generate positive and negative fixtures for every roadmap feature before advertising it. The corpus manifest records expected profile, dimensions, source, oracle, precision, and stock/future status. | each feature |
| `QA-02` | P0 | **Missing** | Run the ISO/IEC 18181-3 decoder conformance suite or an auditable equivalent and record per-test precision results. Passing local round trips alone cannot produce a “full decoder” claim. | full decode path |
| `QA-03` | P0 | **Partial** | Differentially compare decode with libjxl and encode with both libjxl/`djxl` and the Rust `jxl` oracle where applicable. Cover cross-products of tools, not only isolated single-feature fixtures. | each feature |
| `QA-04` | P1 | **Partial** | Fuzz raw/container parsing, entropy metadata, MA trees, transforms, coefficients, frame references, ICC/brob/jbrd, cancellation, and resource limits. Add truncation and corruption at every bounded stream layer; no panic, hang, OOB, or partial authoritative result. | parsers and engines |
| `QA-05` | P1 | **Missing** | Build lossy encode evaluation with Butteraugli plus PSNR/SSIM and artifact-focused cases, bitrate matching, deterministic source hashes, and images from tiny through 16K across square, portrait, panorama, and extreme one-pixel axes. | `VDCT-E02` |
| `QA-06` | P1 | **Partial** | Expand animation, HDR/wide-gamut, ICC/CMYK, extra-channel, progressive, JPEG-reconstruction, metadata, and mixed Modular/VarDCT corpora. Include 1/255/256/257/2048 boundaries and 16K where memory permits. | feature rows |
| `PERF-01` | P2 | **Partial** | Add in-process libjxl decode/encode baselines. Keep external-process `djxl`/`cjxl` results labelled separately; they are not fair warm-library comparisons. | stable correctness |
| `PERF-02` | P2 | **Partial** | Record GPU timestamps, host submit/wait/map time, pipeline compilation, upload/readback, peak active/pooled/driver memory, submissions, maps, and output hashes for isolated, warm, concurrent, batched, animation, display, and CPU-readback paths. | `API-04`, stable correctness |
| `PERF-03` | P2 | **Partial** | Autotune validated 32/64/128/256 workgroups by adapter/profile/output/workload, including encoder kernels. Reject choices exceeding workgroup storage or device limits and persist profiles with typed version checks. | each tunable kernel |
| `PERF-04` | P2 | **Missing** | Establish release gates for CPU-readback parity/wins, GPU-resident/display wins, concurrent throughput, encode speed, quality/size, and peak memory on multiple native adapters plus browser WebGPU. No universal win is claimed from one Apple M5 snapshot. | `PERF-01`, `PERF-02` |

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
7. **Complete containers and JPEG reconstruction**: `CONT-02..06`, `ENC-05`.
8. **Raise encoder quality and breadth**: `VDCT-E03..06`, `ENC-01..04`, `RENDER-06`.
9. **Close conformance and performance gates**: `QA-01..06`, `PERF-01..04` across native and
   browser adapters.

The coding-mode selector, shared typed entropy-stream ABI, and first nonzero-AC multi-group DCT8
decode milestone are implemented. The immediate next target is restoration and remaining
`ENT-D01/FRONT-02` topology, followed by mixed-strategy `VDCT-D01..05`. The corresponding GPU
nonzero-AC encoder remains required before the fixture can be re-encoded without a CPU pixel path.

## Claims explicitly prohibited before their gates pass

- Executing all 27 inverse-transform kernels is not full VarDCT support while nonzero AC entropy,
  quant fields, mixed strategies, and passes remain missing.
- Executing all MA predictors is not full Modular support while Palette, Squeeze, arbitrary RCT
  stacks, group sizes, and progressive streams remain missing.
- Emitting one standards-compatible fixed Gradient or zero-AC stream is not a production-complete
  encoder; normative output ability and quality/rate-control search are separate gates.
- Parsing `jxlc`/`jxlp` is not complete container support without metadata, `brob`, `jxli`, streaming,
  unknown-box policy, and JPEG reconstruction.
- Having blend kernels and animation metadata types is not decoder animation support without GPU
  reference slots, hidden frames, dependency ordering, coalescing, and lifetime accounting.
- A CPU oracle, external wrapper, generated fixture, or CPU fallback cannot expand the production
  GPU capability claim.

## Primary references

- [JPEG XL format overview](https://github.com/libjxl/libjxl/blob/main/doc/format_overview.md)
- [libjxl encoder API and feature settings](https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h)
- [libjxl codestream metadata and extra-channel types](https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/codestream_header.h)
- [libjxl encoder effort/tool selection](https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md)
- [libjxl codec architecture overview](https://github.com/libjxl/libjxl/blob/main/doc/xl_overview.md)

The ISO text and official conformance material remain normative. The public libjxl sources above
are implementation and audit references, not permission to substitute its CPU codec in production.
