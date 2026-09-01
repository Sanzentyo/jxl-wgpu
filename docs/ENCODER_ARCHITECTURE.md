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
reversible color transform, the Gradient predictor, packed-signed residual mapping, zero-run
formation, hybrid-uint tokenization, and histogram accumulation. Rust reads only the resulting
entropy artifacts and serializes standard JPEG XL metadata, prefix trees, TOC, and group bytes. A
GPU mapping or shader failure is an encode failure.

The repository does not vendor or retain `libjxl` as an upstream source tree. The production crates
use ordinary Cargo dependencies; official source is consulted only for specification and
implementation audits.

## Implemented profile

`LosslessModularBackend` advertises exactly:

| Property | Implemented value |
|---|---|
| Coding mode | Modular lossless |
| Color models | Gray (one NonColor `X000` plane), RGB (`Rgb`/`XYZ1`), RGBA (`Rgb`/`XYZW`); RGBA alpha is one unassociated extra channel |
| Sample depths | every integer `1..=16` (`1..=8` in `u8` words, `9..=16` in `u16` words) |
| Input | pitch-linear `wgpu::Buffer`; single-plane, unsigned, native byte order, `ChromaSubsampling::None` |
| Extent | `1..2^30` per axis, further bounded by device limits |
| Frame/group layout | standard 256x256 PassGroups, multi-group, row-major TOC |
| Animation | `true` (5 blend modes, signed crop, 4 reference slots, timecodes) |
| Determinism | `CrossDevice` integer GPU artifacts and deterministic host assembly |
| Progressive passes | `max_progressive_passes = 1` |
| Implemented stages | `ColorTransform`, `ModularTransform`, `ModularPrediction`, `ModularResidualTokenization`, `HistogramReduction` |
| Predictor | JPEG XL Gradient predictor |
| Modular transforms | none |
| Entropy | JPEG XL prefix code with LZ77 distance 1, not ANS; fixed MA tree |
| Filters | Gaborish off, EPF zero iterations |
| Output | raw codestream or standard `jxlc` container; private `jwgp` index emitted only for single-group Gray8 containers |

The backend rejects textures, planar RGB, BGR/BGRA, non-native byte order, chroma subsampling,
YUV/NV12, MSB-aligned sub-16-bit words, explicit non-sRGB color specifications, and progressive
passes > 1. YUV/NV12 ingestion is not implemented.

### GPU artifact ABI

The token kernel is still `@compute @workgroup_size(1)` (`lossless_modular.wgsl:162`).
Parallelism comes only from the number of (PassGroup, channel) pairs in the dispatch; one
invocation scans its whole group serially. Streamed multi-batch jobs use two submissions per batch
(one histogram pass, one serialization pass). This remains a correctness milestone, not the
eventual performance topology. Its readback buffer consists of little-endian `u32` words:

```text
word 0       event_count
word 1..19   raw hybrid-token counts (19 entries)
word 20..52  LZ77 hybrid-token counts (33 entries)
word 53..    event_count records of:
              kind, token, extra_bit_count, extra_bits
```

`kind == 0` is a raw residual token and `kind == 1` is a zero-run token. No source sample or residual
plane is copied into a private container box. The ABI is bounded before allocation: at most
`pixels + ceil(pixels / 8) + 1` events per group channel.

The parameter ABI is one `#[repr(C)]`, `bytemuck::Pod` Rust value and the matching WGSL structure:

```text
ModularParams / Params = {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
    output_word_offset: u32,
    channel: u32,
    channels: u32,
    bytes_per_sample: u32,
    sample_mask: u32,
    _padding: array<u32, 55>, // [u32; 55] in Rust
}
size = 256 bytes, alignment = 4 bytes
```

A compile-time size assertion and a test for size, alignment, byte order, field order, and the WGSL
declaration prevent accidental ABI drift. An explicit 256-byte array stride keeps every batch
boundary valid for portable storage-buffer offset alignment. The source is never bound as an
unchecked whole buffer. Admission computes the final sampled byte with checked
`offset + (height - 1) * row_stride + width * channels * bytes_per_sample`, checks arithmetic for
overflow, rounds the final u32 load up to four bytes, and binds only the enclosing range. The
binding base is rounded down to both the device's `min_storage_buffer_offset_alignment` and u32 word
alignment; WGSL receives the resulting relative offset. Its final address must fit WGSL's u32
address space.

Artifact and MAP_READ allocations use checked word/byte arithmetic, are four-byte copy aligned, and
must fit both `max_storage_buffer_binding_size` and `max_buffer_size`. The public
`LosslessModularEncoder::memory_plan` reports valid bits, component storage bytes, channel count,
format, group grid, full and peak source binding ranges, parameter storage, peak artifact storage,
diagnostic total artifact bytes, readback bytes, batch count, exact GPU submission count, streaming
mode, owned bytes per job, and addressed bytes per job. `for_in_flight(n)` reports checked aggregate
bytes for a caller-selected concurrency ceiling, while `memory_limits` exposes the relevant device
limits.

The parameter, artifact, and mapped-readback allocations form one exclusive reusable buffer set.
Sets match the exact artifact size, remain leased through map completion and consumption, and are
returned safely even when a Future is abandoned. Idle retention defaults to 32 MiB with a
256-set object cap; `buffer_pool_stats`, `set_buffer_pool_limit`, and `clear_buffer_pool` expose
reuse and control. Caller-owned source bindings are neither copied into nor retained by this pool.

The predictor is computed in signed integer arithmetic:

```text
ac = left - top_left
ab = left - top
bc = top - top_left
gradient = ac + top
clamped = (ab xor bc) < 0 ? top : left
prediction = (ac xor bc) < 0 ? gradient : clamped
```

The first row predicts from the left; the first sample of later rows predicts from the first sample
of the previous row. Residuals use the JPEG XL packed-signed mapping. Eight-sample chunks turn a run
longer than seven zeros into the configured LZ77 form.

Raw tokens use hybrid configuration `000`: token zero represents zero; for token `t > 0`, read
`t - 1` extra bits and add `2^(t - 1)`. LZ77 uses configuration `400`: values below 16 are direct;
otherwise token `t` reads `t - 12` bits and adds `2^(t - 12)`. The decoded run length is that value
plus eight and the configured distance is one.

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
`vardct_encode_bounded` and `vardct_encode_quantize` accept every linear `KernelVariant`; their
fixed 16 KiB and 1 KiB workgroup allocations are checked before pipeline creation. VarDCT control
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

LF global carries the shared Modular tree and entropy code (four prefix codes derived from combined
channel histograms); LF groups and HF global are empty; each PassGroup carries its own group header
and channel token streams inside standard row-major TOC groups. Group payload and TOC are byte
aligned.

### `jwgp` acceleration index

The standard `jxlc` remains the source of truth and must decode without private metadata.
For single-group Gray8 containers, `encode_container` adds an optional private `jwgp` box containing
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

- **Multi-group Modular (Slice 3)**: Standard 256x256 PassGroups, multi-group row-major TOC layout,
  two-pass streaming with global histogram aggregation, and out-of-order group completion.
- **Lossless RGB and RGBA (Slice 4 half)**: Interleaved unsigned RGB and RGBA at depths `1..=16`
  with GPU-side reversible color transform (YCoCg) and unassociated alpha extra-channel support.
- **Lossless Modular animation (Slice 6)**: Multi-frame `LosslessModularAnimationSession` supporting
  standard timebases, exact durations and timecodes, signed crop rectangles, all 5 blend modes,
  alpha blending, and 4 reference slots with runtime-neutral in-flight futures.
- **VarDCT baseline plus bounded DCT8 AC (Slice 5 partial)**: `VarDctEncoder` executes all 27
  standard strategies, and `TiledVarDctEncoder` supports multi-LF/multi-AC-group DCT8 grids with
  checked axes through 16K. Both frontends serialize validated exact-binary16 LF dequantization plus
  LF/HF chroma-correlation metadata. The bounded DCT8 kernel additionally performs the forward DCT,
  default-matrix quantization, natural-order coefficient scan, signed token generation, histogram
  construction, and AC bit-fragment serialization without exposing pixels or coefficients to the
  host. Its `HfEntropyPlan` currently selects one prefix cluster for all 495 coefficient contexts,
  disables LZ77, and emits one pass. That plan is a stable policy boundary rather than a temporary
  wire format: future adaptive clustering, ANS/LZ, coefficient orders, and pass selection can use
  different plans while retaining the GPU artifact contract. The scalable/tiled path and non-DCT8
  strategies still quantize AC to zero.

### Remaining work

The authoritative encoder items, dependencies, priorities, and acceptance gates are the `MOD-E`,
`VDCT-E`, `ENT-E`, `ENC`, and encoder-facing `IO` rows in
[`FULL_JPEG_XL_ROADMAP.md`](FULL_JPEG_XL_ROADMAP.md). After the structural-refactoring gate, the
nearest work remains parallel Modular token production, native YUV/NV12-family ingestion, scalable
and strategy-complete nonzero VarDCT AC coding, and the rate/quality control built on top of it.
Batched codec submission and advanced performance instrumentation stay separate from
format-completeness claims.

Until an item is implemented and validated, capability negotiation must reject it. Benchmarks,
wrappers, and CPU oracles do not expand the advertised production capability.
