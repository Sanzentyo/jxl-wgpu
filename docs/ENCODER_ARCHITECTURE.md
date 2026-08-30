# GPU-only JPEG XL encoder architecture

Status: executable profile plus production-facing orchestration API. The only profile currently
advertised by a concrete backend is the deliberately narrow lossless grayscale profile described
below. Everything else is rejected through a typed capability error.

## Non-negotiable boundary

The production encoder may use the host for bitstream/container orchestration, validation, job
ordering, and deterministic serialization. It may not use the host to normalize pixels, predict
samples, form residuals, transform or quantize coefficients, tokenize image data, build image-data
histograms, or silently replace a failed GPU job.

For the executable profile, `lossless_gray8.wgsl` reads the caller's `wgpu::Buffer` and performs
the Gradient predictor, packed-signed residual mapping, zero-run formation, hybrid-uint
tokenization, and histogram accumulation. Rust reads only the resulting entropy artifacts and
serializes standard JPEG XL metadata, prefix trees, TOC, and group bytes. A GPU mapping or shader
failure is an encode failure.

The repository does not vendor or retain `libjxl` as an upstream source tree. The production crates
use ordinary Cargo dependencies; official source is consulted only for specification and
implementation audits.

## Implemented profile

`LosslessGray8Backend` advertises exactly:

| Property | Implemented value |
|---|---|
| Coding mode | Modular lossless |
| Samples | one unsigned 8-bit grayscale plane |
| Input | pitch-linear `wgpu::Buffer`, `PixelFormat::non_color(Unsigned, 8, [X])` |
| Extent | `2..=256` in each dimension |
| Frame/group layout | one still frame, one fused group, one pass, `is_last` |
| Predictor | JPEG XL Gradient predictor |
| Modular transforms | none |
| Entropy | JPEG XL prefix code with LZ77 distance 1, not ANS |
| Filters | Gaborish off, EPF zero iterations |
| Determinism | integer GPU artifacts and deterministic host assembly |
| Output | raw codestream or standard `jxlc` container with a validated `jwgp` index |

The backend rejects textures, RGB, alpha, YUV/NV12, multiple planes, depths other than eight,
dimensions requiring multiple groups, animation, progressive passes, VarDCT, and non-default frame
options. Those formats belong in later capabilities; their presence in `jxl_gpu_formats` is not a
claim that this encoder profile already accepts them.

### GPU artifact ABI

The first kernel intentionally uses one invocation. This is a correctness milestone, not the
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
`pixels + ceil(pixels / 8) + 1` events.

The uniform ABI is one `#[repr(C)]`, `bytemuck::Pod` Rust value and the matching WGSL structure:

```text
Gray8Params / Params = { width: u32, height: u32, row_stride: u32, byte_offset: u32 }
size = 16 bytes, alignment = 4 bytes
```

A compile-time size assertion and a test for size, alignment, byte order, field order, and the WGSL
declaration prevent accidental ABI drift. The source is never bound as an unchecked whole buffer.
Admission computes the final sampled byte with checked `offset + (height - 1) * row_stride + width`,
checks the full `offset + height * row_stride` arithmetic for overflow, rounds the final u32 load up
to four bytes, and binds only the enclosing range. The binding base is rounded down to both the
device's `min_storage_buffer_offset_alignment` and u32 word alignment; WGSL receives the resulting
relative offset. Its final address must fit WGSL's u32 address space.

Artifact and MAP_READ allocations use checked word/byte arithmetic, are four-byte copy aligned, and
must fit both `max_storage_buffer_binding_size` and `max_buffer_size`. The public
`LosslessGray8Encoder::memory_plan` reports source binding, uniform, artifact storage, readback,
owned, and total addressed bytes per job. `for_in_flight(n)` reports checked aggregate bytes for a
caller-selected concurrency ceiling, while `memory_limits` exposes the relevant device limits.

The uniform, artifact, and mapped-readback allocations form one exclusive reusable buffer set.
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
LosslessGray8Encoder::submit / submit_container
    -> LosslessGray8Submission: Future<Output = Result<Vec<u8>, EncodeError>>

LosslessGray8Encoder::encode / encode_container
    -> the same submission through its blocking wait path
```

On native `wgpu`, each context owns a bounded `SubmissionPoller`, and every context clone shares its
single completion worker. `WgpuContext::from_backend` reuses the backend's worker and byte budget,
so encode, decode, and readback do not create per-submission polling threads. Poll capacity is
reserved before queue submission and saturation is a typed retryable error. A browser cannot block
`Device::poll`; its synchronous wait returns an error and callers must await.

`EncoderCapabilities::negotiate` is authoritative. A backend must only list profiles and stages it
executes. The generic API models animation and progressive plans so these can be implemented without
a later async-runtime lock-in, but `LosslessGray8Backend` reports `animation = false` and one pass.

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

The limited group packet contains the global Modular tree/context map, four prefix histograms (one
active grayscale histogram and three fixed unused-context histograms), global Modular group header,
zero transforms, then the active channel token stream. Group payload and TOC are byte aligned.

### `jwgp` acceleration index

The standard `jxlc` remains the source of truth and must decode without private metadata.
`encode_container` adds a `jwgp` box containing only a bounded, hash-bound index into those
codestream bits so the project's GPU
decoder need not first implement a fully generic JPEG XL entropy parser. Unknown-box-aware decoders,
including `djxl`, ignore it. It never stores pixels or residuals.

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

## Next implementation slices

1. Replace the single invocation with row/tile predictor scans, parallel token compaction, and a
   hierarchical histogram reduction while preserving the artifact contract.
2. Add batched small-image and multi-session scheduling so dispatch/readback overhead is amortized;
   measure isolated, sequential, concurrent, and animation workloads separately.
3. Implement multi-group Modular with a global histogram barrier and out-of-order group completion.
4. Add RGB/RGBA and native pitch-linear YUV/NV12-family ingestion through GPU color transforms.
   CUDA-only/block-linear layouts remain outside portable `wgpu` scope.
5. Implement VarDCT in GPU stages: linear-light/XYB conversion, strategy selection, forward
   transforms, adaptive quantization, coefficient tokenization, progressive split, and entropy-ready
   group packets.
6. Add animation frame headers, reference slots, blending, and persistent reference surfaces to the
   existing session API. Only then advertise animation capability.
7. Extend the existing capture/replay and codec harness with GPU timestamps, queue latency, peak
   driver allocations, and lossy quality metrics. It already separates CPU readback, continuous
   decode/encode, concurrent host fan-out, encoded size, and exact `djxl` conformance.

Until a slice is implemented and validated, capability negotiation must reject it. Benchmarks,
wrappers, and CPU oracles do not expand the advertised production capability.
