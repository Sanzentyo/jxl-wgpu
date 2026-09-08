# Encode/decode conformance corpus

The required feature matrix and the conditions for changing a capability from partial to complete
are defined in [`FULL_JPEG_XL_ROADMAP.md`](FULL_JPEG_XL_ROADMAP.md). This file records only the
corpus that is actually checked in and executable.

`tools/jxl_gpu_harness/conformance-corpus.toml` is the source of truth for multi-aspect-ratio and
multi-resolution image coverage. Every case defines an explicit expectation supported by the schema:

- `stock_gpu_round_trip`: executable today through the GPU encoder, GPU decoder, and exact CPU
  readback comparison. All 24 checked-in cases currently specify this expectation. The current
  boundary spans unsigned Gray (u8, u16), RGB (u8, u10), and RGBA (u8, u12 with opaque,
  checkerboard, and horizontal-ramp alpha) with nonzero dimensions below `2^30`, subject to adapter
  and harness memory limits. Single- and multi-group streams use the same path.
- `future_gpu_profile`: deterministic generator and inventory coverage only, representing profiles
  planned for future GPU support. While supported by the schema, there are currently zero
  `future_gpu_profile` entries in the corpus; all 24 inventory entries are active stock round-trip
  cases. When future entries are present, they stay in reports when GPU round-trip mode is selected,
  without fabricating execution results.

The checked-in inventory includes 1x1, tiny (2x2), odd (17x13, 19x11), square (64x64), portrait
(37x101, 127x509), landscape (101x37), panorama (255x31, 4097x1, 16384x1), tall (31x255, 1x4097,
1x16384), 255/256/257 group boundaries, HD 1280x720 (Gray8 and RGB8), FHD 1920x1080 (RGBA8), UHD 4K
3840x2160 (RGB10), UHD 8K 7680x4320 (Gray8), and UHD 16K 15360x8640 (Gray8).

## Incremental transport matrix

`jxl_gpu_bitstream::stream` tests raw codestreams and compact `jxlc`/`jxlp` containers at every
two-chunk split, plus byte-at-a-time signature, file-type, box-header, fragment-index, auxiliary
payload, and codestream delivery. Extended 64-bit auxiliary sizes and a size-zero `jxlc` are split
at every byte. Ordered events must reconstruct the exact codestream; auxiliary events must preserve
their original header bytes and payload. Caller-allocation identity tests prove that raw, `jxlc`,
ordered `jxlp`, and auxiliary payload tails share the supplied `Arc` rather than being copied; the
signature reconstructed across arbitrary boundaries is held inline.

The version-1 out-of-order fixture delivers the final fragment one byte at a time before fragment
zero. It must coalesce those chunks into one four-byte retained payload buffer, report exactly that
logical peak, release it when the gap closes, and emit the canonical codestream order. A five-byte
future fragment under a four-byte limit must fail with typed
`BufferedFragmentSizeLimit` before retained state survives the error. Typed count/size limits cover
the mandatory file-type box, every prefix of a real fragmented animation is either rejected or
matches the existing contiguous transport parser exactly, and a poisoned scanner rejects further
input.

`CodestreamStreamScanner` is then checked against the complete contiguous inventory at every
two-chunk split of the basic fixture and under byte-at-a-time fragmented animation, entropy-coded
TOC permutation, and out-of-order version-1 delivery. The event order is image header, frame/TOC,
physical section ranges, frame end, and finally authoritative stream end. Every reconstructed
section is compared byte-for-byte with its declared absolute range and logical TOC index. A large
VarDCT fixture proves that payload after the bounded metadata probe still shares caller `Arc`
storage and that peak prefix retention is sublinear in codestream size. Typed prefix/offset/trailing
errors, same-call statistics rollback, poisoning, and every truncated basic prefix are covered.
A libjxl-generated `--progressive_dc=2` VarDCT oracle additionally produces LF level 2, LF level 1,
and the regular frame. Both contiguous inventory and one-byte event delivery must resolve its
producer chain exactly as `[None, frame 0, frame 1]`; a synthetic missing level-1 producer must
return `InventoryError::MissingLowFrequencyFrame` before any engine sees the codestream.
A separate actual-adapter `--progressive_dc=1` fixture uses a 1024×128 patterned RGB source. It
must select the logical progressive-DC session, issue exactly two physical submissions, expose only
one visible frame through runtime-neutral async completion, and agree with Rust `jxl` within one
RGB8 code after explicit readback. The deeper `--progressive_dc=2` fixture is also actual-adapter
pixel-conformance evidence: it executes the Modular root, a single-entry intermediate VarDCT
HF-metadata/HF-global/AC continuation, and the final VarDCT frame as four physical submissions under
one logical session. Blocking and runtime-neutral async completion expose only the final frame and
must produce identical bytes within one RGB8 code of Rust `jxl`. Its non-default mode-6 DCT matrix
also proves that bounded parametric metadata replaces the resident matrix region before AC/render.
The same adapter test fills the shared byte budget after the initial staged submission and requires
the cursor-discovered entropy/order/window reservation to fail as typed `MemoryBackpressure`, then
verifies that cancellation releases every retained reservation.
A bit-level unit oracle exercises every parametric mode 0 through 6 across all 27 backend strategy
regions and compares every expanded channel value bit-for-bit with `jxl-vardct`; separate cases
require typed rejection of a transform-incompatible encoding. A 188-byte cjpeg-to-cjxl
JPEG-transcode fixture fixes raw mode 7's general Modular header and entropy cursor. An
actual-adapter test decodes its 192 DCT8 samples, runs the resident inverse schedule, checks the
16-byte status and exact packet end, then reads back the resource table to require bit-exact
`[chroma, luma, chroma] / 2040` values for all 64 raster positions and a real replacement of the
normative default matrix. A second public-decoder actual-adapter test consumes complete 264x64
4:4:4, 4:2:2, 4:4:0, and 4:2:0 cjpeg streams. It checks the exact physical component-plane bytes,
including the 4:2:0 fixture's 17x4/34x8/17x4 LF grids, channel-masked AC traversal,
component-specific resident destinations, quarter/three-quarter edge-replicating upsampling, and
encoded BT.601 YCbCr conversion. Each packed RGB8 result must remain within one code of Rust `jxl`
and optional `djxl`; there is no intermediate pixel readback. Shader coverage remains Naga
semantic validation and actual GPU execution; it does not inspect WGSL source strings.
The three additional sampling fixtures were deterministically derived from the decoded 4:2:0
source with libjpeg-turbo 3.2.0 `cjpeg -quality 90 -sample` values `1x1,1x1,1x1`,
`2x1,1x1,1x1`, and `1x2,1x1,1x1`, then losslessly transcoded by `cjxl` 0.12.0. Their containers
retain `jbrd` reconstruction data, but this decoder assertion covers the `jxlc` pixels rather than
claiming bit-identical JPEG reconstruction.
The reusable pre-restoration component primitive has a separate actual-adapter differential. It
executes horizontal, vertical, and fused two-axis interpolation into an odd 5x3 output and compares
every F32 sample with a scalar quarter/three-quarter, replicated-edge oracle. Naga also validates
both WGSL modules semantically, and compile-time plus unit checks fix the shared 32-byte,
16-byte-aligned resident `Pod` uniform. This proves the primitive and its odd-edge geometry; the
corpus still lacks a valid JPEG XL codestream that combines component subsampling with signaled
Gaborish or EPF, so it is not yet end-to-end subsampled-restoration conformance evidence.
The Modular consumer separately parses the stock lossless profile through one checked logical span
table at every possible byte split and requires identical MA-tree, histogram, hybrid-integer, and
group-range results. Its range-copy tests cross every split, reject gaps/overlaps/truncation, and
exercise unaligned zero-bit checks. Existing actual-GPU Modular tests then run through the same
span-backed bounded uploader. VarDCT separately verifies that one bounded upload segment crossing
three physical spans is byte-exact and that an out-of-range segment returns its typed execution
contract error. Its complete packet plan is identical at every byte split of the checked-in 257x17
EPF2 fixture. A custom-order 438x589 stream is then parsed from one-byte physical spans, including a
cursor-dependent HF continuation, and its coefficient-order words and final entropy cursor match
the previous `jxl_coding` reader exactly. Public event-to-engine ingestion and retained-span
backpressure/cancellation are covered separately by `gpu_decode`: every two-chunk split produces
the contiguous inventory, one-byte fragmented-container delivery reaches a custom engine as a
multi-span source, a second concurrent stream receives retryable admission without consuming its
event, and cancellation releases the exact retained-byte reservation. The actual-adapter selector
test feeds three transport chunks through the same public API for both Modular and VarDCT and checks
the existing output oracles. When `cjxl` is available, the staged local-tree test additionally proves
that source admission survives the LF map, releases after final HF submission, and releases
immediately when its pending session is abandoned. These gates complete `FRONT-03`.

The transform-metadata matrix independently covers all 42 normative RCT types, rejects type 42
with a typed error, and verifies Palette collapse/delta storage, explicit in-place Squeeze ordering,
default odd-size Squeeze expansion, meta/non-meta crossing, and portable-address overflow. One
composed RCT/Palette/Squeeze header is decoded by both the production parser and the Rust `jxl`
metadata oracle, with every wire field compared. These are parser/topology gates only; they are not
counted as GPU inverse-transform or pixel-conformance evidence for `MOD-D03`.
When `cjxl` is installed, its real 1024×128 `--progressive_dc=2` output additionally fixes the
Modular LF2 producer to one default-Squeeze transform: 13 resolved parameters, 40 entropy-visible
channels, leading 8×8/4×4/4×4 planes, no RCT, and a sample count equal to the original three full
planes. Reverse-topology tests recover data- and meta-Palette selections plus odd Squeeze sources;
an explicit work-limit case prevents repeated transforms from turning bounded channel metadata into
unbounded quadratic planning.
The shared Modular predictor tests execute ten portable wide-arithmetic operations over signed
boundary values and deterministic random inputs, including shifts at 0, 31, 32, 63, and 64 bits.
A native-`i64` oracle checks all 14 predictors, wide self-correcting intermediates, and committed
32-bit errors for 512 cases containing extreme integers and binary32 bit patterns. The Palette
execution gate also covers all implicit color components at every 1–32-bit working depth and
explicit delta reconstruction with signed extremes. Existing whole/bounded entropy and chunked
Palette tests verify the shared predictor's storage callbacks and continuation state. These gates
establish working-word arithmetic; the floating-source checkpoint below separately verifies
sample conversion and codestream delivery.
The inverse-Squeeze kernel has separate semantic and execution gates. Naga parses and validates the
WGSL module without inspecting source substrings. An actual adapter compares horizontal and vertical
odd extents plus single-pixel axes against a scalar oracle containing `i32::MIN`, `i32::MAX`, smooth
monotone runs, and wrapping reconstruction. The ABI test fixes the 64-byte/16-byte-aligned `Pod`
uniform, while malformed arena views must fail with typed geometry, reserved-word, or overlap errors.
This establishes the primitive itself, not stock-decoder scheduling or complete `MOD-D03` pixel
conformance.
The inverse-RCT primitive has a parallel gate. Naga performs semantic validation, compile-time and
unit checks fix the 64-byte/16-byte-aligned `Pod` layout, and typed validation rejects unequal,
overlapping, zero-size, out-of-range, and non-linear configurations. Concrete scalar vectors cover
all seven operations and six permutations. One actual-adapter differential executes all 42 types
over odd dimensions, padded strides, nonzero offsets, and `i32` extremes without shader-source text
inspection. Scheduler composition and production entropy input remain separate gates.
Scheduler conformance first composes an in-place horizontal split with a vertical split over both
derived channels and fixes the reverse order as vertical, vertical, horizontal. A stronger
RCT/Squeeze/RCT plan lowers to RCT type 41, three horizontal jobs, then RCT type 5. An actual adapter
executes those five jobs in one encoder and copies three noncontiguous final 9×5 planes into the sole
map; every word matches the scalar schedule with signed-extreme entropy inputs. Separate tests cover
tail-appended residual placement, a zero-width residual for a one-column image, best-fit reuse,
two-sided free-span coalescing, and typed overlap rejection. When `cjxl` is installed, the LF2 root's
13 default parameters must lower to 37 jobs and three full-resolution final plane views within a
two-times arena bound. These tests still initialize decoded entropy samples directly; connection to
the production entropy executor remains a distinct gate.
The generalized entropy descriptor matrix fixes the 32-byte `Pod` layout, cumulative decoded ranges,
and absolute metadata rebasing. A four-channel topology with one shift-mismatched plane proves that
MA references skip incompatible predecessors and retain newest-first order; property 23 emits two
reference slots while property 15 emits none. Existing every-chunk-split profile tests also compare
the appended descriptor metadata, and the full `ShaderParams` word-order test fixes its 244-byte
storage stride, MA-metadata base, and channel-layout offset without inspecting shader source text.

`wgpu_gray8::cjxl_multigroup_local_transforms_finish_each_reused_gpu_lane_exactly` creates a
515×259 RGB fixture and asks installed `cjxl` 0.12 to choose its lossless local Modular transforms.
The resulting six pass groups include local Palette work and three distinct edge geometries. The
production decoder must select descriptor reconstruction, schedule at least one inverse job per
group, retain exactly one 144-byte finalizer record per group, and return byte-exact RGB. The test
also fixes two codec submissions: one GPU execution/strict termination of the DC-global zero-symbol
entropy stream, followed by the coalesced pass-group wave. Both are validated by one aggregate
status map, so this is production transformed multi-group evidence rather than a primitive-only
kernel test.

Two adjacent `cjxl` gates cover the cross-group path. The 515×259 Gray Palette fixture must decode
nonzero DC-global samples, report one frame-resident arena and exactly one Palette dispatch, execute
one 144-byte finalizer, and match both its source and the Rust `jxl` oracle byte-for-byte. The
2051×259 `--progressive --responsive=1` Squeeze fixture declares two passes with a 2× downsampling
boundary after pass 0, exercises a DC-local MA configuration when no outer global tree exists, and
schedules two nonempty LF-group streams before its pass streams. It must report two passes, require
nonzero DC-global samples and inverse jobs with no Palette dispatch, and match source plus Rust
`jxl` byte-for-byte. Together the fixtures prove global entropy, LF/pass-subimage plane assembly,
multi-pass final reconstruction, and one frame-wide inverse; they do not cover intermediate pass
presentation or progressive frame dependencies.

`wgpu_gray8::local_ma_multigroup_codestream_reconstructs_exactly_on_gpu_and_rust` uses the public
`LosslessModularTreeMode::LocalPerGroup` encoder policy to produce a 515×259, six-pass-group stream.
The stock decoder must report six local streams, two resident configurations (global plus one
deduplicated local), nonzero metadata bytes, and Prefix coding, then return byte-exact GPU output.
The same codestream is decoded byte-exactly by the Rust `jxl` oracle. The adjacent optional `djxl`
gate runs both `SharedGlobal` and `LocalPerGroup` containers. A packed-metadata unit test appends two
different descriptor records and checks their three internal offsets are rebased while unrelated
header words remain unchanged. These tests execute the selected metadata base; they do not inspect
WGSL source strings. The encoder's streamed 16K×1 RGB8 case additionally runs `LocalPerGroup`
through multiple bounded artifact batches in blocking and runtime-neutral forms, with Rust `jxl`
and optional `djxl` exact output.

## GPU color-output matrix

`jxl_wgpu::yuv_output` treats color conversion as an executed GPU contract, not descriptor-only
coverage. An independent scalar oracle compares RGB8 codes after D65 BT.709, BT.2020, and
Display-P3 primary conversion; PQ and HLG source/target transfer; and the exact BT.2020 OETF. PQ
uses normalized absolute luminance (`1.0 = 10,000 nit`) while HLG uses scene-linear light. A
separate planar I444 readback compares BT.2020 non-constant-luminance and constant-luminance YCbCr,
including their sign-dependent chroma divisors. Every comparison executes an actual adapter and
allows at most two eight-bit codes for shader/scalar floating-point differences. Mismatched source
declarations, undefined primaries/specifications/transfers, sensor primaries, and source Gamma
without an exponent must fail with typed errors before GPU submission.

The Rust/WGSL ABI gate parses the shader with Naga and reflects the complete uniform field order.
Compile-time assertions independently fix `ImageOutputUniform` at 176 bytes and its three padded
matrix rows at offsets 128, 144, and 160. No test searches shader source text.

The same-queue display gate then consumes stored BT.2020 PQ, BT.2020 SDR-OETF, Display-P3 HLG, and
BT.2020 constant-luminance I444 inputs without a host dependency. It requires a tagged
`Rgba16Float` linear-BT.709 texture, reads its half-float texels back only for the test oracle, and
compares transfer inversion, primary conversion, alpha, luminance-contract tagging, and
constant-luminance reconstruction.
Attempting the same wide/HDR inputs with the default `Rgba8Unorm` descriptor must return a typed
error before submission. Naga semantically validates both generated storage-texture variants; the
144-byte display uniform fixes matrix-row offsets 96, 112, and 128.

## Procedural VarDCT encoder matrix

The `jxl_wgpu_encode` actual-adapter suite generates its VarDCT inputs in memory rather than
checking in large duplicate raster files. Odd 257x17, asymmetric 513x259 and 768x513, horizontal
2056x256 and vertical 256x2056 LF-boundary images exercise padded edge blocks and row/column group
ordering. The two boundary images contain two standard LF groups; the GPU artifact stores one
validated fragment descriptor per group and resets the clamped-Gradient predictor at that boundary.
Rust `jxl` and installed `djxl` must decode each emitted codestream, while the stock GPU decoder plus
explicit readback must differ from Rust `jxl` by at most one RGB8 code.

The bounded 8x8 patterned DCT8 case is also a nonzero-AC gate. The actual GPU performs the forward
transform and default-matrix quantization, emits natural-order signed tokens through one prefix
cluster shared by all 495 coefficient contexts, and supplies the AC fragment consumed by the host
packet assembler. The test requires a nonempty validated fragment and agreement among Rust `jxl`,
installed `djxl`, and the stock GPU decoder. This is evidence for the bounded one-pass DCT8 policy;
it does not cover the still-zero-AC scalable path, other strategies, adaptive clustering, or LZ77.

An additional generated 8x8 patterned case serializes non-default exact-binary16 LF
dequantization, colour factor 256, non-default X/B base correlations, and signed LF factors. The
stock frontend must recover every field exactly. The synchronous encoder and runtime-neutral
Future must emit identical bytes, the patterned fixed-kernel case must exceed 9 dB PSNR, and a
257x1 solid-red scalable-kernel case must exceed 30 dB PSNR. Rust `jxl`, the stock GPU decoder plus
explicit readback, and installed `djxl` must differ by at most one RGB8 code. Lower-level actual-GPU
probes independently read back the LF dequantization/CfL result and the per-cell HF correlation
vectors, so a parser-only round trip cannot satisfy this gate.

Exact-black 16384x1 and 1x16384 cases execute the encoder on an actual adapter with eight LF groups,
64 AC groups, and 74 TOC entries, then decode through Rust `jxl` byte-exactly. The 16384x16384 grid
is checked for 64 LF groups, 4,096 AC groups, and 4,162 TOC entries without asserting that every
adapter or configured byte budget can allocate the full-square source and artifact.

The decode integration corpus separately includes
`crates/jxl_wgpu_decode/test-data/green_queen_vardct_nonzero_ac.jxl.hex`. It is a deterministic
libjxl 0.12.0 re-encode of the checked-in 438×589 green-queen image using VarDCT effort 1,
distance 2, resampling 1, and disabled Gaborish, EPF, dots, patches, and noise. Its six nonempty
pass groups contain a custom DCT8 coefficient order and real AC coefficients. The actual-adapter
test compares GPU RGB8 against Rust `jxl` and, when installed, `djxl`, with a maximum accepted
difference of one code per channel. This fixture is decoder evidence; it is not counted among the
24 exact Modular GPU encode/decode round trips.

`vardct_engine_gpu::global_packet_and_nonzero_ac_resume_across_bounded_gpu_stream_windows` reuses
this production fixture with a 256-byte cap. Its global-tree LF/HF packet and six AC pass groups
expand into multiple ordered uploads backed by one packet stream, one AC stream, and their parameter
buffers. Blocking and runtime-neutral async results stay within one RGB8 code of Rust `jxl`; a late
mutation in the largest pass group must return typed `HfCoefficientGpu`, and abandoning a prefetched
decode must release the shared reservation after the final queue fence. Reported packet and AC
stream bytes may not exceed the cap, and the submission count must equal the initial packet batches,
planned AC batches, and resident pre/post stages without double-counting the co-submitted final
packet command.

`vardct_engine_gpu::vardct_stream_windows_adapt_to_the_shared_frame_budget` opens the same
438×589 global-tree/nonzero-AC fixture at 40-byte and 256-byte caller caps, then chooses a shared
budget strictly between those exact frame totals. Production planning must resolve a four-byte-
aligned cap below 256, report packet/AC peaks at or below it, and keep the complete planned frame at
or below the budget. Runtime-neutral async output remains within one RGB8 code of Rust `jxl`.
A second simultaneous session must expose typed non-blocking `MemoryBudgetError::Exhausted`
backpressure without consuming its source; abandoning the admitted session must drain the budget
after the queue fence, after which retrying that same backpressured session must decode the reference
pixels. A budget one byte below the exact 40-byte layout must fail at open with typed
`MemoryBudgetTooSmall` and matching required/limit fields.

`vardct_engine_gpu::combined_single_packet_resumes_across_bounded_gpu_windows` generates a patterned
32×32 DCT32x32 stream through the GPU encoder and forces its single combined LF/HF packet through a
40-byte cap. More than two LF windows retain the same state before a mapped cursor selects the HF
descriptor. HF metadata then reports the general HF-global/AC cursor, so the test exercises the
same staged path as arbitrary single-entry streams. Runtime-neutral async decode/readback agrees with Rust `jxl` and optional `djxl`
within one RGB8 code, abandoning a prefetched decode must drain the byte budget, and late-window
damage must return typed `PacketGpu(Entropy { .. })` from the final aggregate map.

The decoded fixture SHA-256 is
`95c3cd9a0769da10c1a8c0d4f903d0723bc760eebdd8023d8b7f81af5b73faa2`. It is reproduced from the
checked-in `fixtures/green_queen_vardct_e3.jxl` source with libjxl as follows:

```text
djxl fixtures/green_queen_vardct_e3.jxl /tmp/green_queen.png
cjxl /tmp/green_queen.png green_queen_vardct_nonzero_ac.jxl -d 2 -e 1 -m 0 \
  --resampling=1 --gaborish=0 --epf=0 --dots=0 --patches=0 --noise=0 --quiet
```

`crates/jxl_wgpu_decode/test-data/green_queen_vardct_mixed.jxl.hex` is a deterministic
257x257 libjxl effort-5 crop fixture. Its binary SHA-256 is
`7c9d1e134708f01842ecbf90dd1d553f792e382bc9ee3d4c77a6ef08e25eedad`. It declares LF extra
precision 1, three HF block clusters, custom coefficient orders 0 and 1, and a mixed transform
map whose actual first-block count is smaller than the 33x33 allocation capacity. The
actual-adapter test therefore covers the physical row stride of the GPU block-info channel in
addition to mixed regular/special inverse transforms. GPU RGB8 must differ from Rust `jxl` and
optional `djxl` by at most one code per channel.

It is reproduced with libjxl 0.12.0 and ffmpeg as follows:

```text
djxl fixtures/green_queen_vardct_e3.jxl /tmp/green_queen.ppm --quiet
ffmpeg -i /tmp/green_queen.ppm -vf crop=257:257:0:0 -frames:v 1 /tmp/green_queen_crop.ppm
cjxl /tmp/green_queen_crop.ppm green_queen_vardct_mixed.jxl -d 1 -e 5 \
  --epf=0 --gaborish=0 -x color_space=RGB_D65_SRG_Rel_SRG
```

`crates/jxl_wgpu_decode/test-data/green_queen_vardct_permuted.jxl.hex` is a deterministic
libjxl 0.12.0 center-first re-encode of that decoded 438x589 image. Its binary SHA-256 is
`8c3a5dd8c8b1a5d9b4934810325cb87b65a5985b322a95ecb92303ab6a529a2e`. Six pass groups are stored
in a non-row-major entropy-coded TOC permutation. The structural test verifies that each logical
group selects its original physical bit range, and the actual-adapter test executes the complete
GPU path and permits at most one RGB8 code of difference from Rust `jxl` and optional `djxl`.

It is reproduced with an explicit sRGB interpretation because PPM carries no color profile:

```text
xxd -r -p crates/jxl_wgpu_decode/test-data/green_queen_vardct_nonzero_ac.jxl.hex \
  /tmp/green_queen_vardct_nonzero_ac.jxl
djxl /tmp/green_queen_vardct_nonzero_ac.jxl /tmp/green_queen.ppm \
  --bits_per_sample=8 --num_threads=1
cjxl /tmp/green_queen.ppm green_queen_vardct_permuted.jxl -d 1 -e 3 -m 0 \
  --group_order=1 --center_x=400 --center_y=550 --epf=0 --gaborish=0 \
  --num_threads=1 -x color_space=sRGB --container=0
```

`crates/jxl_wgpu_decode/test-data/green_queen_vardct_gaborish.jxl.hex` uses the same decoded source
and encoder settings, but enables the standard Gaborish weights while leaving EPF disabled. Its
binary SHA-256 is `9b934f7367787132eb44e16698b5c0deb8f884f9bcfabe10a2a36c4c47941feb`.
The actual-adapter test verifies the parsed restoration inventory, executes inverse VarDCT,
resident Gaborish, and RGB8 packing in one GPU submission, and accepts at most one code of
difference from Rust `jxl` and optional `djxl`. It is reproduced with:

```text
djxl fixtures/green_queen_vardct_e3.jxl /tmp/green_queen.png
cjxl /tmp/green_queen.png green_queen_vardct_gaborish.jxl -d 2 -e 1 -m 0 \
  --resampling=1 --gaborish=1 --epf=0 --dots=0 --patches=0 --noise=0 --quiet
```

`green_queen_crop_vardct_epf2.jxl.hex` and `green_queen_crop_vardct_epf3.jxl.hex` are a 257x17
edge-bearing crop derived from the decoded Gaborish fixture. Their binary SHA-256 values are
`d819804cfbdd66f0ae8af4eacb481bb5cadc682162aea2796a7a8b495859fac2` and
`9034b2a4146db13220383400c65dc5949a6272dd77a76945d0987b6f2c8d53a2`. EPF2 retains the complete
standard restoration bundle; EPF3 changes the signaled iteration count and therefore executes
EPF0 before EPF1/EPF2. Both fixtures cross the 256-pixel pass-group boundary and end on partial
8x8 blocks. The actual-adapter test verifies inventory, exact restoration scratch/sigma/uniform
accounting, and at most one RGB8 code of difference from Rust `jxl` and optional `djxl`.

They are reproduced with libjxl 0.12.0 and ffmpeg as follows; the explicit color-space option is
required because PPM does not carry an ICC profile:

```text
xxd -r -p crates/jxl_wgpu_decode/test-data/green_queen_vardct_gaborish.jxl.hex \
  /tmp/green_queen_vardct_gaborish.jxl
djxl /tmp/green_queen_vardct_gaborish.jxl /tmp/green_queen.ppm --quiet
ffmpeg -i /tmp/green_queen.ppm -vf crop=257:17:91:167 -frames:v 1 /tmp/green_queen_crop.ppm
cjxl /tmp/green_queen_crop.ppm green_queen_crop_vardct_epf2.jxl -d 2 -e 1 -m 0 \
  -x color_space=RGB_D65_SRG_Rel_SRG --resampling=1 --gaborish=1 --epf=2 \
  --dots=0 --patches=0 --noise=0 --quiet
cjxl /tmp/green_queen_crop.ppm green_queen_crop_vardct_epf3.jxl -d 2 -e 1 -m 0 \
  -x color_space=RGB_D65_SRG_Rel_SRG --resampling=1 --gaborish=1 --epf=3 \
  --dots=0 --patches=0 --noise=0 --quiet
```

`testsrc_vardct_multi_lf.jxl.hex` and
`testsrc_vardct_multi_lf_skip_smoothing.jxl.hex` are deterministic 2056x256 standard VarDCT
fixtures generated by the MIT/Apache-2.0 `jxl-encoder` 0.3.1 development oracle. Their binary
SHA-256 values are `6d86b9f42ede9f2ecf13687ee4918d83e393d2bd1c135a3b1c32d97420e92e31`
and `4aec136cca138a2063df9e263552ee6c94de9e73e5328096a847f9ccdebb4d63`.
Both contain a 2048x256 LF group followed by an 8x256 tail group, nine 256-pixel pass groups, a
shared LF-global MA tree, one spectral pass, default Gaborish, and EPF1. The first enables adaptive
LF smoothing; the second sets `SKIP_ADAPTIVE_LF_SMOOTHING`. Actual-adapter tests require one codec submission, one
aggregate packet/artifact/pass-group status map, and at most one RGB8 code of difference from Rust
`jxl` and optional `djxl`.

`vardct_engine_gpu::shared_global_tree_packets_resume_across_bounded_gpu_windows` applies a
256-byte cap to the first fixture. Both LF groups must expand into more than two ordered packet
batches over one reusable upload and their shared global MA tree, without an intermediate map. The
last packet command shares the first downstream submission; exact submission accounting, the one
final aggregate status map, runtime-neutral async completion, and Rust-`jxl`/optional-`djxl`
agreement within one RGB8 code are required.

The source is the `jxl-encoder` `test_multi_group` example with `(w, h) = (2056, 256)` and its
deterministic RGB gradient unchanged. Unmodified `FrameHeader::lossy()` produces the skip fixture.
For the smoothing fixture, the development checkout changes only that constructor's `flags` field
from `0x80` to `0` before running:

```text
cargo run --release --example test_multi_group
```

These fixtures prove cross-LF-group addressing and restoration for the accepted global-tree
profile. Ordinary multi-LF-group `cjxl` output uses a different, local per-substream MA-tree
layout. `vardct_packet_gpu::gpu_stages_cjxl_local_ma_trees_without_host_image_entropy`
generates a deterministic 2056x256 RGB PPM, invokes an installed `cjxl` 0.12-compatible CLI with
distance 2/effort 7/raw-codestream output, and requires more than one LF group with no global MA
tree. On an actual adapter it dispatches every LF-local stream, maps the aggregate 64-byte status
records, validates and uses only their entropy-end cursors, packs the following HF-local metadata,
and dispatches every HF stream. The companion
`vardct_engine_gpu::ordinary_cjxl_local_trees_resume_lf_and_hf_across_bounded_packet_windows` runs the
same generated codestream through the stock frame engine with a forced 256-byte stream cap. Both LF
groups must use the 128-byte SelfCorrecting packet state and more than two ordered LF and HF
submissions, while their shared reusable packet stream allocation remains at or below the cap. Only
the final LF command copies the aggregate cursor records; after host descriptor packing, HF resumes
across its correlation, strategy/quantizer, and sharpness channels. The final HF command shares the
first downstream submission.
The test requires the typed pre-HF `UnvalidatedOutputNotSubmitted` handoff error, blocking and
runtime-neutral async RGB8 results within one code of Rust `jxl` and optional `djxl`, exact reported
submission accounting, and complete shared-budget release after normal consumption or cancellation
at the LF stage. A deterministic mutation at 90% of the first LF-group range damages a later HF
window without touching its host-parsed descriptor and must return typed
`PacketGpu(Entropy { .. })` from the final aggregate map before releasing the reservation. The effort-7 stream also exercises X=5/B=5
quant-matrix scales; a lower-level actual-GPU artifact test observes non-default scale multipliers
for all three channels directly in the resident resource vectors.

## Modular orientation and semantic header admission

`testsrc_modular_orientation_*.jxl.hex` contains 23 synthetic lossless images generated with
libjxl `cjxl` 0.12.0 on 2026-09-07. There is no external image or private transport dependency.
PGM/PPM source maxval is `(1 << bits) - 1`; samples above eight bits are big-endian u16.
RGBA sources are generated PNGs with color type 6, 8/16-bit samples, filter 0, sRGB rendering
intent 1, and zlib-compressed rows. Each source gets a minimal little-endian TIFF Exif orientation
IFD through `-x exif=orientation.exif`. The PNG-derived RGBA outputs retain the container emitted
by cjxl; the other fixtures are raw codestreams.

For coordinates `x,y` and `M=(1<<bits)-1`, source samples are:

- R/gray: `(613*x + 107*y + 43*(x XOR y)) & M`.
- G: `((153*x) XOR (271*y)) & M`.
- B: `(259*x + 307*y + 31*(x XOR y)) & M`.
- A: `M - ((181*x + 97*y) & (M-1))`; alpha remains nonzero, so invisible-color removal is irrelevant.
- Palette cases replace gray with `[2,29,113,241][(x/11 + y/7 + (x*y)%5)%4]` using integer division.

Ordinary cases use `cjxl input output -d 0 -e 1 -m 1 --container=0` with explicit
`-x color_space=Gra_D65_Rel_SRG` or `-x color_space=RGB_D65_SRG_Rel_SRG` and the Exif hint.
Palette cases use effort 9. The 2051×259 Squeeze case uses effort 9 plus `-p -R 1` and supplies
multiple LF groups and progressive passes. Hex files contain only the resulting bytes.

`native_modular_orientation_matches_sources_and_both_decoders` compares all samples, including
alpha, exactly with the generated source and Rust jxl 0.6.0. It also compares Gray/RGB color samples
with djxl's explicit-sRGB PGM/PPM output at the original sample depth; that PNM comparison excludes
RGBA alpha. The source oracle transposes row/column collections and reverses traversal independently
of the WGSL coordinate formulas. Assertions require fused output, group-local inverse/finalizer,
frame-wide inverse/finalizer, Palette, and multi-pass LF Squeeze paths to execute.

`oriented_gray_modular_preserves_all_vpi_color_and_numeric_layouts` covers all 30 VPI formats
across all twelve gray fixtures: eight orientations, two Palette extents, progressive Squeeze,
and a one-pixel output axis. Each case runs with whole blocking input and with 4 KiB GPU entropy
windows, 137-byte transport chunks, and runtime-neutral async completion. Output bytes match
between the two execution paths. Native/numeric output is exact; transformed color output differs
from the independent scalar packing oracle by at most one stored code at 8/16 bits on Apple M5.
Exact output lease sizes and full budget release are checked after each frame.

Negative metadata tests cover invalid orientations, unsupported color, associated alpha,
dimension shifts, floating-point extra metadata, extra-channel resampling, and restoration.
Independent integer alpha depths now have positive coverage below. The frontend
accepts supported metadata semantics instead of comparing a fixed image-header bit representation.
Separate bitstream tests cover small/default header forms and typed unknown image/frame/restoration
extension rejection, including empty unknown payloads and bounded extension lengths.

| Fixture suffix | Encoded extent | Depth / channels | Orientation | Binary bytes | SHA-256 |
|---|---|---|---|---|---|
| `gray_1` | 259×257 | 8 / 1 | 1 | 63726 | `70fb523dfef3933f3da7387094db515a9ef51293326d37884fa9d0d9f72c562e` |
| `rgb_1` | 259×17 | 8 / 3 | 1 | 19270 | `6c0160b2c3b321bf3cffbd9100d29ba0c09feb28958dfd090125b20c1eaed4c7` |
| `gray_2` | 259×257 | 8 / 1 | 2 | 63727 | `eca45d5c22bca2104ecd553c5d5055ad99a3a37b214f3ee147e67c3c094956d8` |
| `rgb_2` | 259×17 | 8 / 3 | 2 | 19271 | `f8b8d924696ae673e4fa40168562d20e20054ce4bd25b9700baa64e591b25c62` |
| `gray_3` | 259×257 | 8 / 1 | 3 | 63727 | `67924e83cd2ebc9642c2dba72694dee72151fc9b1708ac4fffef1d1b85228257` |
| `rgb_3` | 259×17 | 8 / 3 | 3 | 19271 | `eeab712786147d6004b084a8fed26786d083d67baf4764a420fb3242023a6ba8` |
| `gray_4` | 259×257 | 8 / 1 | 4 | 63727 | `cf1f86ec6c86e68913b2e8865b5b8c735e29a358fb9352a60c5bd6bf06412111` |
| `rgb_4` | 259×17 | 8 / 3 | 4 | 19271 | `a04851e7b5ac7ab40d94a269f5486dc908c9dd6ef1124eb4d484ac461eabed4b` |
| `gray_5` | 259×257 | 8 / 1 | 5 | 63727 | `7e19365b0ffd76c732c14cb97470165e432a79623c9d5685ea77817393ac3f11` |
| `rgb_5` | 259×17 | 8 / 3 | 5 | 19271 | `c56d972cacddeeda918f73404dbba2d88c5f9114d418a0801c592039374d5a50` |
| `gray_6` | 259×257 | 8 / 1 | 6 | 63727 | `527de4d203f1e2efc39bdc45399feca24a6bf78b1e4f463cf7c0a1b038100daf` |
| `rgb_6` | 259×17 | 8 / 3 | 6 | 19271 | `e3013d71cd2790c3cd7f0fbcb7bf4708083849caee1e1c55ef31bdbbb7acf0f7` |
| `gray_7` | 259×257 | 8 / 1 | 7 | 63727 | `805b6b25b448cf13cb4d4c758573dd44c78ad575051cb37e2c14910e4a116af8` |
| `rgb_7` | 259×17 | 8 / 3 | 7 | 19271 | `919cab8bf0ab63ce968f83d39c8c5f4e6fffa77ed0eb8bddc4cf5e202cafd0ff` |
| `gray_8` | 259×257 | 8 / 1 | 8 | 63727 | `c840399a82f60a26dfa881ca95b75fd73f772399be33670856af06420f333fb1` |
| `rgb_8` | 259×17 | 8 / 3 | 8 | 19271 | `7fd413f9b45b6590df81459f638b38bdc63abfb80bce1e445b24eb09f155a12c` |
| `gray_palette` | 515×259 | 8 / 1 | 6 | 2316 | `9136aba5d5da026ac1099ff6f2f03fae4f1bd4d66a369256f2b63b44d523acf2` |
| `gray_squeeze` | 2051×259 | 8 / 1 | 8 | 42063 | `c9d154ed2a8f7e24c1fd6c7bdd2f51867c773a91e7cb79a952172596e0066296` |
| `gray_single_palette` | 37×23 | 8 / 1 | 7 | 232 | `c91379dd6f2f4755f333ab22cdf43b1e0383696effd97b63b57f3e6371212893` |
| `rgba_16` | 259×7 | 16 / 4 | 5 | 12009 | `f2b4372158726c2887dcaf6342e01bcc53dcef9fbcb5bb7d88b5a42190f7351e` |
| `rgb_12_column` | 1×257 | 12 / 3 | 6 | 1604 | `1abacf220e0841808db21a3986de761bd7f48dd46759ebb0f7eb3e21c8e352fa` |
| `gray_row` | 257×1 | 8 / 1 | 8 | 367 | `5cdd1252953adfdd9fa7f2e7d9908919fe4abd203a80fe3c405dbc948f0f1337` |
| `rgba_8` | 17×9 | 8 / 4 | 2 | 1238 | `54fc40d367b42351c6770868195d3abd584df7e1b4ea9d8bea25e7afb297fe6d` |

This native-orientation coverage does not add Modular HDR/ICC conversion, arbitrary extra channels,
frame resampling or canvas/reference composition. F32 and Keep controls are covered below.

## Shared VarDCT color output

The decoder and render graph use one shared GPU color/layout lowering and word-owned packing
fragment. `vardct_engine_gpu/color_output.rs` reuses the checked-in depth/orientation fixtures below;
no new image provenance is introduced. `generic_color_outputs_preserve_oriented_high_depth_vardct_precision`
tests all 20 color VPI pitch-linear forms plus I444/I422/I420, NV21/NV42, P010/P012/P016,
12-bit planar I420, and linear BGRA: 30 integer layout/transfer choices. Nine F32 choices add all
eight RGB/BGR/RGBA/BGRA packing/planarity combinations in sRGB and interleaved linear BGRA.
The input is the 16-bit,
two-LF-group fixture with orientation 5, producing 17×2056 pixels. Source float RGB from Rust `jxl`
and explicit-sRGB `djxl` PFM is independently converted by the development-only scalar
`jxl_gpu_formats::convert_rgb_f32` oracle after any required SDR transfer conversion.

Every case runs whole-input blocking and fragmented-input async completion with a 256-byte entropy
cap. Exact layout metadata, shared 352-byte output/source uniform accounting, four-byte-rounded
output leases, zero unused sample bits and plane gaps, equality between upload policies, and full
budget release are required. Comparisons operate on stored sample codes rather than individual
bytes, including 16-bit words and 10/12-bit alignment. On Apple M5/Metal (2026-09-07), the maximum
difference is one code at 8–12 bits; at 16 bits it is one versus Rust `jxl` and three versus `djxl`.
The regression threshold is one at 8–12 bits and four at 16 bits (less than 0.000062 normalized).

F32 reconstruction comparisons use linear-light values, with separate encoded-error reporting.
Near black the sRGB slope amplifies reconstruction differences, so a fixed encoded threshold
would change the permitted reconstruction error with brightness. The independent shared-output
test verifies the OETF itself within 0.000002 in encoded values. On Apple M5/Metal, the high-depth
fixture has maximum linear error below 0.000005 against Rust `jxl` and 0.000073 against `djxl`;
the linear-float references themselves differ by 0.000074. The regression thresholds are 0.00002
and 0.0001 respectively. Linear `djxl` PFM is requested directly in the target color encoding.
The sRGB encoded maxima are 0.000046 and 0.000237, also recorded in test diagnostics.

`generic_color_output_combines_jpeg_gray_resampling_and_recursive_dc` repeats I420, P016, and linear
BGRA, linear F32 BGRA and planar sRGB F32 RGBA for rotated 12-bit gray with 4× resampling, a three-frame 16-bit gray DC chain, and the odd
oriented 4:2:0 JPEG transcode. `generic_color_output_converts_d65_primaries_against_djxl` requests
Display-P3/sRGB and BT.2020/BT.709 output, compares planar BGRA against `djxl` PFM explicitly
requested as `RGB_D65_DCI_Rel_SRG` and `RGB_D65_202_Rel_709`, and observes at most one code of
difference. Existing RGB8 dual-oracle cases continue to pass through the shared shader.

The packer GPU test checks 3×1 and 1×3 in every orientation with interleaved RGB and padded planar
U8/F32 RGB/RGBA. Plane starts can be unaligned and occupy a preceding plane's unused final-row tail;
payload, opaque alpha, zero padding, and untouched output guard bytes are checked independently.
Typed negative tests reject inconsistent extents/logical sizes, limited-range RGB, and PQ/HLG
without an explicit luminance mapping. Non-color numeric output, arbitrary ICC conversion, HDR
luminance mapping, broader Modular color conversion and extra-channel decoding remain separate coverage gaps.

`floating_rgb_preserves_oriented_modular_depth_and_alpha` reuses all 23 orientation fixtures with
F32 output, including RCT/Palette/Squeeze, 12/16-bit source normalization, alpha, and the Linear,
sRGB, BT.709 and BT.2020 transfer functions. Source-formula float error is below 0.000002 and
whole versus 4 KiB fragmented async output is byte-identical. `floating_frame_sequences_can_keep_codestream_coordinates`
reuses all nine accepted frame sequences with `OrientationPolicy::Keep` and F32 RGBA: output
geometry is unrotated, timing/dependency metadata is unchanged, and a row/column traversal inverse
of the independently decoded Rust frames checks pixels. Requantized Modular samples are exact;
VarDCT remains within one source code. This includes mixed JPEG/Modular frames and recursive DC.

`floating_rgb_normalizes_every_integer_modular_source_depth` generates 48 tiny GPU-encoded
257×3 Gray/RGB/RGBA cases: every depth from 1 through 16, alternating raw/container framing,
nonconstant samples with both range endpoints, and an odd encoder source pitch. Rust `jxl`
first verifies the integer source exactly. Bounded fragmented GPU decode then checks F32 RGBA
against source-depth normalization within 0.0000001, including independent alpha.

Render-backend tests cover all eight F32 RGB packing forms, finite negative/greater-than-one
values, exact identity-color preservation, D65 primary conversion, and float display with
unaligned rows/planes and independent alpha. F32 display requires `Rgba16Float`; RGB transfers
do not affect alpha. These tests introduce no new checked-in codestream fixtures and do not satisfy crop,
reference retention, associated-alpha or floating-point source conformance.

## VarDCT integer source depths

Twenty synthetic fixtures under `crates/jxl_wgpu_decode/test-data/` exercise XYB input at every
integer depth from 1 through 16, with RGB8 output. The binary P6/P5 source header declares
`MAX = (1 << bits) - 1`. Samples occupy one byte at depths 1–8 and a big-endian two-byte word at
depths 9–16. Zero-based coordinates generate the following values; gray uses `R`:

```text
R = (613*x + 107*y + 43*(x XOR y)) & MAX
G = ((153*x) XOR (271*y)) & MAX
B = (259*x + 307*y + 31*(x XOR y)) & MAX
```

libjxl 0.12.0 runs `cjxl source.pnm output.jxl -d 2 -e 7 -m 0 --container=0 -x
exif=orientation.exif` with the minimal TIFF orientation described below, plus these options.
Every `rgb_N` fixture has two spatial groups, three spectral passes, and options
`--progressive_ac --progressive_dc=0`. Extents precede orientation.

| Suffix after `testsrc_vardct_depth_` | Bits | Source extent / orientation / options | Binary SHA-256 |
|---|---:|---|---|
| `rgb_1.jxl.hex` | 1 | 257×33 / 1 / spectral | `65aecbc37e485bf399c612547d9752596c9ac82dab4d1349c6ccb83b295430f9` |
| `rgb_2.jxl.hex` | 2 | 257×33 / 1 / spectral | `846d800161accbfdddc5a0b198b77732f90f9f2ec0e3dfa9607e0d9f95f04b7c` |
| `rgb_3.jxl.hex` | 3 | 257×33 / 1 / spectral | `b48984eacce71b84ff52d1a88e27235ef785b0d2a6195bab4c8160ff78459437` |
| `rgb_4.jxl.hex` | 4 | 257×33 / 1 / spectral | `51f651e83042b6fcf347966566f243f0bebc083fed75b9413ef143199b2aa64b` |
| `rgb_5.jxl.hex` | 5 | 257×33 / 1 / spectral | `d9f88d6ee9680058b4388e774167a9a389ddb39dd076e0520e6d55b8e7881b39` |
| `rgb_6.jxl.hex` | 6 | 257×33 / 1 / spectral | `735400c4a2a6bdc608187726bee5941e5b5c528ec26717c2c36e9964701a76ed` |
| `rgb_7.jxl.hex` | 7 | 257×33 / 1 / spectral | `7fc3f9fd5ac605fc7a3d6156a6d7fe5f9f761ce3969eb9aef2348d8162c9d13c` |
| `rgb_8.jxl.hex` | 8 | 257×33 / 1 / spectral | `7f522c8d2cb4eb897a421e0fbe20675f8f29528214d8d36b9656be9d6775fd39` |
| `rgb_9.jxl.hex` | 9 | 257×33 / 1 / spectral | `44977b41815ee2c3cd9045bffa3eaa20ce925ee2af317ccfe9e4314a62efbd08` |
| `rgb_10.jxl.hex` | 10 | 257×33 / 1 / spectral | `016ab61e04c3d9721591a933f3d54980f5ae2d7e40e0fb3ef002173372edf509` |
| `rgb_11.jxl.hex` | 11 | 257×33 / 1 / spectral | `06a029f625e2d0ac83061e310552e343afdd99d824dc0f12232a54dcde62e8ee` |
| `rgb_12.jxl.hex` | 12 | 257×33 / 1 / spectral | `5f26c7f3530bc698e5369c6afffe3ee27e9c611b6d5d8f1f6627eb512e05aaf6` |
| `rgb_13.jxl.hex` | 13 | 257×33 / 1 / spectral | `3d444138bb812a292e352fbabc31a29119e4a77081577c4c6b5a0be1539a635f` |
| `rgb_14.jxl.hex` | 14 | 257×33 / 1 / spectral | `2b93c518d74062748a0d79cae02f86ce02b87b208df25a34b023a1b38e97435c` |
| `rgb_15.jxl.hex` | 15 | 257×33 / 1 / spectral | `d7356549f5177e5c8cfe64b191434ea01fb484bec78361552d56170ad98edcd7` |
| `rgb_16.jxl.hex` | 16 | 257×33 / 1 / spectral | `eb7bca07295ce574b65bfe9983481887fbe6d8e252311ca407d7409f273262c2` |
| `gray_12_upsample.jxl.hex` | 12 | 515×259 / 8 / `--resampling=4 --progressive_ac --progressive_dc=0` | `6cfb82a6cef26f6b141454882bbeb581fb788d65a4d1844bab1c4bf6954adfda` |
| `gray_16_dc.jxl.hex` | 16 | 1024×128 / 6 / `--qprogressive_ac --progressive_dc=2` | `2751fd92259cb0bce58e33d6655761445ccea41fe897e10d8e3358323632aee4` |
| `rgb_16_multilf.jxl.hex` | 16 | 2056×17 / 5 / `--qprogressive_ac --progressive_dc=0` | `d88119a3dc7a561130a52c0bb6136c08f898f9947229748d137cd450845009da` |
| `rgb_16_single.jxl.hex` | 16 | 17×9 / 7 / `--progressive_dc=0` | `f59ff8deca38827921476f2fe3146e6fe60b1f8add15951dfd3353faacf2846c` |

`every_integer_depth_through_sixteen_decodes_xyb_to_rgb8_on_gpu` verifies the declared integer
depth and per-pass shifts for all sixteen sources.
`high_integer_depths_combine_with_grayscale_orientation_resampling_and_dc` additionally covers
a rotated 4× grayscale frame, a three-frame grayscale DC chain, two LF groups, and a single-entry
TOC. Both tests run whole-input blocking and fragmented-input async decoding with 256-byte entropy
windows. The upload policies must agree byte-for-byte, output extents must account for orientation,
gray pixels must have equal RGB channels, and all memory reservations must be released.

On Apple M5/Metal (2026-09-07), every case differs from Rust `jxl` and installed `djxl` by at most
one RGB8 code. The `djxl` comparison uses explicitly sRGB float PFM, reads its declared endianness,
reverses bottom-first rows, and rounds/clamps each float to RGB8 once. This avoids PNM's original-depth
quantization, which would discard the lossy reconstruction precision for low-depth sources.
The profile test checks typed rejection of out-of-range XYB and non-8-bit YCbCr depths while
retaining the original depth and color transform. These fixtures do not claim integer input above
16 bits, floating-point source metadata, non-8-bit YCbCr input, or non-RGB8 VarDCT output.

## VarDCT grayscale and orientation

Fifteen synthetic fixtures in `crates/jxl_wgpu_decode/test-data/` exercise packed RGB8 presentation.
The source is binary P6 RGB or P5 grayscale with `255` as its maximum. RGB uses the same formula as
the spectral-pass fixtures; gray is `(13*x + 7*y + 3*(x XOR y)) & 255`. All coordinates are zero-based.
For XYB, libjxl 0.12.0 runs `cjxl source.pnm output.jxl -d 2 -e 7 -m 0 --container=0 -x
exif=orientation.exif` plus the options below. The minimal Exif payload is a little-endian TIFF
containing one SHORT orientation tag, generated by this Ruby expression (`orientation` is 1–8):

```ruby
'II' + [42, 8, 1, 274, 3, 1, orientation, 0, 0].pack('vVvvvVvvV')
```

Each `orientation_N` source is 257×17 RGB with `--progressive_ac --progressive_dc=0`, two spatial
groups, and three passes. Its codestream orientation is `N`; output dimensions transpose for 5–8.
Gray sources use the dimensions and options below. Extents listed here precede orientation.

| Suffix after `testsrc_vardct_` | Source extent / orientation / options | Binary SHA-256 |
|---|---|---|
| `orientation_1.jxl.hex` | 257×17 / 1 / spectral | `58e4134c2d99161f3f727d6d74136f298b841b05ae22397ca5f926b622f979a4` |
| `orientation_2.jxl.hex` | 257×17 / 2 / spectral | `670c5ecb732bb774b621f63407593ce370728458618006a173d602fffa91edb3` |
| `orientation_3.jxl.hex` | 257×17 / 3 / spectral | `8c4c84332f8db294eed8f9ab5dbb5b2b4f0fa1918478213d93c39be1f0572d97` |
| `orientation_4.jxl.hex` | 257×17 / 4 / spectral | `8178a962a8a0335aaa5df4b9f04a742a3f27461d0e1edcfc63e9d68356af1ae5` |
| `orientation_5.jxl.hex` | 257×17 / 5 / spectral | `a443a75e3f4110678bff70e715796518036abb1889917e53ae5a2cb099a35106` |
| `orientation_6.jxl.hex` | 257×17 / 6 / spectral | `4b6bc3e56dd797d038f3aecdaccd13c6ded4cc7ee45d909a7f4d4b7403c7c90f` |
| `orientation_7.jxl.hex` | 257×17 / 7 / spectral | `4452a5b2cc75c7dbdc4ee98a4a5eb94e7054d9dcb3bf6e48ef7a9f0b65f7bb2e` |
| `orientation_8.jxl.hex` | 257×17 / 8 / spectral | `81546604589b657af2df8814224df23126e182d80c1942c364b60d1c14a3484c` |
| `gray_single.jxl.hex` | 17×9 / 3 / `--progressive_dc=0` | `483003ae87b9245aeaf86ec6bb0887a5551b35904b4290427c4d5f14099e6ab6` |
| `gray_progressive.jxl.hex` | 257×129 / 1 / `--qprogressive_ac --progressive_dc=0` | `6fb62f25e384aaaa3ddfa7245945ae135609f8baee873cf5f14dffd5eac93e1b` |
| `gray_upsample.jxl.hex` | 515×259 / 8 / `--resampling=4 --progressive_ac --progressive_dc=0` | `222f3d3afdbd849e9b53fef308abbde1d1d170307dbc8707ccfc6f3e7e80d1c8` |
| `gray_multilf.jxl.hex` | 2056×17 / 5 / `--progressive_ac --progressive_dc=0` | `b3ba071fd8f515b77c9c6268d213164b420e32958aca7818d81b1df2f2cd835d` |
| `gray_dc_ac.jxl.hex` | 1024×128 / 6 / `--qprogressive_ac --progressive_dc=2` | `3a6a8b04fc0f26f70faa6ed44ce5132f652914dc71960177306eff2aa14cdf61` |
| `gray_jpeg.jxl.hex` | 173×101 / 8 / JPEG transcode | `511053e27bc30c7e0b507226ee34796d79a6ee219cdc7e007be19219a8fd6a23` |
| `jpeg_orientation_6.jxl.hex` | 173×101 / 6 / RGB 4:2:0 JPEG transcode | `df0807f3953e70a5581d1c7361995bd99ab8c6ea7cf1bbe3f6395d010feb7474` |

For the JPEG cases, libjpeg-turbo 3.2.0 runs `cjpeg -quality 90 -outfile source.jpg source.pnm`
(default 4:2:0 for RGB). An APP1 segment immediately after SOI contains `Exif\0\0` followed by the
TIFF payload above. `cjxl source.jpg output.jxl --lossless_jpeg=1 --allow_jpeg_reconstruction=0
--container=0` imports that orientation. The checked-in raw codestream concatenates the container's
ordered `jxlp` payloads after their four-byte indexes; auxiliary metadata boxes are omitted.

Actual-GPU tests `all_eight_orientations_normalize_multigroup_progressive_output_on_gpu`,
`grayscale_presentation_combines_orientation_resampling_and_progressive_dc_on_gpu`, and
`jpeg_subsampling_is_expanded_in_codestream_coordinates_before_orientation` run all fixtures through
public `GpuDecoder`, both whole-input blocking and fragmented-input async with a 256-byte entropy
window. Exact output extents and byte counts are checked. Both Rust `jxl` and installed `djxl`
agree within one RGB8 code on Apple M5/Metal (2026-09-07). The two upload policies must agree
byte-for-byte and release all reservations. Every grayscale GPU pixel has equal RGB channels.
The recursive grayscale root verifies that a gray presentation still retains three internal XYB
dependency planes. A separate packer GPU test covers 3×1 and 1×3 in every orientation and verifies
zero padding across all three word phases.

The JPEG cases also prevent two regressions: staged HF metadata and AC traversal must include the
MCU-padded block grid (22×14 for the 173×101 4:2:0 image), and later host matrix uploads must preserve
the nondefault raw DCT8 matrix already decoded on GPU. A resource-range unit test additionally
checks all transposed/AFV aliases and prevents uploads into LF data or the AFV basis. These tests
do not by themselves establish arbitrary output-format or full JPEG XL conformance. Modular
orientation, Keep controls and shared color outputs have separate coverage above.

## VarDCT frame upsampling

Seven libjxl 0.12.0 codestreams live under `crates/jxl_wgpu_decode/test-data/`. They use the same
P6 header and deterministic RGB formula documented in the spectral-pass section below. Each
command is `cjxl source.ppm output.jxl -d 2 -e 7 -m 0 --container=0 --progressive_dc=0
--resampling=FACTOR`. The 4× case additionally uses `--progressive_ac`. The custom 8× case uses
`--upsampling_mode=0 --already_downsampled`; its 73×57 PPM is already at the encoded resolution,
so the presented extent is 584×456. Other PPM dimensions equal their presented dimensions.

| Suffix after `testsrc_vardct_upsample_` | Presented extent | Evidence | Binary SHA-256 |
|---|---|---|---|
| `2.jxl.hex` | 515×259 | 2×, two spatial groups, odd edges | `e59711816d091783e1e625938f16d883a3b60bf8330bfc55f849a545f3d526ed` |
| `4.jxl.hex` | 1027×133 | 4×, three spectral passes, two groups | `94133d7f1a59718834de6fbb2363568f8114107008cb41dd9030b37253e44028` |
| `8.jxl.hex` | 2053×67 | 8×, odd edges and two groups | `3105bb30f464f80f0dfe50c26d5f18a8a27587cd1a4489f7981671a4dc3893cd` |
| `8_custom.jxl.hex` | 584×456 | Custom nearest-neighbor weights, single-entry 73×57 encoded grid | `ce9b47e9e3070207f59f21fb8c6a2de8e263bf190fc380a1a70eadcb23143c87` |
| `2_multilf.jxl.hex` | 4111×17 | Two LF groups and nine pass groups | `c1d7d5cd7538cb5ba8ae3daba3784672216ecd1e2f331d248310a50209348896` |
| `4_thin.jxl.hex` | 17×1 | Mirrored one-sample vertical axis | `86fb6036f6a357ca31bd9386e0deb95704897d85e41329591b9c8ecc6130866e` |
| `8_single.jxl.hex` | 7×5 | One encoded sample on each axis | `3b43087728f340f29829cf286778d2916d5edaff8d6eb8718d38e005d9a0514d` |

`frame_upsampling_uses_header_weights_and_presentation_extent_on_gpu` validates factors, coded
and presented extents, nondefault custom weights, and exact plane/weight/uniform memory costs.
It checks whole-input blocking output and fragmented-input async output with a 256-byte GPU cap
against both Rust `jxl` and optional `djxl`, allowing at most one RGB8 code of error. Both upload
policies must produce identical bytes and release all reservations. A budget that could hold only
the three upsampled planes must be refused before submission. The tests ran on Apple M5/Metal on
2026-09-07. The compact-kernel unit test compares every reflected phase against an independently
constructed symmetric matrix and rejects invalid factors, truncated weights, and non-finite
weights; Naga validates the reused WGSL. Profile tests fix the exact LF/pass-group boundary in
encoded coordinates and reject invalid factors.

Single-entry streams now use bounded LF/HF-metadata staging and the general HF-global/AC parser.
Their image dimensions impose no uniform-transform assumption. These fixtures establish ordinary
VarDCT color resampling coverage; Modular and extra-channel resampling remain separate gaps.

## Spectral and quantized VarDCT passes

The following synthetic fixtures are checked in under `crates/jxl_wgpu_decode/test-data/` as
hex-encoded raw codestreams. They were generated with libjxl `cjxl` 0.12.0. Each P6 PPM has header
`P6\n{width} {height}\n255\n` followed by row-major RGB8 samples, with zero-based coordinates:
`R = (13*x + 7*y) & 255`, `G = ((3*x) ^ (11*y)) & 255`, and
`B = (5*x + 17*y + (x ^ y)) & 255`.

| Fixture suffix after `testsrc_vardct_progressive_` | Extent | Pass/topology evidence | Binary SHA-256 |
|---|---|---|---|
| `spectral.jxl.hex` | 257×129 | Three unshifted AC passes, two spatial groups | `b5246f6478e45c034805eb7fea968e08e65e0406cb98f75d9e443c431eebafe4` |
| `quantized.jxl.hex` | 515×259 | Two AC passes with shifts `[1, 0]`, six spatial groups, center-first TOC | `25558bce86d71acd96ba17a41d365c2059f5ae86020c0abe5637d1b51edd0cba` |
| `multilf.jxl.hex` | 2056×17 | Three unshifted AC passes, nine spatial groups across two LF groups | `4edf4d9627d0ee7909116007822ab37feece908af9b95a064d32dc2f6a814ed6` |
| `dc_ac.jxl.hex` | 1024×128 | Recursive DC levels 2→1→0, global-only Modular root, two final AC passes | `8ab79d20f86bfc8b1349361927f13415d5669d1e98125df2b8b3673fdca04c7b` |

```console
cjxl spectral.ppm spectral.jxl -d 2 -e 7 -m 0 --container=0 --progressive_ac --progressive_dc=0
cjxl quantized.ppm quantized.jxl -d 2 -e 7 -m 0 --container=0 --qprogressive_ac --progressive_dc=0 --group_order=1 --center_x=400 --center_y=200
cjxl multilf.ppm multilf.jxl -d 2 -e 7 -m 0 --container=0 --progressive_ac --progressive_dc=0
cjxl dc_ac.ppm dc_ac.jxl -d 2 -e 7 -m 0 --container=0 --qprogressive_ac --progressive_dc=2
```

`progressive_ac_passes_accumulate_with_independent_tables_on_gpu` checks that distinct per-pass
entropy tables are retained, that every pass receives separate status/resume storage, and that
whole-range blocking and 256-byte-window async execution produce identical RGB8. The latter feeds
37-byte transport chunks through the public streaming decoder, crossing HF-global descriptors and
coefficient packets. `progressive_ac_combines_with_recursive_gpu_resident_dc` checks the fourth
fixture with both window policies and only one visible final frame. Both tests compare every output
sample with Rust `jxl` and installed `djxl`, accepting at most one code of difference; all cases ran
on Apple M5/Metal on 2026-09-07. These are decoder correctness results, not encoder quality or speed
measurements.

`progressive_ac_late_corruption_and_cancellation_release_all_pass_storage` corrupts the last AC
pass's largest packet and requires typed `HfCoefficientGpu` failure without an authoritative frame.
Dropping either failed or prefetched work must release the complete shared reservation. The parser
unit suite additionally repeats a real HF pass descriptor eleven times, preserves shifts 0–3, and
rejects truncation inside the final descriptor. That is parser boundary evidence; eleven-pass image
conformance and intermediate progressive presentation remain separate gaps.

## Bounded Modular stream-window matrix

`wgpu_gray8::fixed_gradient_group_resumes_across_bounded_gpu_stream_windows` creates a standard
193×97 lossless Gray8 codestream with alternating long runs and high-entropy regions through the
production GPU encoder. Rust `jxl` must reproduce the source first. The production decoder is then
given an explicit 256-byte stream-window cap, forcing one channel-fixed Gradient group through
multiple ordered submissions. Its blocking and runtime-neutral async results must both be byte
identical to the source, the reported peak stream allocation must not exceed the cap, and the
submission count must prove that more than one segment executed. A third submission is abandoned;
after a queue fence and callback polling, the shared byte reservation must return to zero.

`crates/jxl_wgpu_decode/test-data/testsrc_modular_weighted.jxl.hex` is a checked-in 193×197 RGB8
raw codestream produced by libjxl 0.12.0. Its binary SHA-256 is
`2c76b3c36ebc6a0c3f6b2107ab0978119d04e08180c43c59ada37a9804fa2442`; the source PPM SHA-256 is
`ed91b02ce3acaa1077a8f184379fc3a37bc4063bff1c447113c81100350a5497`. The profile disables
palette, squeeze, patches, and color transforms other than YCoCg while selecting learned MA and
Weighted prediction:

```text
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i "testsrc=size=193x197:rate=1" -frames:v 1 -pix_fmt rgb24 weighted-single.ppm
cjxl weighted-single.ppm weighted-single.jxl \
  -d 0 -e 9 -m 1 -I 100 -C 6 -g 1 -P 6 -E 0 \
  --modular_palette_colors=0 -X 0 -Y 0 -R 0 --patches=0 \
  --container=0 --num_threads=0 \
  -x color_space=RGB_D65_SRG_Rel_SRG --quiet
```

`wgpu_gray8::weighted_ma_groups_resume_across_bounded_gpu_stream_windows` first requires the
production parser to report `ModularEntropyCoding::Ans`, generic MetaAdaptive reconstruction with
SelfCorrecting prediction, and the exact 112-byte lane state. A 256-byte stream cap forces more
than two ordered submissions. Blocking and runtime-neutral async paths must both match the Rust
`jxl` integer oracle byte-for-byte, and abandoning a submitted frame must release the shared byte
reservation after its completion fence. The test then preserves more than the first two upload
windows, destroys the late ANS tail, and requires a typed
`ModularEntropyRejected { group_index: 0, .. }` rather than partial output or a host pixel fallback.

`wgpu_gray8::every_multigroup_gpu_status_is_validated_from_one_map` uses the same 256-byte cap on a
513×257 multi-group stream, leaves the first 512 bytes of group 1 intact, and corrupts its remaining
entropy bytes. Bounded host metadata must still open; the final aggregate status map must return the
typed `ModularEntropyRejected { group_index: 1, .. }` error. Later segment dispatches therefore
cannot overwrite an earlier sticky GPU failure with a successful status.

Host scheduling tests independently use an unaligned three-bit group start and a 64-byte cap to
verify physical/logical mapping, first/final flags, 16-byte overlap, monotonic yield boundaries,
one-lane scratch isolation, and exact peak bytes. Budget tests show that lane count and stream peak
trade against the same per-frame target. A cap below the 40-byte minimum is a typed error, while
every oversized accepted Modular group is segmented. Rust/WGSL full-record word casts pin the
244-byte parameter record and the 32/48/112-byte, 16-byte-aligned resume layouts. Every composed
shader variant is parsed and semantically validated with Naga; no shader-source substring
assertion is used.

Together these are Prefix+RLE/LZ77, production ANS+Weighted, combined single-entry,
shared-global-tree and staged local-tree VarDCT LF/HF packets, and nonzero/custom-order VarDCT AC
cross-window evidence. Recursive entropy consumers still need the same bounded resume contract,
and broader corruption/truncation fuzzing is still required before `ENT-D02` can be marked done.

## Common entropy differential matrix

The `jxl_wgpu_decode` unit suite assembles the same `modular_entropy.wgsl` fragment used by the
stock Modular, VarDCT packet, and HF pass-group pipelines into a test-only actual-GPU probe. It
compares decoded values with the Rust metadata entropy cursor for canonical Prefix codes and all
four standard ANS histogram encodings (unary, binary, flat, and compressed), exercising both direct
and alias-table buckets, hybrid integers, and LZ77 copies. Negative GPU cases require the exact
truncated-input, invalid-final-ANS-state, nonzero-padding, and overlong-padding status codes. The
probe shader is parsed and semantically validated by Naga even when an adapter is unavailable; no
shader-source substring assertion is used.

## HF block-context differential matrix

The VarDCT unit suite composes the production `vardct_block_context.wgsl` fragment into an
actual-GPU probe. A synthetic valid table covers negative and positive LF thresholds, values equal
to a threshold (the comparison is strictly greater-than), three quant-field segments, every order
channel, and distinct coefficient order IDs. GPU-selected map entries must equal an independent
Rust implementation of the normative X/B/Y LF folding order. Naga parses and semantically
validates the same fragment before adapter discovery; the test does not inspect shader text.

## Frame sequences, mixed modes, and coalescing

`tests/wgpu_gray8/frame_sequence.rs` executes nine positive fixtures on the stock mode-neutral
GPU engine. `test-data/generate_frame_sequences.c` is the offline libjxl 0.12.0 generator and is
never linked to production. Its integer source formulas, encoder options, exact names and timecodes
are checked in. All animations use 30000/1001 ticks per second; loops are three except `modular_many`
(infinite). Odd-indexed source frames have `3*i+1` ticks; even frames have zero, including the final
frame. `layered_still` omits the animation header and presents only the fifth Replace layer.

| Fixture suffix | Encoded extent | Color/depth/orientation | Physical source frames |
|---|---|---|---:|
| `modular_gray` | 259×17 | Gray8, 6 | 5 |
| `modular_rgb12` | 257×9 | RGB12, 8 | 5 |
| `modular_rgba16` | 33×7 | RGBA16, 2 | 5 |
| `modular_many` | 1×9 | Gray8, 5 | 17 |
| `layered_still` | 37×13 | Gray8, 7 | 5 |
| `vardct_rgb` | 257×33 | XYB/RGB8, 6 | 5 |
| `vardct_gray` | 259×17 | XYB/Gray8 presented as RGB8, 8 | 5 |
| `mixed_jpeg_modular` | 259×17 | YCbCr 4:2:0 VarDCT and RGB8 Modular, 1 | 5 |
| `vardct_dc` | 1024×128 | XYB/RGB8, 6; each source adds DC2 dependencies | 3 |

The mixed input JPEG is generated from the generator's RGB8 formula at source index 1 with
libjpeg-turbo 3.2.0 `cjpeg -quality 85 -sample 2x2`; pass it as the generator's optional second argument
(after the output directory). Physical source indices 1 and 4 use `JxlEncoderAddJPEGFrame`; the
others use lossless Modular image frames. No JPEG reconstruction metadata is requested.

Rust `jxl` and `djxl --output_frames` independently decode every presentation. Modular samples
match exactly, including alpha and 12/16-bit valid codes; VarDCT differs by at most one RGB8 code
on Apple M5/Metal. Whole blocking and 4096-byte entropy-window/137-byte fragmented async output
are byte-identical. The tests check orientation-normalized session/output extent, source-indexed
UTF-8 name and timecode, exact tick accumulation, output count/finality, shared input release,
multiple prefetched Modular frames, retryable memory pressure (including DC roots), cancellation
during staged submission, and output clones surviving session drop. Native RGB packing accepts
an explicitly matching sRGB/BT.709/full-range descriptor so both coding modes share one output
contract; it does not accept a different transfer or relabel converted pixels.

`rejected_crop` and `rejected_add` retain their original historical filenames. They now execute
through GPU composition and match both Rust `jxl` and `djxl`. Dependency prefetch is checked
explicitly, and reference-version/malformed-timecode tests continue to exercise the common plan.
The larger composition corpus below covers real reference retention and additional blend modes;
neither corpus alone establishes full JPEG XL conformance.

| File | Decoded hex bytes | SHA-256 of binary codestream |
|---|---:|---|
| `sequence_layered_still.jxl.hex` | 2650 | `87d13a10326aa0c093862dc2d425543e79b95117b6206c8baad76a6c6e3eea88` |
| `sequence_mixed_jpeg_modular.jxl.hex` | 46458 | `eb85c7eeae62b9d2bc7373e40d988bc8e92661c9a34d53f7fc7f476d8028d6ef` |
| `sequence_modular_gray.jxl.hex` | 16368 | `2e6da5ba1e93f4e7769bf0ba4fc82fa70479ad878e91dd5d8a7079fe5be03680` |
| `sequence_modular_many.jxl.hex` | 772 | `abb7e6e8786bc944a0dd98e94dd6d6d6d3b026204df650d9776577c013c002a9` |
| `sequence_modular_rgb12.jxl.hex` | 45715 | `1363ec500429175e16530362d735e8f87eff17ffc4a040039f2d02f26ba19b67` |
| `sequence_modular_rgba16.jxl.hex` | 5808 | `16f5ca14c7576b1347f232b1c0d97d7cea80686a5c939dbdeead9b18ed5605aa` |
| `sequence_rejected_add.jxl.hex` | 676 | `cb24f1b73c3995f382fcb0836f8607063f705e7cd7490d51aa597fda591fd243` |
| `sequence_rejected_crop.jxl.hex` | 681 | `c18e071a2fd7890006343025b3d1128e86a5007f17e2177feb205c08b0360aee` |
| `sequence_vardct_dc.jxl.hex` | 309711 | `4174e65e2235424b31e880d579a5a57e8286972cf388fb191b9caf9de81bb348` |
| `sequence_vardct_gray.jxl.hex` | 12783 | `bec8483bd664976e9b4d0acf85ab59388e522a7b5a58ea9ffbb5a1a703116eba` |
| `sequence_vardct_rgb.jxl.hex` | 41837 | `547b047f5241080f69d9ec2821291ed528deddd16049f319329cf2ea2138f6da` |

## Deterministic source contract

Every case describes:

- `source.model`: `gray`, `rgb`, or `rgba`;
- `source.depth`: `u8`, `u10`, `u12`, or `u16`;
- `source.alpha`: `none` for non-alpha images, or an `opaque`, `checkerboard`, horizontal-ramp, or
  vertical-ramp contract for RGBA;
- `row_layout.alignment`, `extra_padding`, and `padding_byte`;
- a fixed `pattern.seed` interpreted by generator schema version 1.

Samples above eight bits use little-endian canonical in-memory storage. PGM/PPM/PAM writers swap
those samples to the network-order representation required by Netpbm. Reports distinguish two
BLAKE3 hashes:

- `input_hash` covers the complete pitch-linear storage, including deterministic padding;
- `pixel_hash` covers active, interleaved samples only, in the canonical little-endian order.

This makes stride or padding regressions visible without confusing them with decoded-pixel
equality. The stock round-trip path uploads the complete padded storage with an explicit
`ImageLayout` row stride, so the real GPU encoder consumes the declared pitch. The GPU decoder's
active rows must then equal the generator's `pixel_hash`.

## Bounded generation

`LazyImage` validates every multiplication with checked arithmetic and exposes an iterator that
allocates one row at a time. `--max-row-bytes` defaults to 64 MiB and rejects a descriptor before
allocation when its padded stride exceeds that limit. Total logical and storage sizes remain
64-bit metadata; a UHD case is never collected into one `Vec` by the corpus API. Hashing and file
generation are therefore O(row stride) in resident image memory.

## Commands

Inventory all cases and hashes:

```console
cargo run -p jxl_gpu_harness -- conformance --action inventory \
  --output /tmp/jxl-conformance-inventory.json
```

Run exact GPU encode/decode/readback for the checked-in stock cases:

```console
cargo run -p jxl_gpu_harness -- conformance --action gpu-round-trip \
  --output /tmp/jxl-conformance-gpu.json
```

Select cases with repeated or comma-separated `--case` values. Unknown or duplicate selections
are configuration errors.

## Development-only standard fixtures

The external path is intentionally outside all production codec crates. It adds no Rust CPU codec
dependency and cannot replace an unavailable GPU result. Without `--apply`, it only reports the
planned paths:

```console
cargo run -p jxl_gpu_harness -- conformance --action external-fixtures \
  --case tiny-gray8-2x2,hd-rgb8-1280x720 \
  --fixture-dir /tmp/jxl-reference-fixtures
```

To execute installed libjxl tools:

```console
cargo run -p jxl_gpu_harness -- conformance --action external-fixtures \
  --case tiny-gray8-2x2 \
  --cjxl /opt/homebrew/bin/cjxl --djxl /opt/homebrew/bin/djxl \
  --fixture-dir /tmp/jxl-reference-fixtures --apply
```

The harness streams a PGM or PPM source, calls lossless `cjxl`, decodes with `djxl`, parses the
binary PNM output row by row, and verifies extent, channels, maximum sample value, and exact pixel
hash. RGBA entries are kept inventory-only in this external path until a portable alpha-bearing
fixture transport is selected. Existing outputs are not overwritten unless `--force` is supplied.

## GPU crop/blend composition and resident references

`test-data/generate_frame_composition.c` uses libjxl 0.12.0 only as an offline fixture generator.
`tests/wgpu_gray8/frame_sequence.rs` and its `composition.rs`/`reference.rs` children exercise the
production GPU frame executor. The original integer source formulas, crop rectangles, slot
assignments, blend modes, alpha endpoints, names, timecodes and durations are in the generator.
All animations use 30000/1001 ticks per second and two loops; the layered still omits animation.
Nine physical source layers include zero-duration layers and six presentations (one for the
layered still). Recursive DC adds two hidden LF producers to each of three full-canvas layers.

| Fixture suffix | Canvas | Coding, depth, orientation | Purpose |
|---|---|---|---|
| `gray` | 259×17 | Modular Gray8, 6 | Odd multi-group width, all blend modes |
| `rgb12` | 257×9 | Modular RGB12, 8 | Native valid-bit packing after composition |
| `rgba8` | 33×7 | Modular RGBA8, 5 | Alpha zero/full/fractional values and independent sources |
| `rgba16` | 33×7 | Modular RGBA16, 2 | Unassociated high-depth alpha and extended alpha sums |
| `gray_alpha` | 259×17 | Modular Gray16+Alpha5, 8 | Gray expansion and independently normalized alpha through every blend |
| `rgba_mixed_depth` | 33×7 | Modular RGB12+Alpha5, 6 | Mixed-depth alpha rescaling after composition |
| `still` | 37×13 | Modular Gray8, 7 | All nine layers coalesce into one still |
| `vardct` | 259×17 | XYB VarDCT RGB8, 6 | Progressive AC, crop/restoration/color composition |
| `vardct_gray` | 37×13 | XYB VarDCT Gray8, 8 | Gray presentation as RGB |
| `vardct_dc` | 1024×128 | XYB VarDCT RGB8, 6 | Recursive DC references mixed with cropped ordinary layers |
| `mixed` | 259×17 | JPEG YCbCr 4:2:0 / Modular RGB8, 4 | Cross-mode reference versions |
| `gray_clamp` | 259×17 | Modular Gray8, 6 | Foreground clamp with an extended-range background |

The JPEG is the same deterministic libjpeg-turbo 3.2.0 input used for `sequence_mixed_jpeg_modular`.
Pass it as the optional generator argument; only source layer 3 uses that JPEG. Other sources
are generated integer samples. libjxl's public API disallows save slot 3, and its recursive-DC
encoder does not accept these tiny crop layers, so DC is requested only for the three full-canvas
layers. These encoder limitations are not treated as decoder grammar restrictions.

`composed_sequences_validate_every_layer_and_match_two_decoders` checks all eleven ordinary cases:
105 physical producers and 61 presentations, including six LF producers. Whole blocking and
4096-byte entropy-window / 137-byte fragmented async outputs are identical. Native 8/12/16-bit
outputs differ by at most one code from each oracle; the Rust oracle is requested in F32 and
quantized once. `floating_composition_packs_only_after_blending_and_orientation` compares applied
interleaved sRGB RGBA and kept-coordinate planar linear BGRA for the same cases, including alpha.
The error is measured in linear light for RGB and directly for alpha, divided by
`max(1, abs(reference))` to cover extended values: below `3e-6` for Modular and `1e-4` for
VarDCT-containing sequences. This is a scaled error bound, not a claim of that absolute accuracy
at arbitrary HDR magnitude or a test of floating-point JPEG XL source metadata.

The two independent-alpha-depth fixtures use normalized F32 input to libjxl with integer original
precision, retaining the existing source formulas and a separate five-bit alpha maximum. The
Gray+alpha PAM oracle contains two components; the test adapter replicates its gray component
into RGB before comparison with the requested RGBA output. Both oracle comparisons use the
ordinary native/F32 tolerances above, including crops, independent sources and retained alpha sums.

The separate clamp regression retains `350/255` after Add, then multiplies by `191/255` at original
coordinate (253,6). It verifies `350*191/(255*255)` before quantization (absolute error below
`3e-7`), and compares every quantized output against `djxl` within one code. Rust `jxl` 0.6.0
reverses the operands for this frame Multiply and clamps the background, incorrectly producing
`191/255`. This oracle disagreement is not hidden by broadening the ordinary test tolerance.
The primary implementations are [libjxl blending](https://github.com/libjxl/libjxl/blob/main/lib/jxl/blending.cc)
and [jxl-rs frame blending](https://github.com/libjxl/jxl-rs/blob/main/jxl/src/render/stages/blending.rs);
the installed pinned Rust source is the version used for the recorded discrepancy.

The conformance-only header writer in `frame_sequence/reference.rs` re-serializes the known
Modular grammar, preserves libjxl entropy sections unchanged, changes the initial layer to
post-transform ReferenceOnly and swaps slot identifiers 1/3. Gray8 and RGBA16 variants are
accepted by both independent decoders and execute on GPU, exercising non-presented references,
slot 3, subsequent overwrites and the same fragmented async input path. A pre-transform version
is rejected as an invalid post-transform blend background before GPU admission. The writer also
extracts a single final cropped layer from the still fixture; both oracles and GPU agree, with
nonzero crop pixels and zero background. This ensures a one-frame, non-animation crop uses the
common compositor instead of the standalone full-canvas still path.

Memory tests reserve the entire available budget twice before admission and verify retry without
consuming the source. Prefetch reports `FrameDependency` while a presentation is pending, permits
submission of the next after validation while the caller retains the previous output, and releases
unsubmitted input immediately on cancellation. Submitted references/uniforms/output leases retain
exact byte reservations through callback completion. Native Apple M5/Metal is the executed adapter;
WebGPU coverage is compilation only. Associated/shifted/arbitrary extra channels, ICC/non-sRGB
composition, pre-transform patches, and the official decoder conformance gate remain incomplete.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `composition_gray.jxl.hex` | 15919 | `26967f970494c2ffa969990878050fdc867bcde4642a995bd6b17ea4d3dcf7a5` |
| `composition_gray_clamp.jxl.hex` | 15919 | `2a6eb8614b9d3ade28d4d52c309f2d4324a1d5883ff4882b84fe19d529223bf6` |
| `composition_gray_alpha.jxl.hex` | 27761 | `5ec4090112c60cc51122b02db1bede1d1dc3678af0ad7812405bed6a4565fde0` |
| `composition_mixed.jxl.hex` | 51326 | `7127a68b3c70d4467a423843b04bea676ec55c9d56eeb691cc9194372f378d3f` |
| `composition_rgb12.jxl.hex` | 47572 | `352e440c21c5700e50c9484057fde2e36ac76db9869cfc417938a9033d511117` |
| `composition_rgba16.jxl.hex` | 11227 | `76b3ba2a8fcf9ee805409f0167f774c5c9d41e0c477671c9a551c2ea884cb537` |
| `composition_rgba8.jxl.hex` | 10045 | `75bc2e24a3fc2db76e823aa859764d4fa4bd5137fa30e8b629765e67b7dad851` |
| `composition_rgba_mixed_depth.jxl.hex` | 11098 | `429f39ac6f53608aa302dffe8585e35d18a5a2842d1d0981c32fa2c2cec0837f` |
| `composition_still.jxl.hex` | 3717 | `29a2ee4309f3a5ba5ac2528344f895e0b9927b0c71e993a82894fed76102f9ea` |
| `composition_vardct.jxl.hex` | 22274 | `c63af30024305c6cf48a63e6f893456e147610c6eed9a51edc3ace0b8a1a68eb` |
| `composition_vardct_dc.jxl.hex` | 418651 | `0cecab8c95ab07a947fa3331fbe7d9214d37c83c829599ec684d69880c68b6d8` |
| `composition_vardct_gray.jxl.hex` | 3883 | `156e04469e6d86705972b03e65f80c942bb557c2de946d5f058cda6ed37462c5` |

## Modular extra-channel planes and independent precision

`test-data/generate_extra_channels.c` creates six lossless Modular fixtures with libjxl 0.12.0.
It supplies normalized F32 input while declaring integer original precision and enumerated sRGB,
with patches disabled. The production decoder does not link libjxl or run CPU image reconstruction.

| Fixture suffix | Codestream extent | Color depth | Extra declarations | Orientation | Effort |
|---|---|---:|---|---:|---:|
| `data_only` | 17×1 | RGB8 | Depth16, SelectionMask1 | 6 | 1 |
| `rgb12` | 259×17 | RGB12 | Nine planes listed below | 6 | 1 |
| `gray8` | 33×7 | Gray8 | Nine planes listed below | 8 | 1 |
| `gray_alpha` | 257×9 | Gray16 | Alpha5 | 5 | 1 |
| `rgba` | 259×9 | RGB8 | Alpha5 | 3 | 1 |
| `transformed` | 515×259 | RGB12 | Nine planes listed below | 7 | 7 |

The nine-plane order is Depth16, SelectionMask1, Alpha7, SpotColor12, CFA4, Thermal8, Black6,
Optional10, Alpha15. Names are `plane-{index}-depth-{bits}`; spot RGBA is `(0.25, 0.5, 0.75, 0.5)`
and CFA index is 3. All alpha is unassociated and all original dimensional shifts are zero.
For maximum code `M = 2^bits - 1`, each plane is zero when `x % 11 == 0`, `M` when the remainder
is one, and `(193*x + 317*y + 97*c + (x ^ y)*(23+c)) & M` otherwise. Color channels start at
`c = 0`; extra channels follow the one or three color planes. Every integer code is reproducible
without storing a reference image.

`tests/wgpu_gray8/extra_channels.rs` selects every extra plane using native unsigned output with
Keep orientation and normalized scalar F32 with Apply orientation. Native output is exactly equal
to the source formula. F32 differs from both Rust `jxl` and libjxl by less than `2e-7` absolute.
The final extra-channel request for each fixture additionally uses a 4096-byte GPU entropy window
and 137-byte fragmented input with async completion; it still reconstructs every source channel.
Public metadata equals the full inventory, and profile counts distinguish color, extra and total
channels. The effort-7 case asserts multiple inverse operations in the production execution plan.
The 17×1 data-only case also proves that a zero-depth MA tree with one leaf and no decisions is
valid; the public profile validator previously rejected it.

Color tests explicitly preserve spot data and request base RGBA as F32 and native integers at the
image depth. F32 matches both decoders within `2e-7`; native samples exactly match once-quantized
Rust F32. This covers first-alpha selection at extra index 2, additional alpha planes, Gray+alpha
expansion, independent alpha normalization/rescaling, and opaque virtual alpha for data-only extras.
This initial checkpoint used a typed unsupported result for default spot rendering;
`SpotColorPolicy::Preserve` selects base color, and scalar requests expose individual spot planes.
The later GPU spot-color presentation checkpoint below enables default Render in the common decoder.

The pinned Rust `jxl` 0.6.0 exposes `adjust_orientation` but does not consume that option in its
render pipeline. For Keep-coordinate reference values, the test inverts the returned oriented
plane using the independently shared test coordinate mapping; source-formula equality additionally
checks the native GPU coordinates. `test-data/decode_extra_channels.c` requests oriented F32 RGBA
and every F32 extra plane from libjxl with spot rendering disabled. This oracle is compiled only
when `pkg-config` can locate libjxl; compilation or decoding failures then fail the tests.
Both CPU oracles ran on the recorded native Apple M5/Metal adapter with libjxl 0.12.0.

Negative and lifetime tests cover out-of-bounds extra indices, source-count overflow, color formats
incorrectly used for scalar selection, unsupported default spot rendering, admission retry after
reserving the available GPU byte budget, scalar selection from unsupported composed extras,
fragmented input ownership, pending cancellation and
caller-retained output clones. Original prediction geometry keeps the image working depth, while
the finalizer's per-view masks apply independently declared original depths, matching the
[libjxl Modular decoder](https://github.com/libjxl/libjxl/blob/main/lib/jxl/dec_modular.cc).

These fixtures cover nominal unsigned 1–16-bit full-resolution Modular samples. Associated alpha,
shifted/resampled extras, floating and wider precision, extended Modular sample ranges, reserved
extra-channel meanings, actual spot rendering, VarDCT side images and general extra-channel frame
composition remain full-format gates. WebGPU evidence is compilation only.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `extras_data_only.jxl.hex` | 230 | `593e5b5e8b0ff3000a87750722056550598389df95e65577a4fad31e5d81f71a` |
| `extras_rgb12.jxl.hex` | 42598 | `8d3670e7559cf146f8898d5273ed0197035ed19a9b2f387e2d90eba85cd49237` |
| `extras_gray8.jxl.hex` | 2831 | `a1e081afa962f42e2c0fb009b294cf2660b9e78ba3545e377ecff718305cf2df` |
| `extras_gray_alpha.jxl.hex` | 4165 | `7dc247d04103486429dde631fd522e276657ec3fe837af2071aec766389ddab2` |
| `extras_rgba.jxl.hex` | 8251 | `ebd817ddfb26ee8e07c3af133e5d7b19a3f534b40051a1bac41d7f8284c1a79d` |
| `extras_transformed.jxl.hex` | 769485 | `a352bd19c118c5704a99455144b95910abe352cd14f62402ab9469c38371d57b` |

## Resident VarDCT Modular substream staging

`generate_extra_channels.c OUTPUT_DIRECTORY --vardct` uses libjxl 0.12.0 to generate seven
VarDCT streams at distance 1 with lossless extra channels (distance 0). Original integer depths,
extra types/names, source formulas and spot/CFA metadata are the same as the Modular extra corpus
above. The first five cases use effort 1 and the transformed/progressive cases effort 7; patches are disabled.
The original six have a single TOC entry and their extra samples reside in the global Modular substream.

| Suffix | Original extent and color depth | Extras | Orientation | Token start / ending cursor in codestream bits | Entropy samples / meta channels |
|---|---|---|---:|---|---|
| `data_only` | 17×1 RGB8 | Depth16, SelectionMask1 | 6 | 910 / 1243 | 45 / 1 |
| `rgb12` | 33×7 RGB12 | Nine planes listed above | 6 | 2393 / 18537 | 2079 / 0 |
| `gray8` | 33×7 Gray8 | Nine planes listed above | 8 | 2408 / 18732 | 2079 / 0 |
| `gray_alpha` | 37×9 Gray16 | Alpha5 | 5 | 812 / 2430 | 333 / 0 |
| `rgba` | 63×9 RGB8 | Alpha5 | 3 | 885 / 3436 | 567 / 0 |
| `transformed` | 127×129 RGB12 | Nine planes listed above | 7 | 24012 / 551118 | 148077 / 2 |

`wgpu_engine::side_image::modular::tests` parses only bounded header/MA/transform descriptors,
then uses the same resident executor as the production raw matrix path. All extra integer planes
equal the deterministic original codes exactly. Once normalized by each declaration's maximum,
they agree with both Rust `jxl` 0.6.0 and optional native libjxl within `2e-7` absolute. The shared
test-only CPU helpers are in `tests/common/extra_channel_oracle.rs`; orientation is reversed only
for reference lookup, while the resident planes remain in codestream coordinates.

The GPU validates entropy termination and returns an absolute bit cursor without byte alignment;
the following LF header is parsed at that cursor. The 17×1 case has a two-channel Palette, and
the effort-7 case has Palette metadata planes of widths 502 and 128. Meta channels always belong
to the global substream, including widths beyond a 256-pixel pass group. These fixtures therefore
exercise one-, two- and nine-plane outputs and inverse Palette reconstruction independently of
the raw-matrix three-plane overlay. Core buffer/uniform bytes match the precomputed reservation,
and all reserved bytes are released after completion. Apple M5/Metal is the executed adapter.

`tests/vardct_engine_gpu/extra_channels.rs` now also drives all six through the public decoder.
It adds `rgba_progressive` (63×9 RGB8+Alpha5, orientation 2, effort 7, `PROGRESSIVE_AC=1`), whose
independent LF-global section exercises byte-padding validation before multi-pass color decoding.
All seven use blocking Apply and fragmented asynchronous Keep, preserve exact extra-channel
metadata and actual color depth, and reconstruct F32 RGBA against Rust `jxl` and native libjxl.
Observed Rust color maxAE is at most `3.37e-5` in encoded sRGB; first-alpha F32 values agree exactly
on Apple M5/Metal. The test thresholds are `3e-4` for Rust color, `2e-3` for native color and `2e-7`
for alpha. The standalone output test additionally supplies negative/overshoot signed Alpha5
samples at a nonzero word offset across all eight orientations and both single-pixel axes; F32
retains signed normalization and U8 clamps during packing rather than wrapping by a depth mask.

Memory tests check exact initial reservation, budget-driven upload reduction, a 39-byte caller
cap/minimum-budget rejection, backpressure
without consuming the source, canceled callback ownership after multiple windows, retry and final release. A corruption
fixture zeroes only entropy bits 885..3436 of `rgba`, preserving both descriptors and later color
data; GPU status rejects it before a color plan or frame is exposed. Color planning remains absent
until global cursor validation; later LF/HF descriptors retain their separately admitted dynamic
bytes. Shifted samples and associated alpha were gaps at that checkpoint; the extensions below
add both. No production picture data or entropy token crosses to a CPU decoder.

`tests/vardct_engine_gpu/extra_channels/scalar.rs` selects all 32 extra planes in those seven
fixtures. Native unsigned output with Keep orientation equals the source formula exactly at each
declaration's depth; normalized F32 with Apply orientation and fragmented 1024-byte-window input
matches Rust `jxl` and native libjxl within `2e-7` absolute. Requests include spot data under the
default policy, since scalar selection does not request color rendering. Memory assertions show
zero color planes, inverse-transform scratch, restoration and resampling allocations, a 64-byte
output uniform and four-byte output status. All reservations retire after frame/session release.
Wrong indices/depths fail before admission. A progressive fixture whose final AC section is zeroed
still validates the global extra stream, then returns `HfCoefficientGpu` without a scalar frame.

The standalone scalar packer test checks 640 combinations of two resident source domains
(signed integer and normalized F32), five integer precisions, four
extents (including both one-sample axes and a forced 2-D dispatch), all eight orientations and
both mappings. It supplies signed negative/overshoot values, nonzero source/binding/plane offsets,
padded source/output rows and guard bytes. Native out-of-range values set a typed rejection;
normalized F32 retains them and agrees with signed scalar division within two ULPs on Metal.
Padding and outside-binding guard bytes remain exact. The existing Modular finalizer runs beside
each F32 case with the same oracle and a successful status, covering its corrected signed source
read. Host tests reject precision mismatch, overflowing rows and insufficient storage/dispatch/
uniform limits, and Naga validates the new shader and 64-byte aligned uniform ABI.

The six internal fixtures additionally run through a 40-byte stream cap (1024 bytes for the large
transformed image). Every original plane, decoded sample count and absolute unaligned cursor is
identical to the whole-range path, and each stops before uploading the rest of its single TOC
packet. All seven public fixtures also compare complete RGBA bytes against whole input under those
caps using fragmented asynchronous input; corruption is rejected in both modes. A GPU unit case
reconstructs 27 constant samples from a zero-bit single-symbol Prefix descriptor at bit positions
0, 1, 7 and 8, including an empty upload with a four-byte sentinel. Host geometry tests cover
unaligned starts, empty ranges, boundary lengths and a u32-sized entropy range described lazily
without allocating its more than 134 million potential window records.

The bounded-global checkpoint on 2026-09-08 Apple M5/Metal covered 268 decoder tests and 390 tests in the rest
of the workspace, plus warning-free Clippy/rustdoc, Rust 1.89, the six-crate WASM check, reference
and Metal harness verification, and indexed codec/readback. GPU suites were validated serially.
An earlier default-parallel decoder run failed two cancellation assertions and left several Metal
waits pending; the complete 44-test `wgpu_gray8` target subsequently passed with
`--test-threads=1`. Concurrent-suite stability remains unverified by this run.

The scalar-output validation reproduced one cancellation assertion even with serial tests:
the sequence test counted retained bytes after its own `Device::poll`, while the backend worker
could still be executing callbacks extracted by a concurrent poll. The callbacks already release
their job ownership before publishing completion. The test now waits for native poll permits to
retire before asserting exactly the caller-held lease, then zero bytes after its release. This
fixes the synchronization assumption without relaxing byte counts or changing production lifetime
ownership; it does not establish the cause of the earlier pending Metal waits.

The scalar-delivery checkpoint covers 273 decoder tests (272 in the full run plus the corrected
cancellation test's targeted rerun) and 390 tests elsewhere in the workspace. One manual allocation
benchmark remains ignored. Formatting, workspace check, warning-free Clippy/rustdoc, Rust 1.89,
six-crate WASM, both 18-case reference/Metal harnesses and indexed codec/readback pass.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `vardct_extras_data_only.jxl.hex` | 293 | `29c6d297ca5dcb8e70f508fac4df01eae85487bf7b1923dc52d5121905ccc084` |
| `vardct_extras_gray8.jxl.hex` | 2616 | `e4291f404b542f7f376346e506976463401aaf0aeddca95155ef9ba4abcdce81` |
| `vardct_extras_gray_alpha.jxl.hex` | 465 | `719ebc5ebcadeb5b484bf1465b8558907b54996aa1bce2f0b00d53cc5d80d00f` |
| `vardct_extras_rgb12.jxl.hex` | 2634 | `235ee8ec0156ded99ecd66281d12747429053a710a8f50494cb39808333e2db4` |
| `vardct_extras_rgba.jxl.hex` | 1607 | `d0e532f5d2eec3aa2a9465332a0743b927f2c72d0bdabbfd42a6c93685353c3f` |
| `vardct_extras_transformed.jxl.hex` | 81790 | `c5af1524af32e6ec9d87bc66118d37551a51615a6c22c902c41a05f61742359f` |
| `vardct_extras_rgba_progressive.jxl.hex` | 1531 | `5151de6b5f7826080cc0f520b3e9ef6d469043463db26d1bf2ae91b73d7bcf81` |

### VarDCT distributed extra channels (2026-09-08)

`generate_extra_channels.c OUTPUT_DIRECTORY --vardct-distributed`, built against libjxl 0.12.0,
adds five descriptor/partition and public decode fixtures. Their color/extra sample formula is
the same as the
preceding global-extra fixtures; every extra channel is encoded at distance zero.

| Fixture suffix | Source | Extra declarations | Global/coded channel count | Group coverage |
|---|---|---|---:|---|
| `alpha` | RGB8, 257×17, orientation 6, effort 1 | Alpha5 | 0/1 | two AC groups; empty global prefix |
| `data` | Gray12, 517×9, orientation 8, effort 1 | nine independently sized types/depths | 2/11 | global Palette metadata; three AC groups |
| `progressive` | RGB8, 259×257, orientation 2, effort 7 | Alpha5 | 12/13 | global Squeeze channels; three passes and four spatial groups |
| `wide` | Gray16, 2049×9, orientation 5, effort 1 | Alpha5 | 0/1 | two LF groups, nine AC groups; empty global with local-tree flag |
| `squeeze` | RGB12, 2051×259, orientation 7, effort 7, responsive | Alpha5 | 10/16 | global and LF Squeeze channels, three passes, clipped AC edges |

The partition test marks the full transformed arena and requires exactly one owner for every
coded sample. It covers global-prefix ordering, LF versus asymmetric pass shifts, progressive
brackets and clipped zero-size channels. Additional regression cases cover a downsampling boundary
on the final pass: it retains the preceding maximum shift and forces its minimum to zero,
preserving unassigned resolutions. These fixtures now decode through the public GPU frame
scheduler. Empty global subimages skip
MA/entropy parsing; corrupting only their unused section suffix preserves the prepared plan.
The header/entropy split follows [libjxl ModularDecode](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/modular/encoding/encoding.cc),
and omission of clipped empty frame channels before local transforms follows
[libjxl DecodeGroup](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_modular.cc).

The low-level AC continuation test uses the actual coefficient shader and shared Modular GPU
executor for 24 combinations of three entropy descriptors and eight starting bit alignments.
Each AC stream selects a nonzero HF preset, decodes the three zero nonzero-counts of one DCT8
block, and hands the next exact bit to a Modular header. That substream reconstructs six resident
integer samples and returns another exact cursor before an unrelated nonzero suffix. Descriptors
cover zero-bit single-symbol Prefix, a two-symbol one-bit Prefix code and terminal-state ANS.
Exact-packet mode rejects the same non-padding suffix; continuation finishes in a nonfinal window,
resumes after a forced Prefix yield, and rejects a corrupted ANS terminal state or truncated initial
state. Host checks reject changed packet bounds, wrong group identity and every failure status.
A real progressive multi-LF plan with 128-byte windows verifies that selecting one pass group's
termination updates every resume record while preserving the other groups' exact termination.
Naga verifies the unchanged 160-byte parameter ABI and the new final word at byte offset 156.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `vardct_extras_distributed_alpha.jxl.hex` | 8912 | `07e454cdf6d14c92f8227ef1656dff9546ddeab2ca2cc82ad4a86e5dd230263a` |
| `vardct_extras_distributed_data.jxl.hex` | 38109 | `8906ec29ed1e8d10f80c3e9fab95a0e768134d583079348f7b5c845f714eb5fa` |
| `vardct_extras_distributed_progressive.jxl.hex` | 95654 | `8b04d3e56120101708a98242cebbb3616586a999f9e600254e36be2722369cc5` |
| `vardct_extras_distributed_squeeze.jxl.hex` | 598421 | `692211f95f0e7f35228f035232a52c90af854eb950cdbaa85b76bb9cc16f98c8` |
| `vardct_extras_distributed_wide.jxl.hex` | 15992 | `8cc1e8c0f0a43b7d3e6f94f30ab1c68028c400434fd8723075673756967e12fe` |

`tests/vardct_engine_gpu/extra_channels/distributed.rs` reconstructs all 13 extra planes as
exact native integer codes and normalized F32, and color/first-alpha output against Rust `jxl`
and libjxl. Whole blocking input and 347-byte fragmented async input with a 1024-byte cap cover
empty global prefixes, local MA trees, Palette metadata, progressive pass assignment, clipped
edges, multiple LF groups and the 2051×259 Squeeze image. Scalar output allocates no color planes.
Normalized extras/alpha agree within 2e-7; color tolerances are 0.0003 against Rust and 0.002 against
libjxl. A separate 128-byte-window test verifies initial memory backpressure/retry, cancellation
during a Modular subimage, complete permit retirement, and late exhaustion without output.
Clearing only the final four bytes of `alpha`'s last AC section preserves its color AC stream but
fails extra entropy validation, for both color and scalar requests under whole and bounded input.

The tiny second LF group of `wide` uses a degenerate complex Huffman code-length alphabet. The
shared descriptor parser now preserves its sole nonzero symbol as a zero-bit code. Regression
tests cover lengths 1–15 and every applicable skip form, exact header consumption, incomplete
alphabets and truncation. The interpretation follows
[libjxl's Huffman descriptor reader](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_huffman.cc).

Validation on Apple M5/Metal: the serial full workspace passes 675 tests with one existing manual
allocation benchmark ignored. The strengthened late-admission assertion also passes in a focused
GPU rerun. Final formatting, workspace check, warning-free Clippy/rustdoc, Rust 1.89 and the
six-crate WASM check pass. Reference and Metal harnesses each pass all 18 cases, and indexed GPU
decode with CPU readback passes. This closes the distributed full-resolution integer-extra path;
At that checkpoint shifted/resampled/floating extras, associated alpha and broader cross-feature
conformance remained; the following checkpoint adds integer resampling.

## Integer channel resampling

`generate_extra_channels.c OUTPUT_DIR --resampled` generates the following ten cases in both
Modular and VarDCT form with libjxl 0.12. Filenames use `extras_` or `vardct_extras_` followed by
the suffix below and `.jxl.hex`. The source formula and independent extra depths are unchanged
from the earlier extra-channel fixtures. Color is lossless Modular or VarDCT distance 1; extras
have distance 0 before the requested downsampling. These are supported still-image fixtures,
not claims of lossless reconstruction after resampling.

| Suffix | Original extent | Color/depth | Color factor | Extra factor | Dimension shift | Orientation |
|---|---|---|---:|---:|---:|---:|
| `resampled_2` | 517×9 | RGB12 + Alpha5 | 1 | 2 | 0 | 6 |
| `resampled_4` | 37×17 | Gray8 + nine independent extras | 1 | 4 | 0 | 8 |
| `resampled_8` | 2051×9 | RGB16 + Alpha5 | 1 | 8 | 0 | 5 |
| `resampled_color` | 259×17 | RGB8 + Alpha5 | 2 | 8 | 0 | 7 |
| `resampled_color4` | 17×257 | Gray8 + Alpha5 | 4 | 4 | 0 | 4 |
| `resampled_color8` | 9×1 | RGB16 + Alpha5 | 8 | 8 | 0 | 2 |
| `resampled_squeeze` | 2051×17 | RGB12 + Alpha5, effort 7, responsive/progressive | 1 | 2 | 0 | 6 |
| `shifted` | 37×9 | RGB8 + Alpha5 | 1 | 2 | 1 | 3 |
| `shifted4` | 37×9 | RGB12 + Alpha5 | 2 | 4 | 2 | 1 |
| `shifted8` | 2051×9 | RGB8 + Alpha5 | 1 | 8 | 3 | 7 |

`tests/vardct_engine_gpu/extra_channels/resampled.rs` drives both producers through the public
decoder. All extra planes are selected independently as native codes and normalized F32, plus
RGBA color/first-alpha output. Whole blocking and 43-byte fragmented asynchronous input with a
1024-byte entropy cap must produce identical bytes. Native resampled codes are within one code
of final rounding of the float oracle; scalar/alpha F32 error must be below `4e-7`, Modular color
below `1e-6`, and VarDCT color below `3e-4` against Rust `jxl` or `2e-3` against libjxl. Every frame
retains only its output lease after completion; releasing it returns the reservation to zero.

Rust `jxl` 0.6 handles `dimension_shift` inconsistently, so the six explicitly shifted cases use
the independent libjxl C oracle and are skipped only when that optional native tool is unavailable.
The other fourteen cases compare both oracles when libjxl is installed. Inventory unit tests also
verify that an all-default frame includes image-header shifts and reject effective factors above
8 or below the color factor. The interpretation follows the reference
[frame-header visitor](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/frame_header.cc) and
[default-field initialization](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/fields.cc).

Ten runtime fixture variants replace the custom transform-data weight fields with all three
binary16 `1/32` compact kernels while preserving the original opsin data and entropy packets.
Raw/container inputs are first resolved to their logical codestream. Both GPU producers compare
color/alpha against Rust and libjxl for these non-default kernels, including 4×/8× color resampling
and the one-sample axis. Separate actual-adapter tests prove reservation before submission,
retry after initial backpressure, cancellation and subsequent successful reuse for both modes.
Host render-plan tests verify aligned disjoint destination views, one shared normalization
scratch plane, weight deduplication and rejection before allocation on geometry/address/limit errors.

At that checkpoint associated alpha, floating source samples, general extra-channel composition
and resampled crop/blend cross-products remained conformance gaps. The following sections add
associated integer alpha and all-channel integer composition, including resampled cases.

Validation on Apple M5/Metal: the serial full workspace passes 682 tests with one existing manual
allocation benchmark ignored. After tightening workgroup and source-address admission, all 159
decoder library tests and all four resampling integration tests pass again. Final formatting,
workspace check, warning-free Clippy/rustdoc, Rust 1.89 and the six-crate WASM check pass. Reference
and Metal harnesses each pass all 18 cases, and indexed GPU decode with CPU readback passes.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `extras_resampled_2.jxl.hex` | 10687 | `e9fb348591270f92512a111d20c50dc9caf99c90ee7805d1152642e14729c5fc` |
| `extras_resampled_4.jxl.hex` | 1491 | `f6c8995e76eccada796f2df0abd96a3c1c2f5c5603cbe7fbb856d91a8f05dc83` |
| `extras_resampled_8.jxl.hex` | 28250 | `b16b9d3c815cf6430dfdb9e9d32b19611b240a5393fbda565f227d2087c124a5` |
| `extras_resampled_color.jxl.hex` | 3418 | `5ecc157683dcecb32949eb8b0fd9f2bb6d0e1e1b7a55a36c8c150576e0232a00` |
| `extras_resampled_color4.jxl.hex` | 376 | `7960a7bcd0ecc72a4b63f660dbb8bd4f7d2566e5ae185cdcfa9921608d7aa24a` |
| `extras_resampled_color8.jxl.hex` | 111 | `f9a429de9771559a9127c9c4871cbb79776b5cbd486dab2f8bbc37259f3efa5e` |
| `extras_resampled_squeeze.jxl.hex` | 98793 | `7edf36718efcad326ce119da67e2ecc3bff07474408f2f221f6b4819953c4348` |
| `extras_shifted.jxl.hex` | 942 | `083ae680f7e5fcceea763b8d5acd30837e6692b1948fc2c3633267297dc9b84d` |
| `extras_shifted4.jxl.hex` | 480 | `56b4e62579a85b344271783ea9fa28def4cc1298cecdbe7773d02e42aeaefd30` |
| `extras_shifted8.jxl.hex` | 50176 | `940d6e4ea71af8377d658b66bfb49e238a1933f9b1f3afe622b48ea68e57df99` |
| `vardct_extras_resampled_2.jxl.hex` | 6791 | `4b41f970e5984854e9b58021f4f5749080024df14ad78467fed9785655cef15b` |
| `vardct_extras_resampled_4.jxl.hex` | 1556 | `4f2a534676de0bb309ffc878e871c9ed5572c29f72f6d6ea893dea7a7cee8288` |
| `vardct_extras_resampled_8.jxl.hex` | 5877 | `9c808eca038ec93727c7d483e8d8b0c0c11622e907a35a9293ad1cb0d3648a07` |
| `vardct_extras_resampled_color.jxl.hex` | 2400 | `b04d13da9f3d67b43d6182149dd4ab0783a5a697e1c850e54137475fe3f63dd6` |
| `vardct_extras_resampled_color4.jxl.hex` | 535 | `56c1a03182ae862e9db34035abe9bc9c2ef9e36cf377d7608a3baadfb5eda8e4` |
| `vardct_extras_resampled_color8.jxl.hex` | 103 | `247c3daffb86a564d984697b906edf1249ded8b15fe15ec8b57d761844690e7b` |
| `vardct_extras_resampled_squeeze.jxl.hex` | 44561 | `aef8458b298255e2b12e1a6c42fac7549f0f4b3d3c3ef2bee476957ff00c3817` |
| `vardct_extras_shifted.jxl.hex` | 801 | `20e154cdb46cac67b7580709f5e10a39403df8cdd87e8058cf3c548935457516` |
| `vardct_extras_shifted4.jxl.hex` | 324 | `20b89f55f2c160095ab84fab8e7d2c6f2c445af4033f40f37db9ffcbe31d5bb4` |
| `vardct_extras_shifted8.jxl.hex` | 27420 | `8eb30fc3415876b82453dc922445e1763cf9c3a72441f6860a7c5b7a36c73641` |

## Associated integer alpha and output association

`generate_extra_channels.c OUTPUT_DIR --associated` and
`generate_frame_composition.c OUTPUT_DIR --associated` generate 17 additional fixtures with
libjxl 0.12.0. These offline tools are never linked by the production decoder. The still source
uses the preceding extra-channel formula, except color is code 23 at `x % 11 == 0` on odd rows:
those positions deliberately have zero alpha and nonzero invisible color. Only the first alpha
is associated; the second alpha in `associated_data` remains unassociated. Lossy fixtures enable
KEEP_INVISIBLE and use distance 1; Modular is lossless before requested downsampling. Source
precision is independent of alpha precision.

| Still suffix, in both `extras_` and `vardct_extras_` form | Original extent | Color/alpha | Color/extra factors | Dimension shift | Orientation |
|---|---|---|---|---:|---:|
| `associated_same` | 33×7 | RGB8 / Alpha8 | 1/1 | 0 | 4 |
| `associated_rgb` | 259×9 | RGB8 / Alpha5 | 1/1 | 0 | 6 |
| `associated_gray` | 17×257 | Gray16 / Alpha5 | 1/1 | 0 | 8 |
| `associated_data` | 33×7 | RGB12 / nine extras, first alpha at index 2 with depth 7 | 1/1 | 0 | 5 |
| `associated_thin` | 1×9 | RGB16 / Alpha5 | 1/1 | 0 | 2 |
| `associated_resampled` | 259×17 | RGB12 / Alpha5 | 2/8 | 1 | 7 |
| `associated_squeeze` | 2051×17 | RGB12 / Alpha5, effort 7 responsive/progressive | 1/1 | 0 | 3 |

The three composition fixtures use nine physical layers and six coalesced presentations. They
cover Replace/Add/Blend/Mul/MulAdd, separate RGB/alpha background slots, clamping, hidden layers,
negative/oversized/off-canvas crops, extended values and reference replacement. Their source is
the earlier composition formula, with associated Alpha5 and KEEP_INVISIBLE enabled.

| Composition file | Original canvas | Coding/color precision | Orientation |
|---|---|---|---:|
| `composition_associated_rgb.jxl.hex` | 259×17 | Modular RGB12 + Alpha5 | 6 |
| `composition_associated_gray.jxl.hex` | 37×9 | Modular Gray16 + Alpha5 | 8 |
| `composition_associated_vardct.jxl.hex` | 259×17 | VarDCT RGB12 + Alpha5, distance 2, progressive AC | 5 |

`tests/vardct_engine_gpu/extra_channels/associated.rs` exercises all three public policies:
Unassociated (default), Preserve and Associated. Whole blocking and 43-byte fragmented async input
with 1024-byte entropy windows must return identical bytes. Tests cover source declarations,
independent first-alpha selection, native RGB/RGBA (including omitted alpha), interleaved RGBA and
planar BGRA F32, original sRGB and linear output, Apply/Keep orientation, and zero reservations
after final output/session release. Scalar native/F32 selection of every extra ignores the alpha
policy and agrees with libjxl; unresampled codes remain exact. Equal-depth Preserve also exercises
the direct Modular kernel, while conversion retains the alpha view and selects the general packer.

Rust `jxl` 0.6 returns the source association, so the test applies the requested final conversion to
that reference. Explicitly shifted stills retain the earlier libjxl-only oracle rule. The optional
C oracle now links `libjxl` and `libjxl_cms`, exposes Preserve/linear/Keep controls, and writes each
coalesced frame followed by its extras. Associated source-over and the finite `2^-26` output floor
follow [libjxl alpha operations](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/alpha.cc),
[output packing](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/render_pipeline/stage_write.cc)
and [render stage order](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_cache.cc).

For unpremultiplied RGB, the comparison scales error by `max(alpha, 2^-26)` before applying the
existing reconstruction tolerance, then divides by `max(1, abs(scaled_reference))`. Alpha error
is below `4e-7`. Still color limits are `3e-6` for Modular, `3e-4` against Rust VarDCT and `2e-3`
against libjxl VarDCT. Native packing agrees with the corresponding F32 rounding within one code.
Composed color uses `8e-6` for Modular and `3e-3` for VarDCT. Linear output is also compared with
analytic sign-preserving sRGB conversion of the independently decoded original-encoding result.
This checks extended/invisible values even outside the unit RGB cube. The direct CMS check uses
`1e-4` within that cube: its extended-value approximation is not substituted for the analytic
contract. libjxl 0.12's [blending stage](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/render_pipeline/stage_blending.cc)
rejects non-original output encoding for coalesced XYB frames; their linear output is validated
against the original-encoding dual-oracle result followed by the analytic transfer.

An independent f64 BT.2020 constant-luminance reference additionally checks P010 after association,
including inverse/forward BT.2020 transfer, centered chroma averaging and limited-range 10-bit
packing; NV12 and odd-width YUYV/UYVY use the shared scalar layout oracle, including replicated
final-luma alpha. Both compare within one stored code and check zero padding bits. Low-level output
tests reject association conversion without an alpha
binding before allocation, while existing ABI tests parse and validate the enlarged common
192-byte uniform. Production still has no CPU image-domain fallback. Floating source samples remain
a completion gate; the all-channel composition extension below adds integer extras and
associated/resampled composition cross-products.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `composition_associated_gray.jxl.hex` | 5135 | `4ac183404328fcb3169041eecba8f9b1a6b75f5a86d15f5c84549e873bd6f1e5` |
| `composition_associated_rgb.jxl.hex` | 85045 | `df122abd5caf4191f45badc796433357702d064e582056451686f4160c032fb2` |
| `composition_associated_vardct.jxl.hex` | 25884 | `96edfc878b2639a464442d45f297b7ce3bb03f70a3b0e7eb800fca05936f9842` |
| `extras_associated_data.jxl.hex` | 2887 | `d45882bd1fb9b5dd02fce31fddcfd2cc643f180ca7c75a9fab3be0c5b034317a` |
| `extras_associated_gray.jxl.hex` | 6844 | `617fdbb0d3b41990aa63da8bdd87b25191642707ea2bdc207334c5219fa2ab6a` |
| `extras_associated_resampled.jxl.hex` | 3786 | `6799e9bfe11708713aecc507f63d7d0c1d1d850f6104adc08b3a917234b228aa` |
| `extras_associated_rgb.jxl.hex` | 8450 | `e6b049c567566249d382bf01e3b301fd094a6e05f4ac8c4ef0e1de40d8d2155a` |
| `extras_associated_same.jxl.hex` | 891 | `762dc36f6c9fd0b5d8c5b8bfadaa19598c08ba870ac555f3a8427dc3ff2a873b` |
| `extras_associated_squeeze.jxl.hex` | 108755 | `485d026a13c0027fbd5b60ba5600cdcda0f9d1087637d0d93e4b9cfa52fe99fd` |
| `extras_associated_thin.jxl.hex` | 104 | `a8f20037d134ab63b663ec1593ce9d0a9a757396d08ffc88a98a59f9a6f73cb6` |
| `vardct_extras_associated_data.jxl.hex` | 2637 | `fbe3df1bc46d0996cf3cb035ffae692025a55af8b9d3ede7b87632a4f3dc30ff` |
| `vardct_extras_associated_gray.jxl.hex` | 3277 | `303f9cfa65590dfa75bdfb9cdd56fd4674d84bd67cc7f60f3f46bdcfdcc96d48` |
| `vardct_extras_associated_resampled.jxl.hex` | 2162 | `021b546e4d6a857a0703cd774141948b542388aa692531d45ee505c51ffef315` |
| `vardct_extras_associated_rgb.jxl.hex` | 5508 | `5a3e64018ea8e19e980ae40ffaf81283bc54148ea99ea4bb306ea110967bee2d` |
| `vardct_extras_associated_same.jxl.hex` | 734 | `d9da713bb4ab9d81c763f94cd468f3d110adef62e829c577958115026fce8336` |
| `vardct_extras_associated_squeeze.jxl.hex` | 52991 | `6dcbda1a99115d148530cee75dfeddf630b960324849f9aa5cc1cdda9e112da6` |
| `vardct_extras_associated_thin.jxl.hex` | 83 | `c6b744e09e7345970152b41c14ac65113c7f99118f530977bc9d729f15784e95` |

Validation on 2026-09-08 Apple M5/Metal: the serial full workspace passes 686 tests with one
existing manual benchmark ignored. Formatting, all-target/all-feature checks, warning-free Clippy
and rustdoc, Rust 1.89, and the six-crate `wasm32-unknown-unknown` compile check pass. The reference
and Metal verification harnesses each pass all 18 cases; indexed Gray8 decoding to U8 CPU readback
also passes. All 17 new fixture lengths and SHA-256 values match the table above.

## Composition of every integer extra channel

`test-data/generate_extra_composition.c OUTPUT_DIR`, built against libjxl 0.12.0, generates seven
additional independent codestreams. The production path uses no CPU codec. Physical Modular and
VarDCT producers return one accounted, aligned planar F32 allocation containing RGB and every extra
plane. The compositor retains all planes in reference slots; each extra has its own blend mode,
background slot, alpha selector and clamp flag. Color output selects the first declared alpha only
at presentation. Scalar F32 exposes the normalized composed value without clipping or color/alpha
conversion. Native scalar output clamps and rounds once at the declaration's integer depth.

| Suffix in `composition_extras_*.jxl.hex` | Canvas | Color / mode | Extras | Orientation |
|---|---|---|---|---:|
| `rgb` | 259×17 | RGB12, Modular | nine, first alpha unassociated | 6 |
| `gray` | 37×9 | Gray16, Modular | nine, first alpha associated | 8 |
| `vardct` | 259×17 | RGB12, VarDCT distance 2 | nine, first alpha unassociated | 5 |
| `distributed` | 2051×17 | RGB12, VarDCT distance 2, effort 7 responsive/progressive | nine, first alpha associated | 3 |
| `resampled` | 259×17 | RGB12, Modular, color factor 2 / extra factor 8 | nine, first alpha associated, dimension shift 1 | 7 |
| `vardct_resampled` | 259×17 | RGB12, VarDCT distance 2, color factor 2 / extra factor 8 | nine, first alpha unassociated, dimension shift 1 | 2 |
| `data` | 33×7 | RGB8, Modular | Depth16, SelectionMask1, no alpha | 4 |

The nine declarations, in order, are Depth16, SelectionMask1, Alpha7, SpotColour12, CFA4, Thermal8,
Black6, Optional10 and Alpha15. The second alpha has the opposite association to the first. Names
are `composed-plane-C-depth-BITS`, spot RGBA is `(0.25, 0.5, 0.75, 0.5)`, and CFA index is 3.
Source values are integer codes divided by `2^bits - 1`. The deterministic code for local `(x,y)`,
channel `c` and physical frame `f` is
`(193*x + 317*y + 97*c + (x XOR y)*(23+c) + f*(71+c)) AND (2^bits-1)`, overridden to zero when
`x % 11 == 0` and to the maximum when `x % 11 == 1`. Extra source channel numbering follows the
one or three color channels. KEEP_INVISIBLE is enabled, patches disabled, extra distance zero,
and progressive AC enabled. Inputs to the offline encoder use F32 transport but the codestreams
have integer sample declarations.

All seven files have nine physical layers and six presentations. The crop/duration/reference
schedule matches `generate_frame_composition.c`: negative, oversized, partial and fully off-canvas
layers; durations `[1,0,2,1,0,1,1,0,2]`; full Replace, Blend, Add, Multiply and weighted Add; reference
slots 1 and 2 overwritten between presentations and missing slots used as zeros. Color selects
alpha 2 or 8 on alternating layers. Each extra cycles through all five modes, chooses a separate
source slot `(color_source + c % 3) % 4`, alternates alpha 2/8, and has an independent clamp bit.
The no-alpha file exercises color Blend-as-Replace and weighted-Add-as-Add alongside data-plane
blending. Source-over updates only its selected alpha; alpha's weighted Add preserves its own
background. The selected alpha's background comes from its own reference slot. These operations
follow libjxl's [blending implementation](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/blending.cc)
and [alpha equations](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/alpha.cc).

Four tests in `tests/vardct_engine_gpu/extra_channels/composition.rs` cover:

- RGBA F32 plus every extra as normalized scalar F32, all six presentations, whole blocking and
  43-byte fragmented asynchronous input with a 1024-byte entropy window. Outputs are byte-identical
  across input strategies. Numeric requests also prove that association policy does not affect data.
- Native scalar output for every declaration in `gray` and `vardct_resampled`, codestream orientation,
  whole/bounded equality, depth masks and rounding within one integer code of libjxl's composed F32.
- Preserve/Associated/Unassociated planar linear BGRA output for Gray and VarDCT multi-alpha
  composition, compared with libjxl's coalesced original encoding followed by analytic transfer and
  requested association. Near-zero-alpha comparisons undo the output division before evaluating
  reconstruction error.
- Repeated initial memory pressure, dependent prefetch, and cancellation with caller-held previous
  scalar output for Modular, distributed VarDCT and resampled Modular. Every hidden plane/reference
  reservation disappears after submitted work completes; only the caller's output lease remains.

The optional libjxl C oracle runs with preserved alpha and checks every presentation and plane;
its source code remains test-only. Relative error is `abs(actual-reference) / max(1,abs(reference))`:
less than `8e-6` for Modular color, `3e-3` for VarDCT color and `4e-6` for extras. Analytic linear
output uses `2e-5` for Modular and `3e-3` for VarDCT. Rust `jxl` 0.6 independently checks all unshifted
initial Replace presentations. It is not the numeric authority for the subsequent reference chain:
its known clamped-Multiply operand reversal yields `1` where libjxl/GPU preserve `1.8245761` in the
RGB fixture. Its dimension-shift defect also excludes the two resampled files. When libjxl is
unavailable, those numeric checks are skipped; this checkpoint ran with libjxl installed.

Two CPU layout tests prove aligned, nonoverlapping real-offset views and reject a total all-plane
allocation that exceeds the device limit even when RGB alone fits. Compile-time ABI checks cover
128-byte blend geometry, 32-byte per-channel metadata and 64-byte native/scalar output parameters,
including alignment and field offsets. Actual GPU execution validates the corresponding WGSL.
Floating sample declarations, pre-transform patch references, general original color domains and
broader resampling/metadata cross-products remain separate roadmap gates.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `composition_extras_data.jxl.hex` | 11336 | `cb12ce6bdd36d36b3d0d0fb58852fa58f093e8578078f7a131b9acc7c1079520` |
| `composition_extras_distributed.jxl.hex` | 1128801 | `54c4e2ecde9b0f9ab9991514919fc5d4bd6de4710553433084f262a04be9f529` |
| `composition_extras_gray.jxl.hex` | 31315 | `b6f1d09a7b0e9960123436c4b714629473525ff44a242e344663fc29b7a0a3db` |
| `composition_extras_resampled.jxl.hex` | 26973 | `25ef2e245bcdb37e8c84ede27747f182fa14ff1110c41b270fbfec43d3a1b882` |
| `composition_extras_rgb.jxl.hex` | 234974 | `12b6d8721b0237ecabdea5e5541da464e97bdc30c229844695ed77c0ad8ade2f` |
| `composition_extras_vardct.jxl.hex` | 187027 | `1f000437952cbe26e98cea30324806b31e2c4af2e6810f191566836d452db449` |
| `composition_extras_vardct_resampled.jxl.hex` | 12625 | `71ad39384633bc38327aca77fd004cb6f3d91bc47b7d0aacbf23855d7c7c09e8` |

Validation on 2026-09-08 Apple M5/Metal covers all 24 workspace test targets: 692 distinct tests
pass, with one existing manual benchmark ignored. This combines the serial all-target/all-feature
workspace run, the corrected obsolete rejection test, and the remaining encoder suite. The updated
selection contract test verifies six composed scalar presentations and a typed out-of-range index
error, then passes with all features enabled. No runtime implementation change was needed for that
expectation update. Formatting, all-target/all-feature checks, warning-free Clippy/rustdoc,
Rust 1.89 and six-crate WASM compilation pass. Reference and Metal harnesses each pass 18 cases;
indexed Gray8 U8 CPU readback passes. Regenerating all seven fixtures with the checked-in C source
produces byte-identical codestreams and matching lengths/SHA-256 values.

## GPU spot-color presentation

The common decoder now renders every supported integer spot plane at presentation. The private
all-channel surface records its RGB transfer explicitly: standalone XYB retains linear RGB;
blending and post-transform reference storage use original encoded RGB. In either case spots run
after reference storage and before output color conversion/association. Declaration order and
`mix = solidity * normalized_spot_sample` follow libjxl 0.12's
[spot stage](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/render_pipeline/stage_spot.cc)
and [render-pipeline ordering](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_cache.cc).
Only RGB changes, without clamping; raw extras and stored references remain untinted. The metadata
header describes ink RGB as linear, but the reference stage's placement makes its working domain
depend on prior blending/reference storage. That distinction is explicit rather than hidden in a
conversion round trip. Physical Modular/VarDCT engines export preserved planes; `GpuDecoder::wgpu`
routes Render through the common presentation stage, including a sequence containing one still.

`generate_extra_channels.c OUTPUT_DIR --spots` generates ten additional fixtures with libjxl 0.12.0.
The existing generator's other modes are unchanged. All contain nine declarations: Depth16,
SpotColour1, Alpha7, SpotColour12, SpotColour4, Thermal8, SpotColour6, SpotColour10 and Alpha15.
The five inks, in that order, use RGBA `(0.75,0.125,0.25,0)`, `(0.25,0.5,0.75,0.5)`,
`(1.5,-0.25,0.125,1.25)`, `(-0.125,1,0.375,-0.5)` and `(0.5,0.25,1,1)`. The original deterministic
integer-code formula and zero/full-coverage columns are retained. This exercises zero, unit,
negative and greater-than-one solidity, extended RGB, non-leading alpha and noncommuting inks.
Both source coding modes are generated for each suffix; extra-channel distance is zero.

| Suffix in `[vardct_]extras_spots_*.jxl.hex` | Canvas | Color | Orientation | Special case |
|---|---|---|---:|---|
| `rgb` | 33×7 | RGB12 | 6 | two unassociated alpha planes |
| `gray` | 17×9 | Gray16 | 8 | colored ink on gray |
| `thin` | 1×9 | RGB8 | 5 | associated first alpha, preserved invisible color |
| `resampled` | 37×17 | RGB12 | 7 | associated first alpha, color factor 2, extras factor 8, dimension shift 1 |
| `distributed` | 257×9 | RGB12 | 3 | effort 7 responsive and progressive AC |

Four integration tests in `tests/vardct_engine_gpu/extra_channels/spot.rs` and `spot_output.rs`
cover all ten new files, four existing single-spot stills and all seven extra-composition sequences
(the data-only sequence is a no-spot control). They check declaration-order RGB, associated and
unassociated alpha, linear output, planar BGRA, Apply/Keep, native RGB/RGBA at 8/12/16 bits, NV12,
odd packed 4:2:2, and BT.2020 constant-luminance P010. Whole and 43-byte fragmented async inputs
with a 1024-byte entropy window are byte-identical. Numeric native/F32 selection is identical
under Render/Preserve; VarDCT PQ/HLG still returns the explicit luminance-mapping error.

The optional test-only C decoder adds `--render-spots`. Direct rendered libjxl output checks the
original stills and each coalesced presentation. Gray-to-linear CMS discards colored spot channels,
and its transfer approximation for extended composed samples differs from the analytic sRGB
curve, so those linear comparisons independently transform libjxl's rendered sRGB channels.
Multiple-ink expectations independently apply the declared formula to libjxl's untinted color and
extra planes in the appropriate stage domain. Rust jxl 0.6 places output transfer before standalone
XYB spots, so it is not the numeric authority for that mode. The single-spot sequence relative-error
limit is `5e-4`; the multiple-spot formula uses `3e-6` for Modular and `2e-3` for VarDCT, with near-zero
alpha scaling removed before comparison. Native outputs use one code of Modular tolerance and
`ceil(max_code * 5e-5)` for the VarDCT resampled fixture. Target YUV comparison permits one code
and checks high-depth padding. No production CPU codec or image readback was introduced.

A GPU unit test independently isolates the packer: output admission succeeds with one byte less
than the required uniform+ink metadata, then metadata admission fails with exactly 352 bytes for
five inks or 192 bytes for Preserve. Repeating failure rolls back output reservations; releasing
pressure allows retry. Dropping pending work retains all inputs and metadata until GPU completion,
then only a caller-held output lease remains. `SpotColor` has compile-time 32-byte size, 16-byte
alignment and byte-16 RGBA offset checks. Preserve allocates no dummy table.

| File | Encoded bytes after hex decoding | SHA-256 of encoded file |
|---|---:|---|
| `extras_spots_distributed.jxl.hex` | 20860 | `c88cb37c2a9bcd3ff6fac41debe371821405477ed2f86fc09b56e71dfbc2182c` |
| `extras_spots_gray.jxl.hex` | 2009 | `84c3327e79e646489f7c64421f3dfdcb7529d1b5fc3f9b3a4df1fb63a790a070` |
| `extras_spots_resampled.jxl.hex` | 1012 | `49451e16a3a0c1b9656ed0bc7466edbc449933a4d25c56cca4e737c2fbc41156` |
| `extras_spots_rgb.jxl.hex` | 2903 | `843d6a5c7d356d2e77aab251bf84081aa0bc30c2e4656c69d16f1fafd1195e30` |
| `extras_spots_thin.jxl.hex` | 297 | `400c56f4482ed6e8c92407136b4803e49c679927c18ab1dfcd2e8a51f52cbd00` |
| `vardct_extras_spots_distributed.jxl.hex` | 22185 | `067778acdf2313bb98521ffb430776703b10537f53b49ddf68c10d5f59cb2eab` |
| `vardct_extras_spots_gray.jxl.hex` | 1866 | `15e251d85c8580ffeda5c7dadffc8977d6bbc2e54d2f6b62876b85ec3f28a610` |
| `vardct_extras_spots_resampled.jxl.hex` | 874 | `2f3ca859da87bc668c0cbd6ee33d1736c280a25d9a4f81af63712cbc5e4d74cf` |
| `vardct_extras_spots_rgb.jxl.hex` | 2666 | `3eb296c6251f7fe99f0f1830bd08729d159f364e197433c86cfb93eec38d8822` |
| `vardct_extras_spots_thin.jxl.hex` | 344 | `1756771316316c2ce46efc89ba071a9545be8ff521262a77f52751e36ec83d63` |

Validation on 2026-09-08 Apple M5/Metal: one serial all-target/all-feature workspace run passes
697 tests across all 24 targets, with one existing manual benchmark ignored and no failures.
Formatting, warning-free Clippy/rustdoc, all-target/all-feature checking, Rust 1.89 and the six-crate
WASM compile gate pass. Reference and Metal harnesses each pass 18 cases; indexed Gray8 U8 CPU
readback passes. Regenerating all ten new fixtures and the thirteen original stills with the same
C generator produces byte-identical files; the ten new lengths and SHA-256 values match above.
At that checkpoint floating source samples, remaining original color domains and pre-transform
patch references were open full-JPEG-XL roadmap gates. Floating source evidence follows below.

## Floating JPEG XL source precision and rendering

`crates/jxl_wgpu_decode/test-data/floating/` contains 181 encoded fixtures and independent libjxl
0.12.0 binary32 references. Regenerate the 416 hex files with
`cargo run -p jxl_wgpu_decode --example regenerate_floating -- [output-directory]`.
The example compiles offline C encoders/oracles; no production crate links libjxl or performs
CPU pixel reconstruction. Repeated generation on the same libjxl version is byte-identical.

The 154 `BITS-EXPONENT` cases cover every combination of 2–8 exponent and 2–23 mantissa bits.
Each 24×5 grayscale image contains both signs of zero, subnormals, minimum normal, maximum finite,
infinity, and multiple quiet/signaling NaN payloads, plus finite values. The `.f32.hex` file stores
each independently decoded binary32 word. For exponent width eight and total width 11–31,
libjxl 0.12's custom-float encoder cannot represent the subnormal test inputs correctly. The driver
therefore combines a valid custom-precision image header with a binary32 Modular frame containing
the desired raw custom words, with Palette and responsive transforms disabled and Zero prediction.
Frame boundaries come from the checked codestream inventory, not fixed byte offsets; libjxl
decodes and validates every resulting complete file. This workaround belongs only to corpus
generation. All other precision files are encoded directly by libjxl.

`all_floating_precisions_preserve_binary32_bits_through_gpu_decode` compares exact GPU words for
native scalar F32, fragmented 256-byte-window scalar F32, and unchanged-transfer RGB F32 output:
462 complete image decodes. It checks source precision metadata, including exponent width.

The 27 named rendering fixtures combine binary16/binary32/custom primary precision with independently
declared integer and floating extras: depth, selection, two alphas, spot, CFA, thermal, black and
optional planes. Four cases have integer primary images with floating extras, including direct native-alpha conversion. The cases cover Gray/RGB, associated alpha, rotated odd/thin images, 2×/4×/8×
resampling and dimension shifts, distributed entropy, Squeeze, a transform-free global-only
multi-pass Modular frame, and a real progressive-DC dependency. The last case omits extras because
libjxl disables progressive DC when extras exist. Encoder-downsampled floating planes use binary32
to preserve the encoder's arbitrary filter output; custom precision remains covered by unfiltered
and transformed fixtures. Five nine-layer animations cover hidden layers, crop intersections,
reference replacement and independent per-plane blend/alpha selectors.

For each case, `.f32.hex` contains every coalesced frame's RGBA and original extra planes;
`.spots.f32.hex` contains spot-rendered RGBA and `.associated.f32.hex` contains RGBA with source
association preserved. `floating_channels_resample_compose_and_render_against_libjxl` compares
every extra, base RGBA, rendered spots and preserved association under both whole and fragmented
256-byte input. The two GPU runs are word-identical and release their reservations. Reference
limits are `2e-6 × (1 + |reference|)` for scalar/alpha and `2e-5` for Modular color; VarDCT color
uses `0.003`. When unpremultiplication amplifies a tiny alpha, color error is measured after
undoing that scale with the existing `2^-26` floor, as in the associated-alpha corpus above.

`floating_color_quantizes_only_at_the_requested_integer_output` verifies nine still/composed/DC
cases against libjxl F32 followed by one RGB8/RGBA8 clamp/round, within one code. Mapping tests reject
integer normalization of floating sources, floating mapping of integer extras, and non-scalar or
non-F32 NativeFloat storage. A CPU topology gate verifies actual global-only, distributed and
Squeeze inventories. The precision ABI test fixes packed integer/binary16/binary32 values
and rejects invalid or overflowing declarations. These tests do not establish full source color
management, integer depths above 16, pre-transform patch references, or lossy/XYB Modular presentation.

Validation on 2026-09-08 Apple M5/Metal: 705 distinct tests pass across 26 workspace targets;
one existing manual benchmark remains ignored. The workspace run caught an integer-depth diagnostic
regression; after correction, its profile target, expanded floating target and all remaining targets
pass. Formatting, warning-free Clippy/rustdoc, all-target/all-feature checking, Rust 1.89 and the
six-crate WASM gate pass. Reference and Metal harnesses each pass 18 cases, and the indexed Gray8
U8 readback case passes. All 416 floating-corpus files regenerate byte-identically with libjxl
0.12.0; the extended generators also reproduce all 20 original integer still/composition fixtures.

## Integer JPEG XL precision through 31 bits

`crates/jxl_wgpu_decode/test-data/integer/` contains 82 encoded fixtures and their references
(286 files). Reproduce them with libjxl/libjxl_cms 0.12.0 using
`cargo run -p jxl_wgpu_decode --example regenerate_integer -- [output-directory]`.
The C generators are compiled offline with `-std=c11 -Wall -Wextra -Werror`; production never
links or selects the CPU codec. Integer and floating generation now share checked subprocess,
hex, and independent extra-plane oracle utilities in `examples/support/offline.rs`.

The 42 precision fixtures consist of 31 grayscale declarations, six independently declared
RGB/alpha combinations, three large 31-bit predictor cases and two 29-bit RCT cases. Their source
`.u32.hex` files record exact codes, including zero, maximum, adjacent endpoints, the 24-bit
mantissa boundary, large half-range values and deterministic full-range words. Cases with 257×9
images force Gradient, Weighted and predictor 13; the 129×5 RGB cases force RCT 6 and 41.
CPU inventory tests require real RCT jobs and a separate distributed Squeeze topology, rather
than trusting encoder settings or fixture names. The direct tests compare every source code,
including integer alpha rescaling computed independently in Rust `u64`, through whole input and
256-byte entropy windows supplied in 43-byte transport chunks. Selected alpha retains its own
native precision. The F32 tests compare independently decoded libjxl RGBA and scalar samples
within one binary32 ULP.

libjxl's public encoder [accepts integer precision only through 24 bits](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/encode.cc).
Its image API also converts pixels through F32. To preserve every source bit, the precision
generator stores raw words in binary32 Modular frames, then serializes explicit integer sRGB
image metadata. Frame boundaries come from inventory, not fixed byte offsets. The RCT cases use
29-bit/7-exponent-bit floating representation so the encoder retains enough working headroom
to actually emit RCT. No Palette is emitted, so changing sample interpretation does not alter
implicit Palette entries. The final integer codestream is parsed in full and independently
decoded by libjxl before any reference is saved.

The 40 rendering fixtures include 20 public-encoder streams with primary and extra precisions
17–24, then 20 variants with extended declarations. Fourteen variants cover each XYB primary depth
18–31 (17-bit input is covered by the original), completing the earlier 1–16-bit corpus. Six more cover Modular resampling/distribution, both-mode resampling, recursive DC, and
Modular/VarDCT animation at 31 bits. Extended variants locate fixed-width integer precision
fields by parsing the image grammar, increase each extra's declaration by seven bits, and clear
the 16-bit-working-buffer hint. They retain all encoded frame bytes, names, crops, blend selectors,
transforms and entropy. References are recomputed from these complete streams; changing a
Modular declaration changes its normalization denominator while XYB metadata does not rescale
the reconstructed color.

Rendering covers two independently associated alphas, depth, selection, spot, CFA, thermal,
black and optional planes, all five blend modes, 2×/4×/8× color and extra reconstruction, dimension
shifts, orientation, Squeeze/distributed streams, recursive progressive DC and seven nine-layer
animations. The shared `tests/support/rendering.rs` checks every selected extra, base RGBA, spots
and preserved alpha association against libjxl. Scalar/alpha tolerance is `2e-6 * (1 + abs(reference))`;
Modular color uses `2e-5` and VarDCT color uses `0.003`, with the established inverse-alpha
amplification adjustment. Whole and bounded fragmented outputs must be byte-identical; frame
counts, metadata and released reservations are checked. Eight additional color-output cases
exercise RGB8/RGBA8 and native 17–31-bit words, including zero high padding after composition.
RGB8/RGBA8 permit one code; native wide color retains the same normalized reconstruction
tolerance. Source metadata precision is not a claim of 31-bit accurate lossy reconstruction.

`unsigned_quantization_and_alpha_rescaling_are_exact_on_gpu` compares 34,224 arithmetic cases:
all 31×31 integer alpha depth pairs, endpoints and deterministic values, plus binary32 values
on both sides of half-code boundaries. Rescaling uses a Rust `u64` reference. Quantization uses
the exact rational value in Rust `u128`; even F64 multiplication can round a 55-bit product onto
the wrong side of a half-code boundary. The production WGSL uses two `u32` limbs, requiring no
F64 capability or new buffer. Extended scalar-packer tests cover 17/24/25/31-bit storage, negative
and overshoot values, decoded F32 inputs, all orientations, one-pixel axes, padded rows,
nonzero binding offsets, two-dimensional dispatch and guard bytes. Invalid integer declarations
and mismatched native storage remain typed failures.

The first validation caught two test-oracle assumptions: F64 was insufficient for an exact
half-code reference, and libjxl's chosen group size did not guarantee distribution. The reference
now evaluates exact rationals and generation explicitly selects 128-pixel groups. Both corrected
checks pass along with all precision/rendering cases. The decoder's unrelated original-color,
lossy/XYB Modular, pre-transform reference and full conformance requirements remain open.

Validation on 2026-09-08 Apple M5/Metal: 713 distinct tests pass across 28 workspace targets;
one existing manual benchmark remains ignored. A serial workspace run passes 712 tests, and
the subsequently expanded integer target passes all six tests across 40 rendering fixtures;
the profile target and expanded CPU topology gate also pass. Formatting, warning-free
Clippy/rustdoc, all-target/all-feature checking, Rust 1.89 and the six-crate WASM gate pass.
Reference and Metal harnesses each pass 18 cases, and the indexed Gray8 U8 readback case passes.
All 286 integer-corpus files regenerate byte-identically with libjxl 0.12.0. The shared generator
refactor reproduces all 416 floating-corpus files and all 20 original integer still/composition
fixtures byte-identically.

## Lossy Modular color and restoration

`crates/jxl_wgpu_decode/test-data/lossy_modular/` contains 19 complete encoded streams and
independent libjxl references (76 files, 5,388,441 bytes). Reproduce them with
`cargo run -p jxl_wgpu_decode --example regenerate_lossy_modular -- [output-directory]`, using
libjxl/libjxl_cms 0.12.0. The C generators use only the offline public encoder API and are compiled
with `-std=c11 -Wall -Wextra -Werror`; production uses no CPU image codec. Fifteen stills cover XYB
and original-sRGB Modular, Gray/RGB, all eight orientations, independent alpha, nine mixed extra
planes, binary32 source metadata and floating extras, one-pixel axes, 2×/4×/8× reconstruction and
257×17 distributed Squeeze. Four nine-layer animations include RGB, gray, resampled color/extras
and floating samples, with all five blend modes, independent extra-channel references and alpha.

GPU color normalization implements the [Modular Y/X/(B-Y) and LF dequantization contract](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_modular.cc).
Gaborish and EPF precede resampling and the shared RGB/XYB/JPEG color packer. Modular EPF uses a
[frame-constant inverse sigma](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_frame.cc)
with the [normative minimum sigma validation](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/loop_filter.cc).
A CPU topology gate parses every physical frame, requires actual Modular coding, verifies the
XYB/original color distinction, every EPF iteration count and real multi-group Squeeze, and checks
that filtering has a frame arena when group boundaries are crossed. Compared with EPF disabled,
the EPF1/2/3 references change 67/130/132 color words; the corpus actually exercises the filters.

`tests/lossy_modular.rs` compares every delivered extra plane, base RGBA, spot presentation and
preserved alpha against checked-in libjxl output. All presentations must match between complete
input and 256-byte entropy windows supplied in 43-byte transport chunks. Color tolerance is
`1e-4 * (1 + abs(reference))`; extras and alpha retain `2e-6`. For unpremultiplied RGB, error is
measured after undoing the output alpha division with the established `2^-26` floor. Eight further
cases exercise final RGB8/RGBA8 and 16-bit RGBA quantization, including resampled and composed
outputs; tolerances are one and seven codes respectively. Every session must release its shared
reservations after its outputs are dropped.

The Rust `jxl` 0.6 oracle independently verifies all 15 stills and the initial Replace presentation
of each animation. It preserves stored associated color, so that case is compared to libjxl's
explicit preserved-alpha output. Subsequent multi-extra reference chains encounter the previously
pinned clamped-Multiply operand reversal and are validated against live libjxl and GPU output.
The test also requires live libjxl output to equal the checked-in reference when libjxl is installed.

Initial GPU comparison exposed spot inks being applied after the XYB transfer function. The common
executor now retains unreferenced Modular XYB in linear RGB, like VarDCT, and applies spots before
presentation transfer; referenced/blended frames retain original-sRGB values. The finalizer copies
already converted color without applying the transfer a second time. The public `color_output`
module replaces `vardct::output` and accepts explicit RGB, XYB and JPEG-component sources. The
workspace ABI gate also caught the generic scheduler's old EPF padding-field declaration; both
Rust records now match the constant-sigma WGSL layout. Original ICC/wide-gamut/HDR color management,
pre-transform patch references and the remaining render graph still have independent completion gates.

Validation on 2026-09-08 Apple M5/Metal: the serial all-target/all-feature workspace run passes
719 tests, with one existing manual allocation benchmark ignored and no failures. Formatting,
warning-free Clippy/rustdoc, Rust 1.89 and the six-crate WASM compile gate pass. Reference and Metal
harnesses each pass 18 cases; indexed Gray8 U8 CPU readback and the final output-quantization
rerun pass. All 76 new corpus files, 416 floating-corpus files, 286 integer-corpus files and
20 original integer still/composition fixtures regenerate byte-identically with libjxl 0.12.0.
