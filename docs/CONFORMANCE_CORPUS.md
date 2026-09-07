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

## Shared VarDCT color output

The decoder and render graph use one shared GPU color/layout lowering and word-owned packing
fragment. `vardct_engine_gpu/color_output.rs` reuses the checked-in depth/orientation fixtures below;
no new image provenance is introduced. `generic_color_outputs_preserve_oriented_high_depth_vardct_precision`
tests all 20 color VPI pitch-linear forms plus I444/I422/I420, NV21/NV42, P010/P012/P016,
12-bit planar I420, and linear BGRA: 30 layout/transfer choices. The input is the 16-bit,
two-LF-group fixture with orientation 5, producing 17×2056 pixels. Source float RGB from Rust `jxl`
and explicit-sRGB `djxl` PFM is independently converted by the development-only scalar
`jxl_gpu_formats::convert_rgb_f32` oracle after any required SDR transfer conversion.

Every case runs whole-input blocking and fragmented-input async completion with a 256-byte entropy
cap. Exact layout metadata, shared 320-byte output/source uniform accounting, four-byte-rounded
output leases, zero unused sample bits and plane gaps, equality between upload policies, and full
budget release are required. Comparisons operate on stored sample codes rather than individual
bytes, including 16-bit words and 10/12-bit alignment. On Apple M5/Metal (2026-09-07), the maximum
difference is one code at 8–12 bits; at 16 bits it is one versus Rust `jxl` and three versus `djxl`.
The regression threshold is one at 8–12 bits and four at 16 bits (less than 0.000062 normalized).

`generic_color_output_combines_jpeg_gray_resampling_and_recursive_dc` repeats I420, P016, and linear
BGRA for rotated 12-bit gray with 4× resampling, a three-frame 16-bit gray DC chain, and the odd
oriented 4:2:0 JPEG transcode. `generic_color_output_converts_d65_primaries_against_djxl` requests
Display-P3/sRGB and BT.2020/BT.709 output, compares planar BGRA against `djxl` PFM explicitly
requested as `RGB_D65_DCI_Rel_SRG` and `RGB_D65_202_Rel_709`, and observes at most one code of
difference. Existing RGB8 dual-oracle cases continue to pass through the shared shader.

The packer GPU test checks 3×1 and 1×3 in every orientation with interleaved RGB and padded planar
RGB/RGBA. Plane starts can be unaligned and occupy a preceding plane's unused final-row tail;
payload, opaque alpha, zero padding, and untouched output guard bytes are checked independently.
Typed negative tests reject inconsistent extents/logical sizes, limited-range RGB, and PQ/HLG
without an explicit luminance mapping. Numeric/float RGB output, arbitrary ICC conversion, HDR
luminance mapping, Modular orientation, and extra-channel decoding remain separate coverage gaps.

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
do not claim Modular orientation, keep-orientation controls, arbitrary output formats, or full
JPEG XL conformance.

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
