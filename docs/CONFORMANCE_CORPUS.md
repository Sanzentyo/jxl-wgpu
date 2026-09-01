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
the appended descriptor metadata, and the full `ShaderParams` word-order test fixes its new 240-byte
storage stride and channel-layout offset without inspecting shader source text.

## Procedural VarDCT encoder matrix

The `jxl_wgpu_encode` actual-adapter suite generates its VarDCT inputs in memory rather than
checking in large duplicate raster files. Odd 257x17, asymmetric 513x259 and 768x513, horizontal
2056x256 and vertical 256x2056 LF-boundary images exercise padded edge blocks and row/column group
ordering. The two boundary images contain two standard LF groups; the GPU artifact stores one
validated fragment descriptor per group and resets the clamped-Gradient predictor at that boundary.
Rust `jxl` and installed `djxl` must decode each emitted codestream, while the stock GPU decoder plus
explicit readback must differ from Rust `jxl` by at most one RGB8 code.

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
40-byte cap. More than two windows share the same 64/128-byte state across the LF-to-HF transition;
there is no intermediate status map and the final packet command shares the first downstream
submission. Runtime-neutral async decode/readback must agree with Rust `jxl` and optional `djxl`
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
236-byte parameter record and the 32/48/112-byte, 16-byte-aligned resume layouts. Every composed
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
