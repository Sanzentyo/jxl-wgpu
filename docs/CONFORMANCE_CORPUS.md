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

## Procedural VarDCT encoder matrix

The `jxl_wgpu_encode` actual-adapter suite generates its VarDCT inputs in memory rather than
checking in large duplicate raster files. Odd 257x17, asymmetric 513x259 and 768x513, horizontal
2056x256 and vertical 256x2056 LF-boundary images exercise padded edge blocks and row/column group
ordering. The two boundary images contain two standard LF groups; the GPU artifact stores one
validated fragment descriptor per group and resets the clamped-Gradient predictor at that boundary.
Rust `jxl` and installed `djxl` must decode each emitted codestream, while the stock GPU decoder plus
explicit readback must differ from Rust `jxl` by at most one RGB8 code.

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
`vardct_engine_gpu::ordinary_cjxl_local_trees_complete_through_two_stage_frame_engine` runs the
same generated codestream through the stock frame engine. It requires two queue submissions, the
typed pre-HF `UnvalidatedOutputNotSubmitted` handoff error, blocking and runtime-neutral async RGB8
results within one code of Rust `jxl` and optional `djxl`, and complete shared-budget release after
normal consumption or cancellation at the LF stage. The effort-7 stream also exercises X=5/B=5
quant-matrix scales; a lower-level actual-GPU artifact test observes non-default scale multipliers
for all three channels directly in the resident resource vectors.

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
