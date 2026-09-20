# JPEG reconstruction metadata corpus

These 36 paired inputs test `jbrd` metadata parsing and canonical emission. The original JPEG
is the byte oracle; pixel equality is insufficient. Tests use the immutable identities and
explicit scan/padding expectations in
[`corpus::jpeg_reconstruction`](../../../../tools/jxl_test_support/src/corpus/jpeg_reconstruction.rs).
The corpus contains 140 scans and also supplies unchanged originals to the
[GPU coefficient checks](../../../../docs/CONFORMANCE_CORPUS.md#gpu-jpeg-quantizer-and-coefficient-binding).

## Sources and reproduction

`bench_oriented_brg`, `cafe` and `grayscale_jpeg` reuse the unchanged JPEG XL inputs in
[`official_conformance`](../../../jxl_wgpu_decode/test-data/official_conformance/README.md),
pinned to libjxl/conformance `b1d0f990b03e57bf6d137c365cd5dc8b470b9191`. Their `source.jpg`
files match the original `test.json`'s `reconstructed.jpg` SHA-256 exactly. The objects are at
`https://storage.googleapis.com/jxl-conformance/objects/<sha256>` using that published hash.
Bench and grayscale are CC0; cafe is ITU-T T.24 with the upstream test-data reproduction
permission. Upstream code/data notices and attribution remain in the linked corpus.

The remaining 33 original JPEGs are development fixtures generated with libjpeg-turbo 3.2.0
and explicit byte-preservation edits. Each checked-in `source.jpg` is the input, not output
from the code under test. Recreate its JPEG XL pair with libjxl/cjxl 0.12.0:

```console
cjxl source.jpg input.jxl --lossless_jpeg=1 -d 0 -e 7 --num_threads=2
```

Regeneration is an explicit fixture mutation, separate from running the tests. Preserve the
original JPEG hashes and inspect any encoder-version-dependent JXL changes before replacement.
The JPEG seeds use `cjpeg -quality 83 -strict`, a 65×33 P6 source, and these selectors:

| Case | Additional cjpeg options |
|---|---|
| gray_restart | `-grayscale -restart 1B` |
| gray_progressive_restart | `-grayscale -progressive -restart 1B` |
| rgb_sequential / rgb_progressive | `-rgb`, with `-progressive` for the latter |
| ycbcr444_sequential / ycbcr444_progressive | `-sample 1x1,1x1,1x1`, with `-progressive` for the latter |
| ycbcr420_sequential | `-sample 2x2,1x1,1x1` |
| ycbcr420_progressive_restart | `-sample 2x2,1x1,1x1 -progressive -restart 3B` |
| ycbcr440_progressive_restart | `-sample 1x2,1x1,1x1 -progressive -restart 1B` |

For zero-based pixel coordinates `(x,y)`, the P6 components are
`(19*x + 7*y + (x*y)%29) & 255`, `(5*x + 31*y + (x^y)) & 255`, and
`(43*x + 13*y + (x*y)%61) & 255`. The exact header is `P6\n65 33\n255\n`;
the complete PPM SHA-256 is `511b8608ede285a371944b7158185f0aed9c6af566920e5e6c9146cafdf62fff`.

Twelve variants preserve the RGB seed's scan entropy (the final row uses the YCbCr444 seed):

| Case | Source JPEG edit |
|---|---|
| empty_dht | Insert an empty DHT before the first DHT. |
| merged_dht | Concatenate DHT payloads into one marker. |
| unused_quant | Copy the first DQT as an unused table with selector 3. |
| custom_selectors | Change component IDs to 17/83/255 and all quantization/Huffman selectors to 3. |
| quant16 | Change SOF0 to SOF1, expand quantization entries to 16 bits and add 256. |
| icc_chunks | Insert three ICC APP2 chunks, split at profile offsets 101 and 334. |
| exif_rotated | Insert big-endian TIFF Exif orientation 6. |
| xmp_comment | Insert XMP plus binary COM and opaque APP15. |
| metadata_combined | Combine the ICC, Exif, XMP, COM and APP15 edits. |
| marker_fill | Insert three `ff` bytes immediately after SOI. |
| tail_bytes | Append `preserved\0tail\xff\x01` after EOI. |
| merged_ycbcr | Concatenate each DHT/DQT family into one marker. |

The ICC bytes come from the existing `splines/animation_spline.icc`. The complete exact APP,
COM, fill and tail bytes are retained in each original JPEG. These are preservation probes;
native acceptance of a noncanonical JPEG is not a normative JPEG syntax claim.

Six entropy-edge sources exercise gray restart padding changed to all zero or alternating
bits (150 bits), one/two/three extra ZRL symbols before EOB in an otherwise zero 8×8 gray
block, and a progressive 2057×1025 gray image with 33,282 zero blocks. Zero-block sources
use constant P5 value 128 and the same cjpeg quality. Six further variants change only unused
progressive padding to zero/alternating bits: gray restart (6 scans, 1076 bits), RGB (14 scans,
63 bits), and YCbCr420 restart (10 scans, 318 bits). Byte stuffing is recalculated after padding
edits. Independent libjpeg coefficient extraction verified that those padding edits preserve
coefficients; the checked-in test here exercises metadata and original-JPEG bytes.

## Executable oracle

```console
JXL_REQUIRE_NATIVE_ORACLES=1 cargo test --locked -p jxl_gpu_bitstream -- --test-threads=1
```

[`../jpeg_reconstruction_oracle/main.cpp`](../jpeg_reconstruction_oracle/main.cpp) compiles
with C++17 and `pkg-config --cflags --libs libjxl`. The test requires these native tools;
absence fails instead of silently skipping. It subscribes only to JPEG reconstruction and
full-image completion, sets a bounded JPEG output buffer, and requires a final success event.
Pixel-output requests and incomplete/missing reconstruction fail without publishing a file.
The oracle remains outside the production dependency graph.

Each source is reconstructed unmodified and after replacing only `jbrd` with Rust output at
Brotli quality/window pairs 0/10, 6/22 and 11/24. All 144 results must equal the original JPEG
bytes. Codestream and other auxiliary payload bytes stay unchanged. Each official source also
tests missing `jbrd`, an empty payload, two truncations and two trailing bytes: 18 native
rejections, including a control that forbids pixel fallback. Portable tests additionally cover
every payload truncation, all record views, canonical stability, exact logical ownership and
one-byte-short limits, marker/entry/decoded-body bounds, window/ratio bounds, malformed grammar,
and duplicate or forbidden wrapped boxes.

This corpus establishes metadata interoperability. Production JPEG reconstruction still needs
validated GPU frame geometry, quantization, coefficients, entropy and output ownership.
