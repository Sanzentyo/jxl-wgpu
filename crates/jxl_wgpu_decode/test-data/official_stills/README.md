# Official still-image conformance references

These twelve inputs, ICC profiles and JSON descriptors are unchanged files from the
[JPEG XL conformance corpus](https://github.com/libjxl/conformance/tree/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases)
at commit `b1d0f990b03e57bf6d137c365cd5dc8b470b9191`. The upstream BSD license is in
[LICENSE](LICENSE).
The upstream [source-image list](https://github.com/libjxl/conformance/blob/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases/README.md)
identifies these twelve source images as CC0 and permits test-data reproduction for testing.

Each `reference.npy.gz` is a lossless `gzip -n` compression of the original
`reference_image.npy` object. The uncompressed SHA-256 is recorded in its adjacent,
unmodified `test.json`. The integration test pins the input and descriptor SHA-256,
then verifies every loaded profile and decompressed reference before GPU execution.
No reference pixel or error limit is regenerated from this decoder.

| Case | Width × height × components | Per-channel RMSE limit | Absolute peak limit |
|---|---:|---:|---:|
| `alpha_nonpremultiplied` | 1024 × 1024 × 4 | 6.1035e-05 | 6.1035e-05 |
| `alpha_premultiplied` | 1024 × 1024 × 4 | 3.815e-06 | 3.815e-06 |
| `alpha_triangles` | 1024 × 1024 × 4 | 0.001953125 | 0.001953125 |
| `blendmodes` | 1024 × 1024 × 4 | 0.004 | 0.0001 |
| `delta_palette` | 555 × 751 × 3 | 0.000976562 | 0.000976562 |
| `grayscale` | 200 × 200 × 1 | 0.0001 | 0.004 |
| `grayscale_jpeg` | 200 × 200 × 1 | 1.0e-05 | 0.004 |
| `lossless_pfm` | 500 × 500 × 3 | 0.0 | 0.0 |
| `lz77_flower` | 834 × 244 × 3 | 0.000976562 | 0.000976562 |
| `patches_lossless` | 1600 × 1096 × 4 | 0.000976562 | 0.000976562 |
| `spot` | 600 × 400 × 6 | 3.815e-06 | 3.815e-06 |
| `sunset_logo` | 924 × 1386 × 4 | 0.000244141 | 0.000244141 |

Comparisons use all components, preserve associated alpha and disable spot rendering,
as in the official runner. Color requests explicitly select each reference encoding; `spot`
compares original numeric components and retains its exact ProPhoto ICC bytes in the image
inventory. Its v2 profile has header illuminant Z = `0xd32b`, rather than canonical D50 `0xd32d`:
no CMS transformation is requested for this reference. A separate negative test requires color
conversion to reject the profile before GPU admission. Alpha and both spot planes remain
independent components. The lossless F32 case includes negative and extended values:
its zero-error limits require exact IEEE-754 words. No clipping or lax comparison is
applied. Whole and fragmented 16 KiB entropy-window decoding must also agree exactly.
Every output is read after dropping its session, and all budget reservations must be released
after dropping the retained frame. Additional independent f64 comparisons exercise Linear,
BT.709 and BT.2020 output from the extended signed `alpha_triangles` samples without changing
the official sRGB comparison or its limits.

The grayscale XYB case requests the reference linear Gray output and separately preserves its
embedded original printer ICC. `grayscale_jpeg` compares the original normalized Gray samples;
this pixel check does not cover its required byte-identical JPEG reconstruction. `lz77_flower`
compares original normalized RGB values under its enumerated gamma declaration. Its generated
original and reference ICC files are retained and hash-checked, but this test does not claim an
exact-byte generated-profile export. `patches_lossless` requests its unchanged original ICC as
the output profile and preserves alpha. `blendmodes` and `delta_palette` use sRGB. All profiles
and published limits remain unchanged.

The official objects can be downloaded from
`https://storage.googleapis.com/jxl-conformance/objects/<sha256>` using the digests in
`test.json`. Recreate compressed references with
`gzip -n -c reference_image.npy > reference.npy.gz`. Inputs and descriptors come
from the pinned corpus checkout above.
