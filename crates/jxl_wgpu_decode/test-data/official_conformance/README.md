# Official conformance references

The [GPU target](../../tests/official_conformance/main.rs) exercises these references;
measured results and validation scope are recorded in the
[conformance corpus](../../../../docs/CONFORMANCE_CORPUS.md#official-conformance-descriptor-coverage-2026-09-20).

The inputs, ICC profiles, JSON descriptors and uncompressed NPY bytes are unchanged from
[libjxl/conformance](https://github.com/libjxl/conformance/tree/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases)
at commit `b1d0f990b03e57bf6d137c365cd5dc8b470b9191`. The upstream BSD license is in
[LICENSE](LICENSE). Its [source-image list](https://github.com/libjxl/conformance/blob/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases/README.md)
permits reproduction of the bitstreams and decoded references for testing; source-image
credits and licenses are listed below.

This directory has 25 unique inputs and 37 directly compared descriptors, including 12
`_5` variants. Three additional descriptors are pinned to the existing
[spline](../splines/README.md) and [CMYK](../cmyk/README.md) GPU families. Together they
represent all 40 upstream descriptors across 27 unique inputs. This coverage does not
establish full JPEG XL conformance. The separate GPU JPEG byte-output target now matches the
three published original JPEGs exactly; exact generated ICC export remains open.

The test pins each input and descriptor hash before reading reference pixels, then checks
all loaded ICC and uncompressed NPY hashes. Storage compression uses `gzip -n` without
changing a reference pixel. `progressive/reference.npy.gz.part00` through `.part02` are
consecutive pieces of one original gzip stream, each at most 48 MiB. The reader joins them
before decompression and validates the original NPY digest. `mul_no_extra_channels` has an
NPY stored directly in upstream Git, rather than named in `test.json`'s hash map; its exact
844-byte Git blob is independently pinned as
`ff062d75fbcbf1d75bc99d13549b08202ef401f3150e3aed7cd015b711ba509b`.

| Case | Frames | Width × height × components | Reference encoding | Per-channel RMSE limit | Absolute peak limit |
|---|---:|---:|---|---:|---:|
| `alpha_nonpremultiplied` | 1 | 1024 × 1024 × 4 | sRGB | 6.1035e-05 | 6.1035e-05 |
| `alpha_premultiplied` | 1 | 1024 × 1024 × 4 | sRGB | 3.815e-06 | 3.815e-06 |
| `alpha_triangles` | 1 | 1024 × 1024 × 4 | sRGB | 0.001953125 | 0.001953125 |
| `animation_icos4d` | 48 | 128 × 128 × 4 | sRGB | 0.0001 | 0.005 |
| `animation_newtons_cradle` | 36 | 480 × 360 × 4 | sRGB | 0.000976562 | 0.000976562 |
| `bench_oriented_brg` | 1 | 606 × 500 × 3 | original numeric | 1.0e-05 | 0.004 |
| `bicycles` | 1 | 1024 × 631 × 3 | sRGB | 0.000976562 | 0.000976562 |
| `bike` | 1 | 2048 × 2560 × 3 | BT.709 | 0.0001 | 0.007 |
| `blendmodes` | 1 | 1024 × 1024 × 4 | sRGB | 0.004 | 0.0001 |
| `cafe` | 1 | 1280 × 1600 × 3 | original numeric | 1.0e-05 | 0.004 |
| `delta_palette` | 1 | 555 × 751 × 3 | sRGB | 0.000976562 | 0.000976562 |
| `grayscale_jpeg` | 1 | 200 × 200 × 1 | original numeric | 1.0e-05 | 0.004 |
| `grayscale_public_university` | 1 | 2880 × 1620 × 1 | original numeric | 0.000976562 | 0.000976562 |
| `grayscale` | 1 | 200 × 200 × 1 | linear Gray | 0.0001 | 0.004 |
| `lossless_pfm` | 1 | 500 × 500 × 3 | sRGB | 0.0 | 0.0 |
| `lz77_flower` | 1 | 834 × 244 × 3 | original numeric | 0.000976562 | 0.000976562 |
| `mul_no_extra_channels` | 1 | 8 × 8 × 3 | sRGB | 0.0001 | 0.004 |
| `noise` | 1 | 500 × 606 × 3 | sRGB | 0.0001 | 0.004 |
| `opsin_inverse` | 1 | 500 × 606 × 3 | sRGB | 0.0001 | 0.004 |
| `patches_lossless` | 1 | 1600 × 1096 × 4 | original ICC | 0.000976562 | 0.000976562 |
| `patches` | 1 | 1600 × 1096 × 4 | linear RGB | 0.0001 | 0.004 |
| `progressive` | 1 | 4064 × 2704 × 3 | linear RGB | 0.0001 | 0.02 |
| `spot` | 1 | 600 × 400 × 6 | original numeric | 3.815e-06 | 3.815e-06 |
| `sunset_logo` | 1 | 924 × 1386 × 4 | sRGB | 0.000244141 | 0.000244141 |
| `upsampling` | 1 | 800 × 600 × 4 | sRGB | 0.0001 | 0.004 |

`variants/` contains the unchanged `_5` descriptors. Every variant has its own pinned input,
descriptor, ICC and NPY identities and original limits. `animation_icos4d_5`, `patches_5` and
`upsampling_5` have different NPY pixels, stored here separately; both original references
are compared against the same GPU output. Other variants share their primary's exact
reference objects. No second GPU decode is needed for an alternate tolerance. The existing
60-frame spline GPU test retains RMSE 0.0001 / peak 0.004; an identity test establishes that
`animation_spline_5` uses the same objects and metadata with limits 0.02 / 0.06. The existing
CMYK test retains its five independent channel comparisons at 0.000976562.

Comparisons include all components, preserve associated alpha and disable spot rendering,
as in the upstream runner. Whole and fragmented 16 KiB entropy-window decoding must agree
bit-for-bit. Animations compare every final frame's name, duration, timestamp and final flag,
plus timebase, loop and timecode metadata. Frames are held past session destruction; all
frame reservations must release after the retained outputs are dropped. The fixture target
uses explicit 2 GiB transient limits for its largest image, without changing production
budget defaults.

Reference encodings are explicit. Gray XYB requests linear Gray while retaining its original
printer profile. JPEG Gray, oriented BRG, Cafe, public-university Gray and LZ77 flower use the
original numeric component domain. Bike uses enumerated BT.709; Patches and Progressive use
reference linear RGB. Lossless Patches requests its unchanged original ICC as output. Other
color cases request sRGB. Color values are neither clipped nor compared with widened bounds;
`lossless_pfm`'s zero limits require exact IEEE-754 words. Additional independent f64 transfer
comparisons retain the original signed `alpha_triangles` sRGB check.

Spot's original profile has a noncanonical D50 illuminant at offset 68, and oriented BRG has
nonzero reserved header bytes at offset 84. Their numeric comparisons retain the original
profile bytes without requesting CMS conversion. Separate color-request tests require typed
ICC rejection before GPU admission, including while the entire memory budget is held. Those
profiles are not weakened or replaced to make a color conversion pass.

The generated ICC files for enumerated gamma, BT.709 and Gray remain unchanged reference
objects; inventory checks verify the original declarations. They are not evidence of an
exact-byte generated-profile export. The three JPEG-origin cases compare pixels in this target;
[GPU original-JPEG reconstruction](../../../../docs/CONFORMANCE_CORPUS.md#gpu-original-jpeg-byte-reconstruction)
separately checks every byte of their published original JPEG objects.

## Source-image credits

- CC0, per upstream: alpha_nonpremultiplied, alpha_premultiplied, alpha_triangles,
  animation_icos4d, bench_oriented_brg, bicycles, blendmodes, delta_palette, grayscale,
  grayscale_jpeg, lossless_pfm, lz77_flower, mul_no_extra_channels, noise, opsin_inverse,
  patches, patches_lossless, spot and sunset_logo. Existing animation_spline and CMYK
  source images are also CC0.
- animation_newtons_cradle: Dominique Toussaint (DemonDeLuxe), modified by Scetoaux;
  [source and attribution](https://commons.wikimedia.org/wiki/File:Newtons_cradle_animation_book_2.gif),
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- upsampling: Daniel G. (2005), Ed g2s (2009), CyberShadow (2019);
  [source and attribution](https://commons.wikimedia.org/wiki/File:PNG_transparency_demonstration_1.png),
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- bike and cafe: ITU-T T.24, per upstream's source list and test-data reproduction permission.
- grayscale_public_university: Arnold and Richter Cine Technik GmbH,
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), per upstream.
- progressive: Copyright © 2003–2007 Microsoft Corporation. Its full upstream copyright and
  permission notice is retained unchanged in [progressive/source](progressive/source).

The conformance inputs and references are imported unchanged; no new image render or
pixel adaptation was made. `_5` variants inherit the corresponding source-image credit.

## Reproducing storage

Inputs, JSON descriptors and the small Multiply NPY come from the pinned Git tree. Other
objects are addressed by their unmodified `test.json` SHA-256 values at
`https://storage.googleapis.com/jxl-conformance/objects/<sha256>`. Recreate ordinary storage
with `gzip -n -c reference_image.npy > reference.npy.gz`. For Progressive, split that gzip
stream into consecutive 48 MiB chunks; joining all three chunks must recover the same
compressed bytes, and decompression must recover the original published NPY hash.
