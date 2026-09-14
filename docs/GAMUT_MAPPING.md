# Explicit RGB gamut mapping

`GpuOutputRequest::with_gamut_mapping(GamutMapping)` selects a GPU presentation policy for
an output with enumerated RGB chromaticities. It applies to RGB, gray and YUV storage, including
ICC input converted to that RGB space. The default request preserves the existing extended F32
color conversion. ICC device output instead uses its selected profile method/rendering intent;
this RGB policy rejects ICC, numeric, and unspecified native output targets before submission.

```rust
use jxl_gpu_protocol::{GamutMapping, LuminanceRange};
use jxl_wgpu_decode::{GpuOutputRequest, vardct_rgb8_format};

let output = GpuOutputRequest::color(vardct_rgb8_format())?
    .with_tone_mapping(LuminanceRange::new(0.0, 80.0).unwrap())
    .with_gamut_mapping(GamutMapping::default())?;
# Ok::<(), jxl_wgpu_decode::Error>(())
```

The generic render graph exposes the same policy through
`ImageOutputRequest::with_gamut_mapping`. Low-level `ImageOutputParams::with_gamut_mapping`
accepts `None` to remove it without losing the original identity-conversion shortcut.

## Geometry and ordering

The policy follows the linear RGB geometry of the pinned libjxl `GamutMapScalar` primitive.
`GamutMapping::new(preserve_saturation)` accepts a finite value in `[0,1]`; the default is 0.1.
For an out-of-gamut color, compute its luminance using the target primary matrix. Intersect the
line from that color to the equal-luminance neutral with the lower and upper faces of the RGB
unit cube. Blend the required neutral fractions according to the preference, then normalize any
remaining highlight by its largest component. Zero favors luminance preservation and one favors
saturation. Already in-gamut colors pass through the primitive unchanged. A target white outside
its primary triangle has negative luminance weights and is rejected for this policy.

Outside the cube, nonpositive luminance maps to black. The WGSL scales extended finite RGB before
subtraction to avoid overflow. An active lower cube face is evaluated as an exact zero using
differences from the minimum component, preventing cancellation residue from becoming visible
through PQ/HLG. Integer sign checks prevent negative subnormal words from passing a comparison
which an adapter may flush to zero. The mapped RGB is bounded to `[0,1]` before output transfer.
Nonfinite color processing and every possible F32 conditioning case remain separate conformance
work; unmodified numeric samples and disabled color-conversion paths retain their existing rules.

Mapping runs after reference storage, composition, spot rendering and explicit tone mapping,
in target-primary linear RGB. Output transfer, alpha association, YUV conversion, chroma
subsampling and quantization follow it. Alpha and other extra channels are not gamut mapped.
Internal reference surfaces ignore the presentation request, avoiding accumulation across frames.

When [tone mapping](TONE_MAPPING.md) protects light below the image's absolute/relative
`linear_below` threshold, that protection also skips gamut mapping. If the threshold reaches
either source or target white, the tone policy preserves absolute light throughout, and gamut
mapping is skipped throughout as well. This priority avoids a decreasing discontinuity above a
protected region which cannot fit the display. Protected F32 output can consequently exceed the
unit cube; integer output still quantizes and clips to its storage range. Without an explicit
tone request, the gamut policy applies over the entire RGB range.

## GPU contract

The shared image-output uniform is **304 bytes**. `GamutMappingParams` occupies bytes 288–303:
one vector containing normalized target-primary luminances and the preference. A negative
preference disables the operation. The earlier 48-byte tone record retains its offset 240;
tone mode 5 distinguishes globally protected absolute light from ordinary fitted-range scaling.
The codec source record remains 160 bytes, so the two conversion uniforms total **464 bytes**.
The resident ICC dispatch remains 320 bytes and its tone payload remains 48 bytes.

The mapping itself adds no image buffer, dispatch, readback or storage binding. Decoder requests
use the common final presentation surface, whose existing buffers and dispatches are still
required and fully admitted. All image-domain work stays on the GPU. Host work only validates
the output policy and computes luminance coefficients from color metadata.

## Evidence and remaining work

The [native generator](../crates/jxl_wgpu/test-data/gamut_mapping_generator/README.md) provides
5,000 libjxl records for BT.709, BT.2020 and Display-P3, five preferences, a signed RGB grid and
strict cube-boundary neighbors. Actual Apple M5 Metal runs compare **45,000 color components**
across 1-, 32- and 256-invocation workgroups against native and independent F64 intersections,
with a fixed `4e-6` linear allowance declared before execution. Guard regions, exact in-gamut
primitive values and alpha are checked. Independent extended cases include negative luminance
and values through ±1e30. RGB8, planar BGRA/RGBA F32, NV12 and 12-bit I444 tests cover transfer
ordering and packing; F32 comparisons keep the original `5e-5` encoded allowance.

All **56 existing HDR streams / 80 images** run with and without tone mapping, three output
transfers, both layouts and whole/bounded fragmented input. Their **1,920 presentations** compare
**3,362,688 color components** to native reconstruction followed by independent F64 conversion,
tone and gamut intervals. Relative/absolute, zero, partial and globally protected thresholds are
included. Progressive outputs remain immutable, final bytes agree across input/layout choices,
alpha words match unmapped output, and input/GPU budgets return to zero. Eight embedded ICC
Modular/VarDCT × RGB/Gray × original/XYB sources add 64 presentations and 29,376 color components.
The existing source/profile/reference files remain byte-identical.

Exact admission, repeated pressure/retry, program sharing, reversed completion and cancellation
tests also include gamut mapping after ICC and spot processing. Wider HDR feature/LF and CMYK
combinations, requested-profile tone/gamut policy, automatic display adaptation, display textures,
full numeric/ISO precision and the remaining codec/container/encoder gates stay open in the
[full JPEG XL roadmap](FULL_JPEG_XL_ROADMAP.md).
