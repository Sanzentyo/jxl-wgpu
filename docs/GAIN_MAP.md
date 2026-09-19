# JPEG XL HDR gain maps

`jxl_gpu_bitstream::gain_map` reads and writes version-zero `jhgm` bundles and exact
ISO 21496-1 rational parameters. `GpuDecoder::decode_gain_map` reconstructs stills on the GPU
at the requested display headroom, including HDR baselines and either headroom direction.
`decode_alternate` selects the alternate endpoint. This is a bounded profile of `CONT-07`;
full gain-map and JPEG XL
conformance remain open in [the roadmap](FULL_JPEG_XL_ROADMAP.md).

## Request and color contract

The caller supplies an ordinary `GpuOutputRequest::color` with explicit enumerated output color
or a supported ICC target profile.
Call `decoder.decode_gain_map(encoded, request, rendering, GainMapDecodeLimits::default()).await`
to receive a validated `GpuFrameLease<GpuImageFrame>`. `GainMapRendering` contains:

- `rendition`: `GainMapRendition::Alternate` or `DisplayHeadroom(f64)`, in nonnegative, finite stops
  (`log2(nominal display peak / reference white)`). Endpoints clamp to the base or alternate.
- `reference_white`: a positive finite `DisplayIntensity` defining the luminance of one in the
  gain equation, including its offsets. The default is 203 cd/m²; callers can specify another white.

`decode_alternate(encoded, request, limits)` uses the default rendering: alternate endpoint and
203-nit reference white. F32 linear output preserves extended light; integer output follows the
existing rounding/clipping contract. Ordinary `open` and `stream` select the baseline independently
of gain-map metadata.

This entry point accepts a complete container with exactly one underlying `jhgm` box, plain or
`brob`, with either ordering of base/alternate headroom. Each used codestream must have one complete
still presentation. The primary and auxiliary streams independently use the existing GPU
Modular/VarDCT, original/XYB and frame execution paths; this entry point adds no CPU image decoder.
Embedded previews are excluded from the main-image execution plan.

The gain application space is selected by `use_base_color_space`: the baseline's enumerated
primaries, or the bundle's enumerated alternate primaries. The baseline is converted to linear
light in that space, with unassociated alpha. Auxiliary samples remain in their original component
domain: the gain image's presentation transfer is not applied to multiplier data. Gray maps
repeat their one component for RGB; RGB maps retain all three components.

Resampling aligns image edges. For baseline coordinate `x`, auxiliary coordinate is
`(x + 0.5) * auxiliary_width / baseline_width - 0.5`, clamped to the auxiliary extent, with the
corresponding expression for `y`. Four-sample bilinear interpolation precedes clamping to `[0,1]`.
Equal extents use a direct read. For each channel, using the exact metadata fractions:

```text
fraction = clamp((display_headroom - base_headroom) / (alternate_headroom - base_headroom), 0, 1)
weight = sign(alternate_headroom - base_headroom) * fraction
gain = sample ^ (1 / gamma)
log_gain = minimum * (1 - gain) + maximum * gain
scale = baseline_intensity_target / reference_white
result_linear = ((baseline_linear * scale + base_offset) * 2 ^ (weight * log_gain)
                 - alternate_offset) / scale
```

`Alternate` selects fraction one directly. Exact cross-products determine the rational headroom
ordering; a fused F64 multiply-add retains the endpoint residual before lowering the weight to F32.
At fraction zero the ordinary baseline is returned exactly, including its same-encoding transfer
bypasses, with neither offset applied. The unused auxiliary image is not decoded; bundle syntax
and its codestream signature are still checked. Equal headrooms also select the baseline, matching
[libavif's degenerate-case policy](https://github.com/AOMediaCodec/libavif/blob/b994fe4601c62d6f98dbff295bd5c251940789b0/src/gainmap.c).
A positive fraction that underflows to F32 zero still applies
the offsets, preserving its distinction from an exact baseline selection.

The shared output kernel then performs requested primary/transfer conversion, explicit gamut
mapping, primary orientation, alpha association and packing. Alpha comes from the primary image;
the existing finite `2^-26` association floor preserves invisible colors. Unit linear RGB denotes
the baseline header's intensity target in nits. PQ scaling and the HLG display OOTF use that
intensity, consistently with ordinary decoder output. Reference white only changes the gain
equation's units; display headroom only chooses its weight. Neither implicitly changes the
output's linear unit or HLG OOTF intensity. The 203-nit default agrees with
[libavif's HDR linear units](https://github.com/AOMediaCodec/libavif/blob/b994fe4601c62d6f98dbff295bd5c251940789b0/src/colr.c);
it is an explicit rendering policy, not inferred from authored headroom metadata or
claimed to be the only white allowed by ISO.

ICC output first retains the gain result as un-oriented, unassociated planar F32 RGBA in the
enumerated linear application space. The shared GPU ICC presentation converts it to the target
profile, then packs orientation, alpha and the requested layout. Supported RGB/Gray and complete
ICC device outputs use the same profile-method and component-layout contracts as ordinary decode.
`icc_rendering_intent` selects the profile connection; Bradford adaptation is required. PCS Y=1
retains the baseline image's unit white, independently of the gain reference white. `Preserve`
alpha resolves against the primary image's declaration before the straight intermediate is packed.
ICC uses its profile intent for gamut behavior; the enumerated RGB gamut option is unavailable.
An exact baseline selection still returns ordinary output directly, including ICC output.

Typed unsupported profiles include animation, preview selection,
progressive output, nonidentity auxiliary orientation, auxiliary alpha/extra channels, CMYK gain
samples and ICC application spaces. A baseline ICC can use the ordinary GPU CMS when
the map explicitly selects an enumerated alternate application space, but expanded ICC pixel
coverage remains open. Tone mapping is rejected until an alternate-image luminance model is
provided. Portable F32 application limits weighted log gains to `[-120,120]` and requires normal
F32 white-scaling factors in both directions; these checks precede image submission. Complete numeric-range and
combined frame-feature conformance still need broader evidence. Incremental gain-map delivery,
automatic map generation and HDR encoding policy are not implemented.

## Metadata and ownership

`GainMapBundle::parse` borrows the auxiliary codestream and serialized color payloads. Its
`metadata` contains signed/unsigned 32-bit fractions with nonzero denominators, positive gamma
and exact checked gain ordering. Single-channel records expand to three equal entries. Writing
chooses canonical one/three-channel records without reducing fractions. Every fraction has its own
denominator; the version-zero records occupy 61 or 141 bytes. The six reserved flag bits must be zero.
Direction is derived from the ordering of base/alternate headroom, with no separate direction flag.
The minimum reader version must be zero. A newer writer with that minimum is accepted: its writer
version and opaque trailing `extensions` are retained when serializing. Version-zero writers may not
have trailing data. All metadata, including extensions, fits the bundle's 65,535-byte length field.
Field lengths, color padding, fraction validity and payload bounds are checked before use.
Raw auxiliary streams must start with the JPEG XL signature; full image validation belongs to
the decoder.

`GainMapBundle::new` takes metadata, optional serialized JPEG XL ColorEncoding, optional
JPEG XL-compressed ICC bytes and an auxiliary raw codestream. `encode` emits a bundle payload for
`MetadataBox` or the ordinary container writer. This can wrap GPU-encoded auxiliary packets;
the API does not synthesize a gain image from a pair of renditions. The compressed ICC bytes
remain exact, while bounded host metadata reconstruction also exposes the original ICC profile.
Tighter encode limits are rechecked without reconstructing the ICC twice.

`GainMapLimits` defaults to 64 MiB each for payload and auxiliary codestream, and 16 MiB each
for transformed and decoded ICC metadata. `GainMapDecodeLimits` also carries the independent
[container metadata limits](CONTAINER_METADATA.md), including Brotli expansion/window bounds.
These are logical payload bounds, separate from allocator overhead and decoder/GPU admission.

When a gain is selected, the auxiliary decode completes before the baseline decode. Their GPU buffers stay leased through
gain application. The common completion owner retains both inputs, final output and uniforms
even if the consumer is cancelled; it releases temporary resources before successful completion.
For ICC output, a second completion owner retains the gain intermediate, ICC program, conversion
scratch, final output and packing parameters through validation or cancellation. The gain inputs
retire before this stage; the gain intermediate retires when ICC completion releases it.
The returned image keeps the primary frame's metadata and frame-slot permit, while its output
buffer owns a separate byte reservation. See [GPU memory accounting](WGSL_MEMORY.md).

## Independent evidence

The [offline generator](../crates/jxl_wgpu_decode/test-data/gain_map_oracle/README.md) pins libjxl
for bundle/image interoperability, libavif for ISO fractions, and libultrahdr for gain application. The
64 streams cross four baseline modes, four auxiliary modes, Gray/RGB maps and BT.709/BT.2020
application primaries. They include 1×1, 17×9, 7×5 and 29×13 maps, all eight orientations,
independent alpha, negative gain minima, positive offsets and unequal channel gamma.

The integration tests compare 128 Keep/Apply outputs and 78,336 RGBA values against native decode
followed by independent F64 primary, resampling and gain math. Thirty-two further outputs cover
Linear/sRGB/PQ/HLG, Display-P3, planar/interleaved BGRA, U8/F32 and associated/unassociated alpha
(19,584 values). The fixed linear bound is `2e-4 * (1 + abs(reference))`, with alpha `2e-7`.
Transfer tests propagate this bound through independent interval arithmetic, adding the shared
`5e-5` output arithmetic allowance and one integer code. Retained images are reread exactly after
later submissions; final budgets return to zero. Unit tests cover portable WGSL/176-byte uniform
layout, completion, consumer cancellation and rejection at both output/uniform byte limits.

Eighty further outputs (48,960 values) cross both headroom directions, endpoint clamping and
intermediate display headroom with all four auxiliary coding modes and Gray/RGB maps. Their
linear/alpha bounds remain the same. Forty-eight unchanged native HDR streams add 384 outputs
(774,656 values): PQ/HLG sources, four intensities, original/XYB, Modular/VarDCT, Gray/RGB, widths
17 and 257, and reference whites 100/203/500 nits. Both headroom directions have three applications
per stream into Linear/PQ/HLG and an exact ordinary-baseline comparison. All 464 headroom
selections also run libavif's weight selection followed by the native weighted gain primitive
on independently supplied working pixels. These are 464 native invocations, not new stored fixtures.

HDR comparison propagates the existing codec uncertainty through both inputs. Original Modular
components use `1e-5 * (1 + abs(reference))`; XYB/VarDCT use `1/1024` in the same normalized bound.
Baseline bounds pass through the source EOTF/OOTF where needed. Auxiliary texel bounds pass through
bilinear interpolation, clamping, inverse gamma and signed exponentiation; four product corners
handle negative baseline-plus-offset values. The gain arithmetic allowance remains `2e-4`, then
the existing output interval and `5e-5` packing allowance apply. Eight separate original-gain GPU
decodes (4,528 values) verify those codec bounds. This accounts for near-zero gamma amplification
without increasing the fixed bounds used by the earlier image and native-formula tests.
Equal headrooms, unused truncated auxiliary images, tiny nonzero weights, rejected white scales
and exact rational endpoints have focused tests.

ICC output adds 160 declared source/rendition pairings: the 64 gain-map streams and both headroom
directions for all 48 HDR stills. Thirteen target profiles cover matrix/TRC, legacy XYZ/Lab LUTs,
v2 black-point preparation, identity MPE and 1/2/3/4/5/15 device components. Four intents and
planar/interleaved U8/F32 output produce 2,560 images and 4,688,736 component comparisons, including
alpha, orientation and reordered device components. These pairings are not a Cartesian product
of every source and profile. Independent F64 gain/Bradford equations feed the shared C++ ICC
interval evaluator and Little CMS 2.19. Existing codec, gain, PCS and profile arithmetic bounds
are propagated without widening them. Six exact-baseline ICC comparisons skip a truncated unused
map; separate checks cover primary alpha preservation, pre-image rejection of unsupported
adaptation, late ICC byte admission, cancellation and immutable retained output.

Native gain application is separately checked at `3e-6 * (1 + abs(reference))`. Its intermediate
working pixels are saved because libultrahdr's primary matrix uses six-decimal coefficients;
the GPU reference instead derives its matrix independently from chromaticities in F64. Native
baseline decode stays in linear BT.709 before this explicit primary conversion: a native CMS
connection to different primaries chooses a different below-black extension from this crate's
unbounded output contract. These distinctions preserve the predeclared tolerances. There is no
single native end-to-end `jhgm` renderer used as an oracle.

Native helpers also read and rewrite 64 Rust-serialized bundles and 128 ISO metadata records
(both headroom orderings) with byte/fraction equality. Six compatible-writer reads and 11 malformed
record rejections agree with libavif; Rust additionally preserves the compatible extensions.
Metadata-only tests cover both channel layouts, truncation at every byte, unsupported minimum
versions/reserved flags, compatible writer versions, exact ordering beyond F32 precision,
bounded ICC reconstruction, payload limits and borrowed codestream identity. Duplicate/missing
boxes and unsupported rendering profiles fail before GPU allocation.

The initial implementation used libultrahdr's obsolete draft grammar. Its common-denominator and
backward-direction flags were not adopted into the published ISO format; libavif records the
[removal of the direction flag](https://github.com/AOMediaCodec/libavif/commit/4654e63067219d6018caf9f0cbaf6075a0d06052).
Those fields have been removed from the Rust API and wire format. The 21 affected fixture metadata
payloads have been regenerated using current libavif; all 128 primary/auxiliary codestreams,
256 native pixel planes and the manifest remain byte-identical. The other 43 metadata records
already used separate denominators and remain unchanged. Subsequent headroom/HDR rendering tests
reuse these and the existing HDR fixtures without changing any compressed images or pixel references.

The [published Part 2 third-edition summary](https://cdn.standards.iteh.ai/samples/iso/iso-iec-18181-2-2026/8fe37de68af84f79a5df779b89a66a83/iso-iec-18181-2-2026.pdf)
lists the HDR Gain Map box in clause 9.11. The sample contains the contents/revision summary, not
the complete normative grammar. Concrete wire interoperability here uses the pinned
[libjxl gain-map implementation](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/extras/gain_map.cc)
and current [libavif ISO writer](https://github.com/AOMediaCodec/libavif/blob/b994fe4601c62d6f98dbff295bd5c251940789b0/src/write.c)
and [reader](https://github.com/AOMediaCodec/libavif/blob/b994fe4601c62d6f98dbff295bd5c251940789b0/src/read.c).
Official conformance material and the complete normative text remain required to close `CONT-07`.
