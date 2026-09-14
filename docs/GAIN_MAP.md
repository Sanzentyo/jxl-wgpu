# JPEG XL HDR gain maps

`jxl_gpu_bitstream::gain_map` reads and writes version-zero `jhgm` bundles and exact
ISO 21496-1 rational parameters. `GpuDecoder::decode_alternate` reconstructs the alternate still
on the GPU. This is the initial rendering profile of `CONT-07`; full gain-map and JPEG XL
conformance remain open in [the roadmap](FULL_JPEG_XL_ROADMAP.md).

## Request and color contract

The caller supplies an ordinary `GpuOutputRequest::color` with explicit enumerated output color.
Call `decoder.decode_alternate(encoded, request, GainMapDecodeLimits::default()).await` to receive
a validated `GpuFrameLease<GpuImageFrame>`. F32 linear output preserves extended light; integer
output follows the existing rounding/clipping contract. Ordinary `open` and `stream` select the
baseline independently of gain-map metadata.

This entry point accepts a complete container with exactly one underlying `jhgm` box, plain or
`brob`, and a forward map whose base HDR headroom is zero. Each codestream must have one complete
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
gain = sample ^ (1 / gamma)
log_gain = minimum * (1 - gain) + maximum * gain
alternate_linear = (baseline_linear + base_offset) * 2 ^ log_gain - alternate_offset
```

The shared output kernel then performs requested primary/transfer conversion, explicit gamut
mapping, primary orientation, alpha association and packing. Alpha comes from the primary image;
the existing finite `2^-26` association floor preserves invisible colors. Unit linear RGB denotes
the baseline header's intensity target in nits. PQ scaling and the HLG display OOTF use that
intensity, consistently with ordinary decoder output. This API returns the exact alternate
rendition; it does not choose an intermediate display headroom.

Typed unsupported profiles include HDR baselines/backward direction, animation, preview selection,
progressive output, nonidentity auxiliary orientation, auxiliary alpha/extra channels, CMYK gain
samples, ICC application spaces and ICC output. A baseline ICC can use the ordinary GPU CMS when
the map explicitly selects an enumerated alternate application space, but expanded ICC pixel
coverage remains open. Tone mapping is rejected until an alternate-image luminance model is
provided. Portable F32 application limits log gains to `[-120,120]`; complete numeric-range and
combined frame-feature conformance still need broader evidence. Incremental gain-map delivery,
automatic map generation and HDR encoding policy are not implemented.

## Metadata and ownership

`GainMapBundle::parse` borrows the auxiliary codestream and serialized color payloads. Its
`metadata` contains signed/unsigned 32-bit fractions with nonzero denominators, positive gamma
and exact checked gain ordering. Single-channel records expand to three equal entries. Writing
chooses canonical one/three-channel and common/separate-denominator layouts without reducing
fractions. Version, reserved flags, field lengths, color padding and trailing metadata are checked.
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

The auxiliary decode completes before the baseline decode. Their GPU buffers stay leased through
gain application. The common completion owner retains both inputs, final output and uniforms
even if the consumer is cancelled; it releases temporary resources before successful completion.
The returned image keeps the primary frame's metadata and frame-slot permit, while its output
buffer owns a separate byte reservation. See [GPU memory accounting](WGSL_MEMORY.md).

## Independent evidence

The [offline generator](../crates/jxl_wgpu_decode/test-data/gain_map_oracle/README.md) pins libjxl
for wire/image interoperability and libultrahdr for ISO fractions and gain application. The
64 streams cross four baseline modes, four auxiliary modes, Gray/RGB maps and BT.709/BT.2020
application primaries. They include 1×1, 17×9, 7×5 and 29×13 maps, all eight orientations,
independent alpha, negative gain minima, positive offsets and unequal channel gamma.

The integration tests compare 128 Keep/Apply outputs and 78,336 RGBA values against native decode
followed by independent F64 primary, resampling and gain math. Thirty-two further outputs cover
Linear/sRGB/PQ/HLG, Display-P3, planar/interleaved BGRA, U8/F32 and associated/unassociated alpha
(19,584 values). The fixed linear bound is `2e-4 * (1 + abs(reference))`, with alpha `2e-7`.
Transfer tests propagate this bound through independent interval arithmetic, adding the shared
`5e-5` output arithmetic allowance and one integer code. Retained images are reread exactly after
later submissions; final budgets return to zero. Unit tests cover portable WGSL/160-byte uniform
layout, completion, consumer cancellation and rejection at both output/uniform byte limits.

Native gain application is separately checked at `3e-6 * (1 + abs(reference))`. Its intermediate
working pixels are saved because libultrahdr's primary matrix uses six-decimal coefficients;
the GPU reference instead derives its matrix independently from chromaticities in F64. Native
baseline decode stays in linear BT.709 before this explicit primary conversion: a native CMS
connection to different primaries chooses a different below-black extension from this crate's
unbounded output contract. These distinctions preserve the predeclared tolerances. There is no
single native end-to-end `jhgm` renderer used as an oracle.

Native helpers also read and rewrite 64 Rust-serialized bundles and 128 ISO metadata records
(both directions flags) with byte/fraction equality. Metadata-only tests cover all four fraction
layouts, truncation at every byte, unsupported versions/flags, exact ordering beyond F32 precision,
bounded ICC reconstruction, payload limits and borrowed codestream identity. Duplicate/missing
boxes and unsupported rendering profiles fail before GPU allocation.

The [published Part 2 third-edition summary](https://cdn.standards.iteh.ai/samples/iso/iso-iec-18181-2-2026/8fe37de68af84f79a5df779b89a66a83/iso-iec-18181-2-2026.pdf)
lists the HDR Gain Map box in clause 9.11. The sample contains the contents/revision summary, not
the complete normative grammar. Concrete wire interoperability here uses the pinned
[libjxl gain-map implementation](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/extras/gain_map.cc)
and [libultrahdr fraction implementation](https://github.com/google/libultrahdr/blob/6929c2b087e74120e6de52f361e77b06f07b1441/lib/src/gainmapmetadata.cpp).
Official conformance material and the complete normative text remain required to close `CONT-07`.
