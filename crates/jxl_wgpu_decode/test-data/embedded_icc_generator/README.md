# Embedded ICC samples and original color

`main.cpp` uses libjxl **0.12.0** and Little CMS **2.19**. From the repository root:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu_decode/test-data/embedded_icc_generator/main.cpp \
  -o /tmp/jxl-embedded-icc-generator \
  $(pkg-config --cflags --libs libjxl libjxl_cms lcms2)
/tmp/jxl-embedded-icc-generator crates/jxl_wgpu/test-data/icc /tmp/embedded-icc
diff -rq crates/jxl_wgpu_decode/test-data/embedded_icc /tmp/embedded-icc
```

The 40 files contain eight 17×9 JPEG XL containers, two exact ICC profiles, two input references
and 28 original-color references. All sample files use little-endian binary32 byte hex. The original
12 files are unchanged. RGB uses the resident ICC corpus's `gamma_v4.icc` with different channel
exponents (1.75, 2.1875, 2.5). Gray uses that corpus's sampled red curve in a native D50 Gray profile.
Profile dates and IDs are deterministic. Both profiles retain `want_icc`; the encoder does not
replace them with enumerated JPEG XL color metadata.

Each profile covers Modular/VarDCT × original/XYB. Color and independent alpha use binary32;
input values are exact dyadic fractions. Default native original-color decoding verifies both
original/data ICC byte arrays. Lossless Modular matches every input word. The `.native.f32.hex`
references also retain actual VarDCT device pixels; lossy VarDCT is not compared to encoder input.

Each original stream has `linear`, `srgb` and `other` color references. `other` requests the other
corpus profile, crossing RGB/Gray in both directions. Little CMS uses relative intent,
`NOOPTIMIZE | NOCACHE`, with no black-point compensation. For linear and sRGB output, native ICC
execution uses its XYZ double interface followed by independent CIE/Bradford geometry and the
sRGB OETF. No surrogate linear ICC profile changes the target basis.

The matching `.scalar.f32.hex` files use the shared independent f64 reference equations in
`tools/jxl_test_support/native/icc`. Little CMS only decodes exact profile tags for this oracle;
its pivoted matrix inversion and analytical/exhaustive curve inverse do not call production
Rust/WGSL. Every native component must also lie within the separately propagated native precision
interval. No native component is masked in this corpus. Alpha remains unchanged in all references.

The GPU's primary color assertion uses the independent scalar result. Native values are retained
as additional evidence. A Gray VarDCT sample near black differs by about 0.000388 between Little
CMS and the scalar result because the native sampled-curve path has lower precision. GPU error
for that conversion is below 0.0000023. The decoder's end-to-end F32 bound is 2e-4, including the
separately checked VarDCT reconstruction error; original device VarDCT output has a 2e-5 bound.
Lossless Modular device samples and all alpha words must match exactly.

`tests/embedded_icc` exercises common color decoding with complete input and 43-byte transport
fragments using 256-byte entropy windows. It covers same-profile RGB/Gray, requested linear/sRGB
and other ICC output, planar/interleaved F32, U8 channel order/padding, and alpha association before
quantization. A six-presentation Gray case replaces only the existing fixture's color declaration,
retains all frame entropy bytes and validates Add/Multiply values above one after Exif-six rotation.
Private GPU tests check exact initial program/output/scratch admission, retry, program reuse and
completion-owned cancellation.

Numeric tests retain the original common/standalone coverage for all eight alpha streams and
original Modular color samples. Metadata-only substitutions into established 17/31-bit RGB and
5/16/24/32-bit floating Gray fixtures preserve all other headers and every physical frame byte.
Integer storage and widened IEEE-754 words, including signed zero, subnormals and nonfinite values,
remain exact. Same-profile F32 color output with preserved alpha also keeps those floating words;
it does not evaluate a profile-to-itself transform or select an unsupported CMS intent.

Native profile requests alone are insufficient evidence. libjxl 0.12.0 rejects an explicit output
request for these same ICC profiles on original-color images; on XYB it accepts the request but
then fails decoding. A requested linear output can change pixels while the data-profile query
still reports the original ICC. This generator therefore decodes original device pixels first
and performs independently checked conversion afterward.

Common original ICC RGB/Gray color execution is a partial capability in the full JPEG XL goal.
Broader ICC XYB conformance, enumerated-source ICC targets, spot rendering, standalone codec
color admission, LUT/MPE/Lab/CMYK, broader intent/gamut policy and HDR/display integration remain open.
Neither native reference executable nor CPU pixel conversion is a production dependency.

YCbCr reconstruction and converted presentation use the existing device corpus plus the separate
[ICC YCbCr generator](../embedded_icc_ycbcr_generator/README.md). Its 146 source substitutions and
five independently bounded conversion cases extend the same common execution path.

XYB reconstruction and original-device reference addition have a separate
[generator and precision contract](../embedded_icc_xyb_generator/README.md). Direct linear
output and RGB/Gray requested profiles reuse these four XYB inputs without changing their bytes.
