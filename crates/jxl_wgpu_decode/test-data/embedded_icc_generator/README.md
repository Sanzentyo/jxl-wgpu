# Embedded ICC numeric conformance

`main.cpp` uses libjxl **0.12.0** and Little CMS **2.19**. From the repository root:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu_decode/test-data/embedded_icc_generator/main.cpp \
  -o /tmp/jxl-embedded-icc-generator \
  $(pkg-config --cflags --libs libjxl libjxl_cms lcms2)
/tmp/jxl-embedded-icc-generator crates/jxl_wgpu/test-data/icc /tmp/embedded-icc
diff -rq crates/jxl_wgpu_decode/test-data/embedded_icc /tmp/embedded-icc
```

The 12 files contain eight 17×9 JPEG XL containers, two exact ICC profiles and two interleaved
binary32 input references encoded as little-endian byte hex. RGB uses the existing GPU ICC
corpus's `gamma_v4.icc` with different channel exponents (1.75, 2.1875, 2.5). Gray uses that
corpus's sampled red curve in a native D50 Gray profile. Profile dates and IDs are deterministic.
Both profiles retain `want_icc`; no conversion to enumerated JPEG XL color metadata is permitted.

Each profile covers Modular/VarDCT × original/XYB. Color and independent alpha use binary32.
Input values are exact dyadic fractions. The generator verifies default native original-color
Modular decoding against every input bit and checks both native original/data ICC byte arrays.
These are references for exact original samples and independent alpha, not XYB color references.

`tests/embedded_icc` verifies all alpha planes through both the common engine and the standalone
codec engines, and original Modular RGB/Gray planes through both applicable entry points. It
uses complete containers and 43-byte transport fragments with 256-byte entropy windows. Further
cases replace only the color declaration in established 17/31-bit RGB fixtures, their independently
declared alpha depths, and 5/16/24/32-bit floating Gray fixtures. The replacement unwraps the
container before copying native ICC bits, compares all remaining header metadata, and verifies
that every physical-frame entropy byte is unchanged. Integer storage bytes and widened IEEE-754
words, including the existing signed-zero/subnormal/nonfinite cases, must match exactly.

Native reference limits matter for subsequent color integration. With the default CMS enabled,
libjxl 0.12.0 rejects an explicit output request for these same ICC profiles on original-color
images; on XYB images it accepts the request but then fails decoding. A linear output request on
original-color images changes samples while its data-profile query still reports the original ICC.
Do not treat successful profile requests or profile labels alone as a pixel conversion oracle.

Current decoder color conversion/composition still rejects embedded ICC. Unfiltered original
Modular numeric samples and independent extra channels in the supported single-frame paths do not
require a matrix/TRC or LUT interpretation.
Codec reconstruction, restoration and LF dependencies keep their own configuration; color
conversion is resolved only when constructing a color output. This is a partial capability within
the full JPEG XL goal.
