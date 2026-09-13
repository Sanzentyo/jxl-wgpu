# ICC conversion after YCbCr reconstruction

`main.cpp` uses Little CMS **2.19** and the shared independent f64 ICC/CIE reference
equations. From the repository root:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu_decode/test-data/embedded_icc_ycbcr_generator/main.cpp \
  -o /tmp/jxl-icc-ycbcr-generator \
  $(pkg-config --cflags --libs lcms2)
/tmp/jxl-icc-ycbcr-generator crates/jxl_wgpu_decode/test-data /tmp/icc-ycbcr
diff -rq -x README.md crates/jxl_wgpu_decode/test-data/embedded_icc_ycbcr /tmp/icc-ycbcr
```

The generator reads existing device references for Modular sampling_123, gray,
associated, resampling_8, and VarDCT vardct_ycbcr_bt709_srgb_still. Each source
keeps its existing reconstruction reference and tolerance. The resampled fixture uses
the established independently expanded 4:4:4 reference, including restoration.
No source or reference is obtained from this crate's GPU output.

The RGB and Gray profiles are the exact files in embedded_icc. For each source,
the 60 output files cover linear BT.709, sRGB, and the other corpus ICC profile.
RGB-to-Gray and Gray-to-RGB conversions therefore exercise actual plane-count changes.
Every file contains one hexadecimal binary32 word per line, interleaved with alpha.

The scalar references evaluate independent matrix/TRC equations. The native references
evaluate Little CMS with relative intent, NOOPTIMIZE | NOCACHE, and no black-point
compensation. Linear BT.709 uses the native XYZ double interface followed by independent
CIE/Bradford geometry; sRGB additionally applies the OETF. Little CMS's floating
interface extrapolates out-of-range device inputs, so the generator explicitly clamps
those inputs to the ICC primitive's specified [0,1] domain. The scalar forward curves
apply the same domain contract internally. Every sample is checked; the JSON output
reports the number of input values needing this clamp.

The lower and upper references propagate the existing codec error through independent
color equations and their established F32 arithmetic bounds. Modular uses absolute
2e-6; VarDCT uses (1 + abs(reference))/1024. Corners of each source error box are rounded
outward before conversion. Monotone source/target curves are checked explicitly.
Each matrix row and each arithmetic-bound endpoint has its extrema at box corners;
the inverse target curve then propagates that interval without a fixed output-code
error allowance. sRGB packing retains the shared output kernel's normalized 2e-6
transfer bound. Alpha retains its existing source error bound.

The native result must lie inside the separately derived native precision interval,
which accounts for Little CMS's 16-bit sampled forward tables and 4096-entry reverse
approximation. This native interval is never used to relax the GPU assertion.
Near black, the Gray-to-RGB native result can differ from the scalar center by about
0.000546; its GPU interval is independently narrower.

The converted YCbCr integration test first checks the actual reconstructed device
values against the source bounds, then checks converted values against these intervals
for planar/interleaved output and whole/fragmented input. It also verifies that
conversion keeps the actual reconstructed alpha words, and that completed output
leases remain unchanged after their session is dropped.
