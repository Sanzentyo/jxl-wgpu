# Native linear RGB gamut reference

`main.cpp` calls `jxl::GamutMapScalar` from unmodified libjxl 0.12.0 commit
`a7a9c787341cf703dede03c2009fa460cae5e5df`. It does not copy the primitive's equations into the
generator. The primary source is [libjxl tone_mapping.h](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/cms/tone_mapping.h).

With a checkout of that exact source at `$LIBJXL_SOURCE`:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -DNDEBUG -ffp-contract=off \
  -isystem "$LIBJXL_SOURCE" main.cpp -o /tmp/jxl-gamut-reference
/tmp/jxl-gamut-reference /tmp/jxl-gamut-native.txt
cmp /tmp/jxl-gamut-native.txt ../gamut_mapping/native.txt
```

Each of the 5,000 rows contains eight decimal binary32 values: primary-set index, saturation
preference, input RGB, native RGB. The sets are BT.709 (0), BT.2020 (1) and Display-P3 (2), using
their D65 matrix Y rows. Preferences are 0, 0.1, 0.5, 0.9 and 1. Inputs combine the Cartesian grid
`{-0.25,0,0.125,0.5,1,1.25,4}` with each component's strict neighbors of zero and one.
Negative-luminance samples are excluded from the native corpus: 333 rows per preference remain
for BT.709/BT.2020 and 334 for P3. Signed components and above-white inputs remain included.

The independent F64 oracle constructs line/cube intersections. Native and actual GPU components
each have a predeclared `4e-6` linear allowance. Three workgroup widths compare 45,000 GPU
components, retaining exact in-cube values, alpha and output guards. Negative-luminance black,
overflow-resistant evaluation through ±1e30, exact lower faces and subnormal sign handling are
explicit implementation policies tested separately. This recipe does not claim equivalence to
libjxl's automatic HDR rendering or a perceptual ICC target gamut. See the complete
[policy and image evidence](../../../../docs/GAMUT_MAPPING.md).
