# ICC matrix/TRC references

`main.cpp` uses Little CMS **2.19**. It serializes ten generated RGB/gray profiles, fixes their
timestamp and clears their optional profile ID, reopens the exact bytes, and converts all 100
ordered pairs with relative colorimetric intent, `NOOPTIMIZE | NOCACHE`, and no black-point
compensation. Float native output is clipped to the declared unit output range.

`scalar.hpp` independently evaluates the ICC.1:2022 matrix/TRC equations in f64, using Little CMS
only to read the profile's exact colorant/curve metadata. It uses pivoted elimination and
analytical roots or exhaustive sampled-segment search. Production Rust/WGSL code is not called.
See [the execution contract](../../../../docs/ICC_MATRIX_TRC.md) for scope, precision intervals
and the two documented Little CMS boundary differences.

The deterministic corpus lives in `../icc`:

- `manifest.json`: dimensions (37×17), ordered profile names and channel counts.
- `<name>.icc`: exact ICC profile bytes; `gamma_v2` is v2.4, the others v4.3.
- `<name>_input.f32le`: interleaved input samples in row order.
- `<source>_to_<target>.reference`: one 28-byte little-endian record per interleaved output
  component: six F32 values (`native`, `exact`, `lower`, `upper`, `native_lower`, `native_upper`),
  then one u32 native-semantics mask. Bit 0 marks the dropped offset at a zero power base;
  bit 1 marks negative input to the offset inverse. Zero means native semantics agree.

The original native values remain in every record, including the marked boundary differences.
All components are always asserted against the independent scalar interval. Neither oracle nor
Little CMS is a production dependency. The unit-domain contract is not an embedded-ICC decoder
admission claim.

From the workspace root, build with a C++17 compiler and the pinned `lcms2` development package:

```sh
mkdir -p .git/icc-regenerate
c++ -std=c++17 -Wall -Wextra -Werror \
  crates/jxl_wgpu/test-data/icc_generator/main.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/generator
.git/icc-regenerate/generator .git/icc-regenerate/corpus
diff -rq crates/jxl_wgpu/test-data/icc .git/icc-regenerate/corpus
cargo test -p jxl_wgpu --test icc -- --test-threads=1
```

Generation is offline. `cargo test` consumes the checked-in bytes and does not spawn an oracle.
