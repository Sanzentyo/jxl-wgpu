# LF consumer geometry and signed samples

These fixtures are generated offline with libjxl 0.12.0. Production decoding stays on the GPU.
Regenerate the 81 generated files with:

```sh
cargo run -p jxl_wgpu_decode --example regenerate_lf_extra_channels -- OUTPUT --geometry
```

Each family has a Modular and a VarDCT LF root. Ordinary native entropy is reframed into a
recursive LF chain; only the dependent VarDCT LF coefficient substream is removed. The original
extra metadata, restoration and remaining entropy are retained. libjxl decodes every result.

| Family | Canvas | LF levels | Extra dimension shift | Alpha / depth precision | Association |
|---|---|---|---|---|---|
| shifted_integer | 65×33 | 2 | 1 | 16 / 20 integer bits | Associated |
| shifted_float | 65×33 | 2 | 2 | 32/8 / 32/8 float | Associated |
| shifted_thin_float | 193×65 | 1 | 3 | 32/8 / 32/8 float | Unassociated |
| signed_float | 65×33 | 3 | 0 | 16/5 / 24/7 float | Unassociated |

Float precision is total bits / exponent bits. Depth inputs in the three floating families span
−0.5 to 1.5 before any native downsampling. Dyadic values keep custom-precision lossless samples
exactly representable. Resampled float extras use 32/8 so the native encoder can represent its
averages. The `signed_float` final and LF output oracles explicitly require negative values;
final depth also exceeds one. Alpha input stays in [0, 1].

An omitted LF-consumer upsampling selector defaults to one, with `dimension_shift` still applied.
The seed generator encodes that effective scale; the frame writer converts effective factors
back to on-wire selectors for ordinary/root frames. Root extras in the shift-1/2 families use an
additional factor of two. Existing `lf_extra_channels` and `lf_conformance` files regenerate
unchanged.

Each `.crop_left` or `.crop_top` final frame is 12 columns and 8 rows smaller than the canvas,
at (−3, 5) or (5, −3). Its LF prediction reads the producer's top-left local block rectangle;
the canvas offset belongs to subsequent composition. Color and alpha use source-over from
reference one; depth uses Add. The hidden background is decoded after the LF producers.
`.crop_foreground` preserves that LF dependency and cropped entropy but places it at (0, 0)
over zero, allowing an independent scalar composition oracle to extract the foreground.

`.lfN.jxl.hex` and `.lfN.linear.f32.hex` describe independently decodable small-image producers.
Tests verify unchanged entropy/metadata, invert native linear color into XYB in F64, clip to the
consumer's local LF grid and recursively expand it before color conversion and composition.
This defines the crate's complete-LF presentation policy; it is not an ISO decoder-precision
claim. Native/GPU color reconstruction uses the existing 5e-4 bound, extras 3e-6, and native
integer packing one code. Unassociation is validated separately against the native-validated
Preserve output, avoiding amplification of reconstruction differences at tiny alpha.

Reference behavior is visible in libjxl's
[frame-header defaults](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/frame_header.cc) and
[consumer DC rectangle bounds](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_group.cc).
