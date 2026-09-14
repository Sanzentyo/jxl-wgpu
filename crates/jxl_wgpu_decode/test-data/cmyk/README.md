# CMYK image references

`layers.jxl`, `layers.icc`, `layers.json`, and the uncompressed contents of `layers.npy.gz`
are unchanged files from the official [cmyk_layers conformance case](https://github.com/libjxl/conformance/tree/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases/cmyk_layers).
The case is listed as CC0 in the upstream test-case directory; its repository license is
also retained in `LICENSE`.

The 512 × 512 image has three complemented CMY components followed by Black and Alpha.
The NumPy shape is `(1, 512, 512, 5)`, little-endian F32 in that order. Every channel uses
the unchanged upstream RMSE and absolute peak limit `0.000976562`. Whole and fragmented
input with a 256-byte GPU window must additionally agree bit-for-bit.

SHA-256 digests, verified by the test before any comparison:

- Input: `d732c8836bf1abeadf310d2e07387a32813ed4690d32650c1c25e541b80eed4a`.
- Original/reference ICC: `4855b8fabb96bdc6495d45d089bb8c8efb1ae18389e0dc9e75a5f701a9c0b662`.
- Uncompressed NumPy: `a01913d4e4b1a89bd96e5de82a5dfb9925c7827ee6380ad60c0b1c4becb53880`.

ICC and NumPy objects are available from
`https://storage.googleapis.com/jxl-conformance/objects/<sha256>`.
The gzip file uses `gzip -n -c`, with no timestamp or filename metadata.

`generated/` contains 18 three-frame streams, their native component images, 144
independent/native ICC reference files and a manifest. See the
[generation and precision contract](../cmyk_generator/README.md).
