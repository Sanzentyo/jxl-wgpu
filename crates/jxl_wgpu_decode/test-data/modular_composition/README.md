# Composed Modular pass fixtures

Regenerate with offline libjxl 0.12.0:

```sh
cargo run -p jxl_wgpu_decode --example regenerate_modular_composition
```

An optional output-directory argument allows byte-for-byte reproduction checks. The generator
compiles `../generate_frame_composition.c` with `--modular-progressive`. Each 2051×17 animation
contains nine physical layers, six presentations and two passes per layer. The RGB8 fixture uses
orientation 6; Gray16 with associated integer alpha5 uses orientation 8; floating Gray16/exponent5
with associated alpha24/exponent7 uses orientation 5. All five blend modes, hidden references,
negative/oversized/off-canvas crops and independent alpha references are exercised.

The `.jxl.hex` files preserve the native transport, including containers. Each `.headers` file
contains only the independently encoded standalone layer's image and frame headers. Test helpers
normalize containers, preserve the animation's entropy bytes and rebuild its standalone TOC.
Regeneration asserts that every resulting standalone codestream equals the native encoding.

Floating source values are exact quarters. The alpha precision leaves enough working bits for
native responsive Squeeze to expose real intermediate images; streams with no decoded global/LF
samples correctly defer publication to their first nonempty pass. Tests use native standalone
prefix flushes followed by independent F64 crop/reference composition, and check final composition
against the original native coalesced animation. No production CPU codec is involved.

Hashes, error bounds and lifecycle evidence are recorded in `docs/CONFORMANCE_CORPUS.md` at the
workspace root.
