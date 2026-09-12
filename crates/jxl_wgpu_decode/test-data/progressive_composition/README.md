# Standalone headers for progressive composition oracles

Generate with `cargo run -p jxl_wgpu_decode --example regenerate_progressive_composition` and
libjxl 0.12.0. The example compiles `../generate_frame_composition.c`, requests its
`--progressive-layers` mode, and retains only the image/frame headers for each of the nine layers
in the existing `composition_vardct`, `composition_vardct_gray`, `composition_vardct_dc` and
`composition_associated_vardct` fixtures.
Each file contains an `image HEX` prefix followed by `frame BIT_LENGTH HEX` records. These are
metadata fragments, not complete codestreams; LF-dependent layers include their LF headers.

`tools/jxl_test_support/src/fixtures/progressive_layers.rs` copies the original fixture entropy unchanged and rebuilds
the TOC under these standalone headers. It checks the original color/restoration metadata and
all frame fields affecting entropy interpretation. Only presentation metadata, origins, blend
references and container-relative positions may differ. Noise/patch-dependent inputs are excluded.

libjxl's `FrameDecoder::Flush` rejects unfinalized frames with crops or blending. The native oracle
therefore flushes these standalone layers and independently composes them in interleaved F64 sRGB,
using committed reference slots, then converts to linear output and applies orientation. Every
scalar final is also checked against native coalesced decoding of the original animation. This
validates composed DC/AC updates without weakening the reference decoder or re-encoding entropy.
For composed LF1 updates, Rust `jxl` 0.6.0 flushes a standalone prefix ending after LF1. The same
independent scalar composition applies the terminal layer's committed references. LF dependencies
are selected by exact physical IDs, allowing a hidden reference layer between LF1 and its visible
consumer. A runtime fixture exercises this order by moving original physical frame 8 between
frames 6 and 7; every header and entropy byte is unchanged. LF2 has no native per-level pixel-oracle
claim. The existing 27 metadata fragments suffice for both orders and are not duplicated.
Production does not link these offline tools or perform CPU pixel/entropy decoding.

`associated_vardct_layer0..8.headers` add nine fragments (628 bytes) for the existing associated-alpha
composition fixture. Standalone headers replace both color and extra-channel blend metadata; all
entropy interpretation fields remain checked. Native logical-prefix flushes provide DC/AC layers,
then an independent F64 compositor applies the original color/alpha reference selectors, clamps,
alpha-over and alpha-weighted-add rules. Every scalar final is checked against native coalesced
output. GPU tests cover both orientations and whole/40-byte fragmented delivery. The complete set
now contains 36 fragments (2,536 bytes); the preceding 27 fragments are unchanged.
