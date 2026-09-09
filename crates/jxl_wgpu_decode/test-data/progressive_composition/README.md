# Standalone headers for progressive composition oracles

Generate with `cargo run -p jxl_wgpu_decode --example regenerate_progressive_composition` and
libjxl 0.12.0. The example compiles `../generate_frame_composition.c`, requests its
`--progressive-layers` mode, and retains only the image/frame headers for each of the nine layers
in the existing `composition_vardct`, `composition_vardct_gray` and `composition_vardct_dc` fixtures.
Each file contains an `image HEX` prefix followed by `frame BIT_LENGTH HEX` records. These are
metadata fragments, not complete codestreams; LF-dependent layers include their LF headers.

`tests/common/progressive_layers.rs` copies the original fixture entropy unchanged and rebuilds
the TOC under these standalone headers. It checks the original color/restoration metadata and
all frame fields affecting entropy interpretation. Only presentation metadata, origins, blend
references and container-relative positions may differ. Noise/patch-dependent inputs are excluded.

libjxl's `FrameDecoder::Flush` rejects unfinalized frames with crops or blending. The native oracle
therefore flushes these standalone layers and independently composes them in interleaved F64 sRGB,
using committed reference slots, then converts to linear output and applies orientation. Every
scalar final is also checked against native coalesced decoding of the original animation. This
validates composed DC/AC updates without weakening the reference decoder or re-encoding entropy.
Production does not link these offline tools or perform CPU pixel/entropy decoding.
