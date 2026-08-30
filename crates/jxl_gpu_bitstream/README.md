# jxl_gpu_bitstream

Bounded JPEG XL transport and codestream inventory for GPU codec front ends.

`parse` validates transport framing for raw codestreams, `jxlc` containers, and ordered or indexed
`jxlp` fragment sequences. Raw and single-`jxlc` codestreams remain borrowed; only fragmented
streams are joined.
After transport validation, `ParsedJxl::codestream_inventory` extracts the standard image header,
animation timing, complete color and extra-channel blending contracts, frame headers, TOC sizes,
and byte/bit ranges for every physical frame section. It never decodes image samples or
frame-section entropy.

The image-header grammar comes from the lightweight `jxl-image` crate. Frame-header and TOC-size
grammar is parsed locally with explicit limits. Entropy-coded TOC permutations use the published
`jxl-coding` metadata decoder, producing both physical bitstream indices and logical TOC indices.
Embedded ICC streams are reconstructed with bounded `jxl-color` primitives and retained alongside
their exact compressed bit range. Neither path decodes Modular, VarDCT, or pixel data. Returned
section ranges are relative to the contiguous standard codestream, so the same inventory applies
to raw, `jxlc`, and reconstructed `jxlp` input.
