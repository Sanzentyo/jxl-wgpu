# jxl_gpu_bitstream

Bounded JPEG XL transport and codestream inventory for GPU codec front ends.

`parse` validates transport framing for raw codestreams, `jxlc` containers, and ordered or indexed
`jxlp` fragment sequences. Raw and single-`jxlc` codestreams remain borrowed; only fragmented
streams are joined.
After transport validation, `ParsedJxl::codestream_inventory` extracts the standard image header,
animation timing, frame headers, TOC sizes, and byte/bit ranges for every physical frame section.
It never decodes image samples or frame-section entropy.

The image-header grammar comes from the lightweight `jxl-image` crate. Frame-header and TOC-size
grammar is parsed locally with explicit limits. Entropy-coded TOC permutations use the published
`jxl-coding` metadata decoder, producing both physical bitstream indices and logical TOC indices;
this does not decode Modular, VarDCT, or pixel data. Embedded ICC payloads remain typed unsupported
grammar. Returned section ranges are relative to the contiguous standard codestream, so the same
inventory applies to raw, `jxlc`, and reconstructed `jxlp` input.
