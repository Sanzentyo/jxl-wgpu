# jxl_gpu_bitstream

Bounded JPEG XL transport and codestream inventory for GPU codec front ends.

`parse` validates transport framing for raw codestreams, `jxlc` containers, and ordered or indexed
`jxlp` fragment sequences. Raw and single-`jxlc` codestreams remain borrowed; only fragmented
streams are joined.
After transport validation, `ParsedJxl::codestream_inventory` extracts the standard image header,
animation timing, frame headers, TOC sizes, and byte/bit ranges for every physical frame section.
It never decodes image samples or frame entropy.

The image-header grammar comes from the lightweight `jxl-image` crate. Frame-header and
non-permuted TOC grammar is parsed locally with explicit limits. Embedded ICC payloads and
entropy-coded TOC permutations are intentionally reported as unsupported grammar: locating the
next frame through either feature would require entropy decoding, which is outside this production
boundary. The returned section ranges are relative to the contiguous standard codestream, so the
same inventory applies to raw, `jxlc`, and reconstructed `jxlp` input.
