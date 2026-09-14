# ICC XYB alpha and blend references

The 48 two-frame 17×9 streams cover RGB/Gray, Modular/VarDCT, straight/associated F32 alpha,
and Replace/Add/Blend/MultiplyAdd/Multiply. The first frame saves reference slot 1 after color
transformation. The second frame uses the selected color operation and a separate Replace
extra-channel operation, whose full-frame source is implicitly slot 0 (transparent here).
Eight additional `_alpha_ref1` Blend streams explicitly blend the alpha from saved slot 1.
The native decoder and Rust test both verify these actual extra-channel selectors.
Color Blend computes the selected alpha's combined coverage, matching
libjxl's `PerformBlending` contract even when that extra channel's own operation is Replace.

Each stream has native and independent scalar device values for both physical layers and their
composition, plus lower/upper acceptance bounds: **48 codestreams and 576 F32LE files**.
See [the generator](../../embedded_icc_xyb_generator/README.md) for reproduction, independent
precision intervals and native CMS limitations. No reference contains GPU-produced pixels.

GPU tests preserve the original alpha association, check both frames in planar/interleaved
layouts, compare whole input with 43-byte fragments and 256-byte entropy windows, and reread
held buffers after session destruction. Alpha, including zero and one, must match exact words.
Eight straight-alpha Add/MultiplyAdd sequences retain composed device values above one.
This corpus does not claim arbitrary crop/reference selectors, output alpha-policy conversion,
transformed targets after composition, or full ICC conformance.
