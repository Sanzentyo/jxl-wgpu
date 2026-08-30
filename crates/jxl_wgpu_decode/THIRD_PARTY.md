# Third-party implementation references

The bounded standard VarDCT frontend follows the JPEG XL field order, default block-context map,
and Modular subimage geometry from the following `jxl-oxide` crates:

- `jxl-vardct` 0.11.1 (`src/lf.rs`, `src/hf_metadata.rs`, `src/hf_pass.rs`, and
  `src/hf_coeff.rs`)
- `jxl-frame` 0.13.3 (`src/data/lf_global.rs`, `src/data/lf_group.rs`, and
  `src/data/hf_global.rs`)
- `jxl-modular` 0.11.3 (`src/lib.rs`, `src/ma.rs`, and `src/predictor.rs`)
- `jxl-render` 0.12.4 (`src/vardct/generic/mod.rs`, adaptive LF smoothing constants and
  neighborhood operation)
- `jxl-coding` 1.0.1 (entropy-descriptor and permutation grammar)

Those crates are distributed under `MIT OR Apache-2.0`. Production code uses `jxl-bitstream` and
`jxl-coding` only for bounded metadata parsing; `jxl-frame`, `jxl-modular`, and `jxl-vardct` are
dev-only scalar-oracle dependencies and are not a CPU codec fallback.
