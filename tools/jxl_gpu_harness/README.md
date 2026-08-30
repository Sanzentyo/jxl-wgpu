# jxl_gpu_harness

Deterministic capture/replay, correctness, benchmark, codec-workload, and autotuning tool for
`jxl_wgpu`.

```console
cargo run -p jxl_gpu_harness -- adapters
cargo run -p jxl_gpu_harness -- verify --backend reference
cargo run -p jxl_gpu_harness -- verify --backend wgpu
cargo run -p jxl_gpu_harness -- bench --backend wgpu --warmup 10 --iterations 100
cargo run -p jxl_gpu_harness -- codec --corpus tools/jxl_gpu_harness/codec-corpus.toml
cargo run -p jxl_gpu_harness -- codec image.jxl --format nv12 --output-target gpu-resident
cargo run -p jxl_gpu_harness -- codec image.jxl --workload warm-sequential --warmup 3 --iterations 20
cargo run -p jxl_gpu_harness -- codec image.jxl --workload batch --batch-size 8 --iterations 10
cargo run -p jxl_gpu_harness -- codec image.jxl --workload concurrent --concurrency 4 --iterations 20
cargo run -p jxl_gpu_harness -- codec animation.jxl --workload animation --output-target display-texture
cargo run -p jxl_gpu_harness -- codec image.jxl --output-target cpu-readback
cargo run -p jxl_gpu_harness -- codec input.gray --operation encode --format gray8 --extent 256x256
cargo run -p jxl_gpu_harness -- codec input.gray --operation round-trip --format gray8 --extent 256x256 --output-target cpu-readback
cargo run -p jxl_gpu_harness -- capture --name gaborish_edge_17x19 --operation gaborish -o case.jxlcap
cargo run -p jxl_gpu_harness -- replay --input case.jxlcap --backend wgpu
cargo run -p jxl_gpu_harness -- tune --output tuning.json --candidates reference,wgpu
```

## GPU codec contract

The `codec` command uses `jxl_wgpu_decode` and `jxl_wgpu_encode` as GPU-required production
boundaries. It never runs the published CPU `jxl` decoder as a fallback and never uploads pixels
decoded by a CPU codec. Explicit CPU readback means copying an already GPU-decoded result into
host-visible memory; it does not mean CPU decoding.

The current executable profile accepts a 2..=256 pitch-linear non-color Gray8 source. Encoding runs
predictor/token/histogram work on the GPU and emits a standard lossless JXL container with a
hash-bound acceleration metadata box. The matching stock decoder reads entropy tokens from the
actual `jxlc` bytes, reconstructs residuals and the Gradient predictor on the GPU, and produces
generic pitch-linear output. `round-trip --output-target cpu-readback` executes both kernels and
requires an exact decoded-byte hash match. Other recognized but unimplemented profiles remain
typed `unsupported` and make the command exit unsuccessfully. CPU `jxl`, `cjxl`, and `djxl` may be
used only by a separately labelled reference/oracle validation, never as the production codec path.

The generic format selector covers non-color Gray8, luma 8/16, planar 4:2:0/4:2:2/4:4:4, NV12/NV21/NV16/NV61/
NV24/NV42, P010/P012/P016, YUYV/UYVY, and interleaved or planar RGB/BGR/RGBA/BGRA. The output target
is independent of scheduling, so cold latency, warm sequential reuse, batches, simultaneous
workers, and animation can each retain GPU pitch-linear output, enqueue same-queue display
conversion, or request explicit host readback. `codec-corpus.toml` labels small, odd-extent, and
large inputs in the versioned report.

## Synthetic capture/replay

Synthetic capture/replay remains independent from the production codec frontend. Corpus cases and
per-operation acceptance thresholds live in `corpus.toml` and `thresholds.toml`. Capture files are
portable and self-validating: metadata and payload lengths are bounded, payload bytes are checked
with BLAKE3, and decoding rejects truncation, trailing bytes, invalid enum values, checksum
mismatches, and unsupported schema versions.

Synthetic `chroma_upsample` cases use `axis=0` for horizontal and `axis=1` for vertical. Optional
`output_width` or `output_height` selects the codec-valid odd crop (`2n - 1`) instead of the full
`2n` extent. Synthetic `epf` cases use `pass=0`, `1`, or `2`; `variable_sigma=1` stores one finite
F32 sigma sample per 8x8 block.

All commands can write the versioned JSON report validated by `schemas/run.schema.json`.
