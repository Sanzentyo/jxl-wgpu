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
cargo run -p jxl_gpu_harness -- codec image.jxl --workload concurrent-burst --burst-size 8 --iterations 10
cargo run -p jxl_gpu_harness -- codec image.jxl --workload concurrent --concurrency 4 --iterations 20
cargo run -p jxl_gpu_harness -- codec animation.jxl --workload animation --output-target display-texture
cargo run -p jxl_gpu_harness -- codec image.jxl --output-target cpu-readback
cargo run -p jxl_gpu_harness -- codec fixtures/gpu_gray8_lossless.jxl --format u8 \
  --output-target cpu-readback --workload warm-sequential --iterations 20 \
  --cpu-baseline-djxl /opt/homebrew/bin/djxl --cpu-baseline-warmup 3 \
  --cpu-baseline-iterations 20 --cpu-baseline-timeout-ms 30000
cargo run -p jxl_gpu_harness -- codec input.gray --operation encode --format u8 --extent 256x256
cargo run -p jxl_gpu_harness -- codec input.gray --operation round-trip --format u8 --extent 256x256 --output-target cpu-readback
cargo run -p jxl_gpu_harness -- conformance --action inventory
cargo run -p jxl_gpu_harness -- conformance --action gpu-round-trip
cargo run -p jxl_gpu_harness -- conformance --action external-fixtures \
  --case tiny-gray8-2x2 --fixture-dir /tmp/jxl-reference-fixtures --apply
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

The generic format selector covers all VPI numeric layouts (`u8`, `s8`, `u16`, `u32`, `s32`,
`s16`, `2s16`, `f32`, `f64`, `2f32`), luma 8/16, planar 4:2:0/4:2:2/4:4:4,
NV12/NV21/NV16/NV61/NV24/NV42, P010/P012/P016, YUYV/UYVY, and interleaved or planar
RGB/BGR/RGBA/BGRA. The decoded-image output target is independent of scheduling: a run can retain
GPU pitch-linear output, enqueue same-queue display conversion, or request explicit host readback.
The encode command always returns standard container bytes to the host; decoded-image output target
telemetry does not describe those bytes. `codec-corpus.toml` labels small, odd-extent, and large
inputs in the report.

The `f64` selector explicitly requests `NativeOrExactF32Widening`: native shader F64 is used when
the backend enabled `SHADER_F64`; otherwise output is the documented exact widening of normalized
F32, not a full-precision F64 division. Library callers that cannot accept that compatibility path
must use `F64OutputPolicy::NativeRequired`, which returns a typed error instead of downgrading.

## Workload and telemetry semantics

The workload names describe what this implementation actually schedules:

- `single-latency` measures one operation after the requested warmup; `iterations` does not multiply
  the measured sample.
- `warm-sequential` repeats an operation on one host thread while reusing the backend objects.
- `concurrent-burst` releases `burst-size` independent host threads at a barrier for each iteration.
  Every worker still makes its own codec submission and completion wait.
- `concurrent` starts `concurrency` persistent host workers. Each worker performs `iterations`
  operations sequentially; workers overlap one another without a barrier between operations.
- `animation` opens and advances an actual animated codestream session. A still input is rejected,
  and the current stock Gray8 profile reports animation as unsupported. Repeating a still with
  `warm-sequential` is only a repeated-still proxy, not an animation result.

Neither host fan-out workload is GPU batching. The current runner does not encode several images or
frames into one command buffer. Reports therefore set `coalesced_gpu_batching` to `false` and expose
the measured-operation counts `codec_submissions`, `codec_completion_waits`,
`display_submissions`, `display_completion_waits`, `readback_submissions`, and
`readback_completion_waits`. Warmup activity is excluded from these counters.

`gpu_output_logical_bytes` sums the addressable decoded GPU layouts. For `cpu_readback`,
`readback_logical_bytes` and `readback_staging_bytes` come from the aggregate readback plan; staging
bytes include four-byte copy padding. `readback_mode` is currently `staged_copy`. A
`display_texture` run enqueues conversion on the same queue but does not wait for display conversion
completion, so it reports display submissions and zero display waits rather than pretending to be
end-to-end presentation latency. `gpu_resident` performs no image readback. `output_bytes` is the
encoded container size for encode and the decoded logical-image total for decode/round-trip.
Per-operation latency starts after a burst/worker start barrier; workload wall time includes thread
creation, synchronization, all operations, and joins. Throughput is derived from that wall time.

## Opt-in external CPU comparator

`--cpu-baseline-djxl PATH` adds a development-only libjxl `djxl` decode record to each codec case.
It never becomes a production fallback and adds no CPU codec dependency to the library crates. The
byte-comparable contract is deliberately narrow: `decode --format u8 --output-target cpu-readback`
on a still Gray8 input. Other operation, format, target, and animation combinations report a typed
`unsupported` comparator without starting `djxl`; a missing executable is a typed `skipped` result.

The comparator first records one cold decode whose interval includes process creation, decode,
temporary PGM file I/O, PGM parsing, and BLAKE3 hashing. It then runs the requested warmup and emits
`warm_repetition` p50/p95, wall throughput, input/output bytes, extent, and hash. Warmup and
iteration overrides default to the GPU workload values. `concurrent-burst` uses the same start
barrier and `concurrent` uses the same persistent host-worker shape, but every CPU operation still
launches a fresh external process. JSON therefore fixes `process_model` to
`external_process_per_operation`; “warm” does not mean a persistent libjxl decoder instance.

Each process has an independently configurable timeout. Captured stdout/stderr diagnostics are
drained to avoid pipe deadlock and bounded to 16 KiB in the report. Version provenance comes from
the selected executable's `--version` output. `verified=true` requires both the per-operation byte
count and pixel hash to match GPU decode + CPU readback. Comparator status is reported separately
and does not silently change the production path or assert that either implementation is faster.

## Multi-resolution conformance corpus

`conformance-corpus.toml` is a typed inventory of deterministic Gray, RGB, and RGBA inputs. It
includes tiny, odd, square, portrait, landscape, panorama, tall, 255/256/257 boundary, HD, FHD,
and UHD 4K extents. Each case records sample depth, alpha pattern, active bytes, row alignment,
extra padding, and padding byte. Generation is row-lazy: validation, BLAKE3 hashing, raw output,
and PGM/PPM/PAM output never allocate a complete large image.

`--action inventory` emits both a padded-storage `input_hash` and an active-pixel `pixel_hash`.
`--action gpu-round-trip` executes only cases labelled `stock_gpu_round_trip`; currently that is
the honest Gray U8 2..=256 profile. Its source is uploaded with the manifest's actual padded row
stride before GPU encoding. RGB, RGBA, higher bit depth, 257+, HD, FHD, and 4K cases remain visible
as `future_gpu_profile` and are not counted as successful codec execution.

`--action external-fixtures` is development-only and a dry run unless `--apply` is present. With
`--apply`, it streams a PGM/PPM source to the chosen `cjxl`, decodes the standard JXL through the
chosen `djxl`, and requires exact extent, channel count, sample maximum, and canonical pixel hash.
No external executable or CPU codec dependency enters a production crate. Existing files are
refused unless `--force` is explicit. See
[`docs/CONFORMANCE_CORPUS.md`](../../docs/CONFORMANCE_CORPUS.md) for the format and support inventory
contract.

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

Capture, replay, benchmark, and codec commands can write the versioned JSON report validated by
`schemas/run.schema.json`. The conformance command emits its own `schema_version = 1` inventory
report so future-profile metadata is not confused with executable codec-case success.
