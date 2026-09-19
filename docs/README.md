# Documentation index

Use these references for maintainer-directed implementation and verification. The
[project README](../README.md#project-phase) states the development phase and
participation policy. Consult large references by relevant heading or feature,
not as a prerequisite reading sequence for every edit.

## Entry points

| Need | Start here |
|---|---|
| Understand the scope or run a fixture | [Project README](../README.md) |
| Select local checks and record evidence | [Internal development and validation](DEVELOPMENT.md) |
| Work with an authorized coding agent | [Repository guidance](../AGENTS.md) |
| Determine whether a feature is supported or complete | [Full JPEG XL roadmap](FULL_JPEG_XL_ROADMAP.md) |
| Use the library APIs | [Decoder](../crates/jxl_wgpu_decode/README.md) / [encoder](../crates/jxl_wgpu_encode/README.md) |
| Run capture/replay or measured CLI workloads | [Harness](../tools/jxl_gpu_harness/README.md) |
| Maintain fixtures and independent oracles | [Test support](../tools/jxl_test_support/README.md) / [conformance corpus](CONFORMANCE_CORPUS.md) |

## API contracts

| Question | Owning reference |
|---|---|
| How do complete input, early previews, and main-image selection differ? | [Decoder executable profile](../crates/jxl_wgpu_decode/README.md#executable-profile) |
| When are LF/pass images valid, and when does animation time advance? | [Intermediate LF and pass images](../crates/jxl_wgpu_decode/README.md#intermediate-lf-and-pass-images) |
| What do source precision, numeric channels, alpha, and orientation mean? | [Decoder sample/output contracts](../crates/jxl_wgpu_decode/README.md#integer-source-samples) and the following output sections |
| What does incremental transport retain and validate? | [Bitstream scanner and inventory](../crates/jxl_gpu_bitstream/README.md) |
| Does standard decoding depend on the private acceleration box? | [Lossless Modular profile](../crates/jxl_wgpu_encode/README.md#lossless-modular-profile); `jwgp` is optional |
| What limits the experimental encoder's quality controls? | [Experimental VarDCT profile](../crates/jxl_wgpu_encode/README.md#experimental-vardct-profile) |
| Which GPU handles keep budget ownership? | [Backend buffer leases](../crates/jxl_wgpu/README.md#render-plan-execution) |
| How are custom submissions, unvalidated images, and numeric display handled? | [Same-queue display](../crates/jxl_wgpu/README.md#same-queue-display) |
| How do aggregate readback, direct mapping, and cancellation interact? | [Readback ownership](../crates/jxl_wgpu/README.md#aggregate-cpu-readback) |

## Implementation and evidence

| Topic | Reference |
|---|---|
| Backend boundaries, scheduling, and GPU output | [GPU architecture](GPU_ARCHITECTURE.md) |
| Encoder stages and responsibilities | [Encoder architecture](ENCODER_ARCHITECTURE.md) |
| Shader ABI, binding, layout, and resource lifetime | [WGSL memory](WGSL_MEMORY.md) |
| CPU-oracle/dependency separation and provenance | [Upstream boundary](UPSTREAM_BOUNDARY.md) |
| Fixture families, reproduction, and numerical bounds | [Conformance corpus](CONFORMANCE_CORPUS.md) |
| Measured performance and workload semantics | [GPU benchmarks](GPU_BENCHMARKS.md) |
| Portable formats and precision policies | [VPI format coverage](VPI_FORMAT_COVERAGE.md) / [format APIs](../crates/jxl_gpu_formats/README.md) |
| Opaque metadata ownership, limits, and writing | [Container metadata](CONTAINER_METADATA.md) |
| ICC profiles, device output, and supported connections | [ICC color](ICC_COLOR.md) |
| Explicit luminance adaptation | [Tone mapping](TONE_MAPPING.md) |
| Requested RGB gamut mapping | [Gamut mapping](GAMUT_MAPPING.md) |
| Still gain maps, headroom, and reference white | [Gain maps](GAIN_MAP.md) |

## Documentation ownership

The roadmap owns capability status and acceptance gates. Crate READMEs own public
behavior; topic documents own detailed contracts; the corpus and benchmark records
own test evidence and measured results. Keep the root README as a bounded overview.

Do not duplicate evolving status paragraphs or fixture totals across entry points.
When reorganizing text, keep any unique contract in its owning reference and remove
redundant or obsolete progress reports. Git history provides historical versions;
it is not necessary to keep another full README snapshot in the working tree.
